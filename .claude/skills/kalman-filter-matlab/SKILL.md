---
name: kalman-filter-matlab
description: >
  Use when building, modifying, or debugging MATLAB Kalman filter code —
  struct-config API with sigmatau.kf.kalman_filter(data, cfg) and
  sigmatau.kf.optimize(data, cfg). Trigger when working in matlab/+sigmatau/+kf/.
---

# Kalman Filter — MATLAB (struct-config API)

MATLAB KF uses a struct-config paradigm (`sigmatau.kf.kalman_filter(data, cfg)`
with a flat 14-field `cfg`). For the cross-language math (Q matrix, Φ matrix,
PID, P update, Wu 2023 q↔h) see `kalman-filter-shared`.

> **Pending migration — G6.** MATLAB will move to Julia's model-type shape
> (`ClockModel{2,3,Diurnal}` + `ClockNoiseParams`) per GEMINI.md §7 Goal G6.
> This skill describes the **current** API for work in current code. Do **not**
> speculatively pre-structure new MATLAB KF code for the migration — that's a
> full ~1-week rewrite on its own branch. Maintain current shape until the
> migration's explicit start. GEMINI.md §1.5 confirms cross-language parity is
> not a current goal.

## Config struct (14 fields)

```matlab
cfg = struct( ...
    'q_wpm',    0.0,     ... % WPM — used by sigmatau.kf.optimize as R only
    'q_wfm',    1e-22,   ... % WFM diffusion
    'q_rwfm',   1e-30,   ... % RWFM diffusion
    'q_irwfm',  0.0,     ... % IRWFM diffusion (optional)
    'q_diurnal', 0.0,    ... % diurnal diffusion (optional; requires nstates=5)
    'R',        1e-22,   ... % measurement noise — used by sigmatau.kf.kalman_filter
    'g_p',      0.1,     ... % PID proportional gain
    'g_i',      0.01,    ... % PID integral gain
    'g_d',      0.05,    ... % PID derivative gain
    'nstates',  3,       ... % 2, 3, or 5
    'tau',      1.0,     ... % sampling interval [s]
    'P0',       1e6,     ... % initial covariance (scalar scales Identity, or full matrix)
    'x0',       [],      ... % initial state ([] triggers LS init on first 100 samples)
    'period',   86400);    % diurnal period (only read when nstates=5)
```

Defaults in `apply_defaults` at `matlab/+sigmatau/+kf/kalman_filter.m:174-189`.

## R vs q_wpm — know which function uses which

`kalman_filter.m:21` reads `R = config.R` (independent field).
`optimize.m:136` reads `R = cfg.q_wpm` (tied to the WPM diffusion).

Implication: a config that works in `kalman_filter` may not be valid for
`optimize` (if `q_wpm = 0`) and vice-versa. The test suite's `test_filter.m`
Test 5 works around this by populating `q_wpm = config.R` in a local copy
before the optimize call (commit `52d12db`).

This asymmetry is captured in AUDIT_02 §6 and resolves itself when G6 unifies
the shape.

## Results struct

```matlab
result.phase_est     % N × 1
result.freq_est      % N × 1
result.drift_est     % N × 1 (zeros for nstates=2)
result.residuals     % N × 1
result.innovations   % N × 1
result.steers        % N × 1
result.sumsteers     % N × 1
result.sum2steers    % N × 1
result.P_history     % cell(N, 1), each entry is (nstates × nstates)
result.config        % echo of input (post-apply_defaults)
```

**`P_history` is a cell array, not a 3-D matrix.** Indexing is
`result.P_history{k}(i,j)`, **not** `result.P_history(i,j,k)`. The 3-D shape
is Julia's; `test_filter.m` Test 3 carried a Julia-style indexing bug that
was only caught after Test 2 was xfail'd (commit `52d12db` fixed it).

## sigmatau.kf.optimize(data, cfg)

```matlab
[q_opt, results] = sigmatau.kf.optimize(data, cfg)
```

- Required cfg fields: `tau`, `q_wpm`, `q_wfm`, `q_rwfm` (all `> 0`). `q_irwfm`
  optional; 0 disables IRWFM dimension.
- Algorithm: `fminsearch` (Nelder-Mead) in log10-space on `[q_wfm, q_rwfm]`
  (plus `q_irwfm` if enabled).
- `q_wpm` is held fixed as `R`. This default matches the textbook tuning
  problem where `R` is known from short-τ ADEV or calibration. Julia's
  `optimize_nll` matches this default post-Fix 2 (commit `1d6b49b`).
- `results` has `.nll, .n_evals, .exitflag, .elapsed`.

## Helper functions

All under `matlab/+sigmatau/+kf/`:

- `build_phi(ns, tau)` — Φ matrix
- `build_Q(ns, q_wfm, q_rwfm, q_irwfm, q_diurnal, tau)` — Q matrix
- `predict_state(x, Phi, last_steer, tau, ns)` — one Φ·x step + steering correction
- `predict_covariance(P, Phi, Q)` — `Phi*P*Phi' + Q`
- `update_pid(pid_state, x, ns, g_p, g_i, g_d)` — PID step; `pid_state = [sumx; last_steer]`

For the math behind each, see `kalman-filter-shared`.

## Known asymmetries inside MATLAB

- **nstates range.** `sigmatau.kf.kalman_filter` supports `nstates ∈ {2,3,5}`
  (5 is diurnal). `sigmatau.kf.optimize` supports only `{2,3}`. Passing `5`
  to the optimizer errors out. AUDIT_01 parking lot noted this; not fixed.

- **No MATLAB equivalent** for Julia's `predict_holdover`, `als_fit`,
  `innovation_nll` (as a public function — there is a private `kf_nll` inside
  `optimize.m:126`). These live Julia-side and have no MATLAB counterpart.

- **No KF cross-validation vs Julia exists.** GEMINI.md §5.2 + Goal G3.
  Adding one requires first resolving the API-shape asymmetry or settling on
  a thin adapter.
