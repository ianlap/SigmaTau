---
name: kalman-filter-shared
description: >
  Use when building, modifying, or debugging shared Kalman filter math across
  MATLAB and Julia — Q matrix, Φ matrix, PID steering, P covariance update,
  LS state initialization, or the Wu 2023 q↔h canonical convention. Trigger
  when working in matlab/+sigmatau/+kf/ or julia/src/{clock_model,filter,predict,optimize,als_fit}.jl.
---

# Kalman Filter — Shared Math

The math is identical across MATLAB and Julia; only the API shape differs. This
skill is the single source of truth for the math. For language-specific API
shape, see `kalman-filter-julia` or `kalman-filter-matlab`.

**API-shape note.** MATLAB uses struct-config, Julia uses model-type dispatch.
GEMINI.md §7 Goal G6 tracks the MATLAB → model-type migration; until that
lands, parity across the two KF implementations is explicitly *not* a goal
(GEMINI.md §1.5). The math below does have to match; the API does not (yet).

## Clock SDE model

Zucca & Tavella (2005) Eq 1. State vector depends on model order:

- 2-state: `x = (phase, frequency)` — drift deterministic or absent
- 3-state: `x = (phase, frequency, drift)` — canonical
- 5-state diurnal: `x = (phase, frequency, drift, diurnal_sin, diurnal_cos)`

Diffusion coefficients carried in `ClockNoiseParams` (Julia) or the config
struct (MATLAB):

- `q_wpm` — WPM, enters as measurement noise `R = q_wpm` (when the API ties
  them; see language skills for the MATLAB `config.R` vs `config.q_wpm`
  asymmetry)
- `q_wfm` — WFM, state-noise diffusion on frequency
- `q_rwfm` — RWFM, state-noise diffusion on drift
- `q_irwfm` — IRWFM (drift random-walk), state-noise diffusion on drift rate

## Q matrix — exact τ powers

For 3-state (`clock_model.jl:82-97`, `matlab/+sigmatau/+kf/build_Q.m`):

```text
Q11 = q_wfm·τ + q_rwfm·τ³/3 + q_irwfm·τ⁵/20
Q12 = q_rwfm·τ²/2 + q_irwfm·τ⁴/8
Q13 = q_irwfm·τ³/6
Q22 = q_rwfm·τ + q_irwfm·τ³/3
Q23 = q_irwfm·τ²/2
Q33 = q_irwfm·τ
```

These are **exact** continuous-time integrations of the clock SDE, not
approximations. Do not drop the higher-order `τ`-power terms — the τ³/3, τ⁵/20,
τ⁴/8, τ³/6 are load-bearing. GEMINI.md §4a.2, §4b.2.

2-state and diurnal Q are derived from the 3-state block (`build_Q(ClockModel2)`
drops rows/cols; diurnal embeds the 3-state Q in positions [1:3, 1:3] with
`q_diurnal` on the (4,4) and (5,5) diagonals).

## Φ matrix (state transition)

```text
ClockModel2:     [1  τ;       0  1]
ClockModel3:     [1  τ  τ²/2; 0  1  τ; 0  0  1]
ClockModelDiurnal: ClockModel3 in [1:3, 1:3], Identity elsewhere
```

`julia/src/clock_model.jl:57-69`, `matlab/+sigmatau/+kf/build_phi.m`.

## PID steering convention

The integral term accumulates **phase error** directly:

```text
sumx += x[1]
steer = -g_p · x[1] - g_i · sumx - g_d · x[2]   # x[2] term only when nstates >= 2
```

`julia/src/filter.jl:107`, `matlab/+sigmatau/+kf/update_pid.m:5`. GEMINI.md §4.
Do not accumulate frequency error (`x[2]`) in `sumx`, do not drop the minus
signs, do not swap `g_p`/`g_i`. This convention is cross-language load-bearing.

Steering is applied **after** Φ prediction, **before** the measurement update.

## Covariance update

Standard form, not Joseph:

```text
P = (I - K·H) · P
P = (P + P') / 2                  # symmetrize
P[i,i] = safe_sqrt(P[i,i])^2      # re-project diagonals
```

`julia/src/filter.jl:82-87`, `matlab/+sigmatau/+kf/kalman_filter.m:80-86`.
GEMINI.md §4a.3, §4b.3.

`safe_sqrt(x)` absorbs numerical drift on the diagonal:

```text
safe_sqrt(x) = abs(x) < 1e-10 ? 0.0
             : x >= 0         ? sqrt(x)
                              : -sqrt(-x)
```

Preserve this guard — without it, floating-point drift occasionally sends
`P[i,i]` slightly negative and subsequent steps propagate NaN.

## Wu 2023 q↔h canonical convention

SP1065 power-law coefficients `h_α` ↔ clock-SDE diffusions `q`. Canonical source:
`julia/src/clock_model.jl:138-157` (`h_to_q`, `q_to_h`), established in commit
`a4cfdb1`. `f_h = 1 / (2·tau0)` is the Nyquist folding frequency.

| α | Noise | q ← h                   | h ← q                     |
|---|-------|-------------------------|---------------------------|
| +2 | WPM  | `q_wpm  = h₊₂ · f_h / (4π²)` | `h₊₂  = q_wpm · 4π² / f_h` |
|  0 | WFM  | `q_wfm  = h₀ / 2`            | `h₀   = 2 · q_wfm`         |
| −2 | RWFM | `q_rwfm = 2π² · h₋₂`         | `h₋₂  = q_rwfm / (2π²)`    |

**Pre-Wu legacy at `ml/notebook.py:463` is debt, not an alternative
convention.** It has `h₊₂ = q_wpm · 2π² / f_h` (factor of 2 off) and
`h₋₂ = 3·q_rwfm / (2π²)` (factor of 3 off). When that notebook is next
touched, replace the inline `analytical_adev` with a call through `q_to_h`.
FIX_PARKING_LOT.md tracks this.

## LS state initialization

First `min(100, N-1)` samples get fit to a polynomial design matrix
`A = [1, t, t²/2, (sin, cos if diurnal)]` by least squares; coefficients
become `x0`, residual variance times `inv(A'A)` becomes `P0` (unless the
caller passed an explicit `P0`).

`julia/src/filter.jl:127-140`, `matlab/+sigmatau/+kf/kalman_filter.m:151-172`.

Keep the `n_fit >= nstates` guard — fewer samples than states makes the
normal equations singular. `n_fit` clamps to `nstates` from below.

## Diagnostics

For innovation-whiteness / residual-bias checks on a run filter, use
`kf_residual_diagnostics` — `matlab/+sigmatau/+stats/kf_residual_diagnostics.m`
or the Julia `SigmaTau` export. Returns Ljung-Box on raw innovations (D1),
Ljung-Box on normalized innovations (D2, optional), and a naive
`|μ| < 3σ/√N` posterior-residual bias check (D3, coarse since posterior
residuals are autocorrelated by construction).
