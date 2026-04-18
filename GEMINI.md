# SigmaTau — Foundational Mandates

This document contains absolute mandates for the SigmaTau project. It takes precedence over tool defaults and general-purpose workflows, and is overridden only by explicit user instructions.

**Scope tags.** Each mandate carries a scope tag. `[stability]` mandates govern the deviation/noise-ID half (`julia/src/{engine,noise,deviations}*.jl`, `matlab/+sigmatau/+{dev,noise,stats}/`). `[kf-julia]` governs `julia/src/{clock_model,filter,predict,optimize,als_fit}.jl`. `[kf-matlab]` governs `matlab/+sigmatau/+kf/`. `[repo]` applies to both halves. Per AUDIT_02 §3, stability and KF halves have zero code coupling today; mandates are scoped accordingly and parity across languages is not a general goal (see §1.5).

**Verification tag.** Each mandate ends with a trailing italicized line of the form `_[Verified YYYY-MM-DD — check: …]_`. The check is the action a future session can run to confirm the mandate still describes reality. Tags for high-stakes mandates (cross-validation tolerance, function length, NEFF_RELIABLE, Plots extension status) include reproducible evidence (commands, file:line, and expected output). Simple mandates get short checks.

**Hard rules enforced in this file.** No mandate is stated as current fact when the code disagrees — aspirations live in §7 Goals, not in mandates. No mandate exists without a verification tag. Institutional-memory notes (e.g. the totdev denominator) are preserved verbatim.

---

## 1. Project Goals

### 1.1 ML pipeline purpose
The near-term product driver is the PH 551 final: predict KF noise parameters (`q_wpm`, `q_wfm`, `q_rwfm`) from a σ(τ) deviation curve on a 10k-sample synthetic dataset, using a random-forest / XGBoost pipeline. The stability-analysis half supplies the features (feature vectors from `julia/src/ml_features.jl`); the KF half supplies the labels. Dataset generation lives in `ml/dataset/*.jl`.

### 1.2 Canonical noise-parameter convention
The Wu (2023) h↔q mapping is canonical (`julia/src/clock_model.jl:138-157`: `h_to_q`, `q_to_h`). The pre-Wu formulas at `julia/test/test_filter.jl:186` (factor-of-3 off on `q_rwfm`) and `scripts/python/plot_kf.py:54-59` (MHDEV component coefficients) are legacy debt, not alternative conventions; `FIX_PARKING_LOT.md:8` tracks the audit.

### 1.3 Observability constraint on the clock model
`ClockNoiseParams` (`julia/src/clock_model.jl:13-18`) carries four noise coefficients: `q_wpm`, `q_wfm`, `q_rwfm`, `q_irwfm`. On currently available datasets, drift random-walk (`q_irwfm`, aka σ₃²) is unobservable — fits collapse into a flat NLL valley. Ian fits three parameters (WPM / WFM / RWFM). Do not propose fitting `q_irwfm` without explicit user sign-off; adding the 4th state burns weeks of label spread.

### 1.4 Packaging — one package, revisit post-PH-551
AUDIT_02 documented zero code-level coupling between the stability half and the KF half, but that is a *descriptive* finding. The product decision is to ship as a single `SigmaTau` package (Julia + MATLAB) for now, with a package-split question deferred to post-PH-551. Do not propose splitting `SigmaTau.jl` into `SigmaTau` + `SigmaTauKF` without explicit sign-off.

### 1.5 KF paradigm asymmetry (Julia vs MATLAB)
Julia KF uses type-dispatch on `ClockModel{2,3,Diurnal}` with noise carried in `ClockNoiseParams`. MATLAB KF uses the pre-refactor `config` struct / `results` struct paradigm and has not been migrated. **Parity between the two KF implementations is not a goal.** No KF cross-validation test exists across languages (see §5). Treat `matlab/+sigmatau/+kf/` as frozen at the struct-config paradigm.

---

## 2. Deviation Engine `[stability]`

### 2.1 Shared engine
All 10 deviations (`adev`, `mdev`, `tdev`, `hdev`, `mhdev`, `ldev`, `totdev`, `mtotdev`, `htotdev`, `mhtotdev`) MUST dispatch through the shared engine — `julia/src/engine.jl` in Julia, `matlab/+sigmatau/+dev/engine.m` in MATLAB. Wrappers supply a kernel plus a `DevParams`/params struct; they do not duplicate engine boilerplate.
_[Verified 2026-04-16 — check: `engine.jl:30` defines `engine(...)`; `engine.m:1` defines `function result = engine(...)`; all 10 Julia wrappers live in `julia/src/deviations/{allan,hadamard,total}.jl` (verified via grep for `^function (adev|mdev|…)`); all 10 MATLAB wrappers are at `matlab/+sigmatau/+dev/{adev,mdev,tdev,hdev,mhdev,ldev,totdev,mtotdev,htotdev,mhtotdev}.m`.]_

### 2.2 Kernel interface — Julia (4-arg)
Julia kernels MUST have signature `kernel(x, m, tau0, x_cs) → (variance::Float64, neff::Int)`, where `x_cs = cumsum([0; x])` is the precomputed prefix-sum vector. The engine computes `x_cs` once per call and passes it to every kernel for O(N) sharing.
_[Verified 2026-04-16 — check: `julia/src/engine.jl:63` computes `x_cs`; `:68` calls `kernel(x, m, tau0, x_cs)`.]_

### 2.3 Kernel interface — MATLAB (3-arg, not yet migrated)
MATLAB kernels currently use `kernel(x, m, tau0) → [variance, neff]`. Each kernel that needs prefix sums recomputes its own `cumsum([0; x(:)])`. This is a deliberate pre-migration state; the CHANGELOG claim of a shared 4-arg contract is Julia-only. See Goal G2 (§7) for the migration target.
_[Verified 2026-04-16 — check: `matlab/+sigmatau/+dev/engine.m:72` calls `kernel(x, m, tau0)` (3-arg); AUDIT_02 §5 notes 5 of 10 MATLAB kernels recompute their own cumsum.]_

### 2.4 Variance return
Kernels MUST return *variance* (σ²), not deviation (σ). The engine takes `sqrt(max(var_val, 0))` with a small-negative-variance clamp (rounding cancellation at `-eps*N`).
_[Verified 2026-04-16 — check: `julia/src/engine.jl:68,83` (`var_val, n = kernel(...)`, `sqrt(max(var_val, 0.0))`); `matlab/+sigmatau/+dev/engine.m:72,87` equivalent.]_

### 2.5 Data types — phase/frequency
Wrappers MUST accept both phase and frequency data via a `data_type` keyword (Julia) or name-value argument (MATLAB). Frequency-to-phase conversion (`cumsum(y) * tau0`) MUST live in the engine, not in wrappers.
_[Verified 2026-04-16 — check: `julia/src/engine.jl:40-44` (`if data_type === :freq; x = cumsum(x) .* tau0`); `matlab/+sigmatau/+dev/engine.m:38-42` (`if strcmpi(data_type, 'freq'); x = cumsum(x(:)) * tau0`).]_

### 2.6 O(N) complexity for mdev / mhdev
The modified-family kernels (`mdev`, `mhdev`, and the modified-total variants) MUST use cumsum-based prefix sums for O(N) complexity per averaging factor. Do NOT use O(N·m) nested loops.
_[Verified 2026-04-16 — check: Julia shares one `x_cs` via the engine (`engine.jl:63`); MATLAB kernels use local `cumsum([0; x(:)])`. Both satisfy the O(N) literal claim; engine-level sharing is a Julia-only optimization (Goal G2).]_

### 2.7 htotdev at m=1
`htotdev(m=1)` MUST use the overlapping HDEV formula, not the total-deviation reflection algorithm. This is the canonical SP1065 fallback; the reflection algorithm is ill-defined at m=1.
_[Verified 2026-04-16 — check: `docs/equations/total.md:58` states the rule; `julia/src/deviations/total.jl:184,217` branches on `m == 1`; `matlab/+sigmatau/+dev/htotdev.m:12,31,36` equivalent.]_

### 2.8 Bias-correction tables
Bias-correction tables for `totvar`, `mtotvar`, and `htotvar` MUST match NIST SP1065 (Riley & Howe) exactly. Deviations from the tabulated coefficients without a new citation are regressions.
_[Verified 2026-04-16 — check: SP1065 at `docs/papers/reference/sp1065.pdf` §5.2; tables in `julia/src/stats.jl` and `matlab/+sigmatau/+stats/`. Spot-check against SP1065 tables 8–10.]_

### 2.9 Totdev denominator (institutional memory — preserve verbatim)
**Totdev MUST use `2(N-2)(mτ₀)²` per SP1065 Eq 25 for phase form (equivalently `2(M-1)` for frequency form, `M = N-1`). Do not change to `N-1`.**
_[Verified 2026-04-14 — check: SP1065 Eq 25 reviewed against `julia/src/deviations/total.jl` and `matlab/+sigmatau/+dev/totdev.m`; denominator matches `2τ²(N-2)` in phase form. Original CLAUDE.md note dated 2026-04-14; preserved verbatim here.]_

### 2.10 Known constraint — mhtotdev EDF coefficients
The mhtotdev EDF coefficients are approximate (Monte-Carlo fit; no published closed-form model). This is the current state, not a binding rule. `TODO.md` High-Priority row "MHTOTDEV EDF Model" tracks refinement. Future sessions should not be surprised by the approximate fit.

---

## 3. Noise Identification `[stability]`

### 3.1 Dual-path identification
Noise ID MUST implement the SP1065 §5.6 dual-path dispatch: use the lag-1 ACF estimator when `N_eff` is large enough for reliability, and fall back to B₁ ratio + R(n) lookup otherwise. When both return NaN, carry forward the most recent reliable α (Stable32 convention).
_[Verified 2026-04-16 — check: `julia/src/noise.jl:34-40` (`if N_eff >= NEFF_RELIABLE; _noise_id_lag1acf; else; …`); `matlab/+sigmatau/+noise/noise_id.m:38-40` equivalent dispatch.]_

### 3.2 Known constraint — NEFF_RELIABLE threshold (mandate cites SP1065 floor; code uses empirical 50)
SP1065 §5.6 cites an N_eff floor of 30 for the lag-1 ACF estimator. The code currently uses 50 with an inline empirical rationale ("the ACF estimator is still high-variance just above [30]"). This is a deliberate product choice, not unmigrated drift. Resolution is tracked as Goal G5 (§7); the direction of reconciliation is deliberately open.
_[Verified 2026-04-16 — check: `julia/src/noise.jl:20-24` (`const NEFF_RELIABLE = 50`, with SP1065 §5.6 and empirical rationale in the preceding comment); `matlab/+sigmatau/+noise/noise_id.m:25-28` (`NEFF_RELIABLE = 50` with matching comment).]_

---

## 4. Kalman Filter

**Shared rule across languages:** the KF integrator MUST consume a noise-parameter container — `ClockNoiseParams` in Julia, the `config` struct in MATLAB — not inline scalar values for `q_wpm`/`q_wfm`/`q_rwfm`. The container must come from a fit (`mhdev_fit`, `optimize_nll`, `als_fit`) or a calibrated prior, not a hardcoded literal.
_[Verified 2026-04-16 — check: `julia/src/filter.jl:149` takes `(data, model)` where `model` carries `model.noise::ClockNoiseParams`; `matlab/+sigmatau/+kf/kalman_filter.m:1` takes `(data, config)` struct.]_

**PID convention (cross-language) — `[kf-julia] [kf-matlab]`.** The integral term MUST accumulate phase error directly: `sumx += x[1]` at each step. Do not accumulate frequency error or use an alternative PID convention.
_[Verified 2026-04-16 — check: `julia/src/filter.jl:107` (`c.sumx += x[1]`); `matlab/+sigmatau/+kf/update_pid.m:5` (`pid_state(1) = pid_state(1) + x(1);`).]_

### 4a. Julia KF `[kf-julia]`

#### 4a.1 Type-dispatch API
The Julia KF API is type-dispatch on `ClockModel2 | ClockModel3 | ClockModelDiurnal`, not struct-config. `kalman_filter(data, model; kwargs)` is the canonical entry point.
_[Verified 2026-04-16 — check: `julia/src/filter.jl:149` (`function kalman_filter(data::Vector{Float64}, model; x0, P0, g_p, g_i, g_d)`); `julia/src/clock_model.jl:26,36,46` define the three models.]_

#### 4a.2 Q-matrix exact τ powers
Process-noise Q-matrix elements MUST follow exact continuous-time integration of the clock SDE. For `ClockModel3` with WFM/RWFM/IRWFM: `Q11 = q_wfm·τ + q_rwfm·τ³/3 + q_irwfm·τ⁵/20`, and the off-diagonals as specified in Zucca–Tavella (2005) Eq 1. No approximations.
_[Verified 2026-04-16 — check: `julia/src/clock_model.jl:82-97` (`build_Q(ClockModel3)`); `julia/src/optimize.jl:103-110` (3-state Q with IRWFM).]_

#### 4a.3 Standard covariance update
Use `P = (I - K·H)·P` (standard form), symmetrize, then re-project diagonals with `safe_sqrt(·)²` to absorb numerical drift.
_[Verified 2026-04-16 — check: `julia/src/filter.jl:82-87` (`Pm = (I - K * H) * Pm; Pm = (Pm + Pm') ./ 2; Pm[i,i] = safe_sqrt(Pm[i,i])^2`).]_

### 4b. MATLAB KF `[kf-matlab]` (struct-config paradigm — frozen)

#### 4b.1 Struct-in / struct-out
MATLAB KF functions take a `config` struct and return a `results` struct. No positional noise-parameter arguments. This paradigm is intentionally frozen pre-Julia-refactor; see §1.5 for why parity is not a goal.
_[Verified 2026-04-16 — check: `matlab/+sigmatau/+kf/kalman_filter.m:1` (`function result = kalman_filter(data, config)`); `matlab/+sigmatau/+kf/optimize.m:1` same pattern.]_

#### 4b.2 Q-matrix exact τ powers
Matches §4a.2. Implementation in `matlab/+sigmatau/+kf/build_Q.m`. No independent cross-check across languages (Goal G3).
_[Verified 2026-04-16 — check: `matlab/+sigmatau/+kf/build_Q.m` matches the τ, τ³/3, τ⁵/20 pattern.]_

#### 4b.3 Standard covariance update
Matches §4a.3 — `P = (I - K*H) * P` plus `safe_sqrt` on diagonals.
_[Verified 2026-04-16 — check: `matlab/+sigmatau/+kf/kalman_filter.m:85,126` (`safe_sqrt` usage); covariance update uses the standard form.]_

---

## 5. Cross-validation & Accuracy

### 5.1 Stability-deviation cross-validation (scope-limited)
MATLAB ↔ Julia deviation point estimates MUST agree within `REL_TOL = 2e-10`. The test covers the 10 deviation *point estimates only* — not EDF, not CI, not any KF output. The test silently returns (without failing) when the Julia reference file `crossval_results.txt` is missing; this is a known gap.
_[Verified 2026-04-16 — check: `matlab/tests/test_crossval_julia.m:47` sets `REL_TOL = 2e-10`; `:10-14` issues a warning and returns when `crossval_results.txt` is absent; `:34-45` lists the 10 deviation handles; `:62-80` compares `result.deviation` only (no `edf`/`ci`/KF fields). The prior "< 10⁻¹²" / "< 10⁻¹⁰ across all output fields" claim was aspirational and has been removed.]_

### 5.2 No KF cross-validation across languages
No test compares MATLAB and Julia Kalman-filter outputs. Goal G3 (§7) tracks adding one.
_[Verified 2026-04-16 — check: `matlab/tests/` contains no file matching `*kf*crossval*`; `julia/test/` contains no file comparing against MATLAB KF output.]_

---

## 6. Code Style & Standards `[repo]`

### 6.1 Function length
Target: ≤100 lines per function. **Six current files exceed 100 total lines** and are tracked as Goal G4 (§7) for splitting:

| File | Lines |
|------|------:|
| `matlab/+sigmatau/+kf/optimize.m` | 215 |
| `matlab/+sigmatau/+kf/kalman_filter.m` | 208 |
| `matlab/+sigmatau/+noise/noise_id.m` | 207 |
| `matlab/+sigmatau/+dev/engine.m` | 163 |
| `matlab/+sigmatau/+stats/ci.m` | 109 |
| `matlab/+sigmatau/+stats/calculate_edf.m` | 106 |

**Caveat.** These are *file* lengths (`wc -l`), not top-level-function lengths. A proper function-length audit would require AST parsing (MATLAB function-end detection), which has not been done. Three of the six files (`noise_id.m`, `ci.m`, `calculate_edf.m`) may actually be compliant at the function level if their top-level functions are short and the bulk is in nested local functions. Until that audit runs, both the file-length count and the function-level compliance status for those three are open.
_[Verified 2026-04-16 — check: `wc -l matlab/+sigmatau/+kf/optimize.m matlab/+sigmatau/+kf/kalman_filter.m matlab/+sigmatau/+noise/noise_id.m matlab/+sigmatau/+dev/engine.m matlab/+sigmatau/+stats/ci.m matlab/+sigmatau/+stats/calculate_edf.m` → 215/208/207/163/109/106. No AST-based function-length audit has been performed.]_

### 6.2 Equation citations in non-trivial blocks
Non-trivial numerical blocks SHOULD cite the source equation (`% SP1065 Eq. 12`, `# Greenhall (2003) Eq. 8`). This is a style standard; spot-check verified (e.g. `julia/src/deviations/total.jl`, `julia/src/noise.jl`), not exhaustively enforced.
_[Verified 2026-04-16 — check: spot-checks pass; some files (e.g. `julia/src/optimize.jl:180-183` per AUDIT_01 §8) have thin docstrings — enforcement is aspirational.]_

### 6.3 Named constants over magic numbers
Numerical constants SHOULD have named bindings (e.g. `CONFIDENCE_DEFAULT = 0.683` in `julia/src/engine.jl:5`, `_P0_SCALE`/`_NM_*` in `julia/src/optimize.jl`, `NEFF_RELIABLE` in `julia/src/noise.jl:24`). Spot-check verified; not exhaustively enforced.
_[Verified 2026-04-16 — check: sampled named constants in engine.jl:5, noise.jl:24, optimize.jl; enforcement is aspirational.]_

### 6.4 MATLAB namespace
All MATLAB library code MUST reside under the `+sigmatau/` package namespace.
_[Verified 2026-04-16 — check: `matlab/+sigmatau/` contains `+dev`, `+kf`, `+noise`, `+stats`, `+util`, `+plot` (empty), `+steering` (empty); no loose `.m` files in `matlab/`.]_

---

## 7. Goals (not mandates)

These are tracked aspirations. They are *not* binding rules — failing a goal is not a regression.

- **G1 — Plots.jl as package extension.** Move `Plots` from `[deps]` to `[weakdeps]` in `julia/Project.toml`, add an `[extensions]` entry, populate `julia/ext/`, and bump `julia` compat to 1.9+. Tracked in `TODO.md` Medium Priority ("Julia Plotting Extensions").
  _[Verified 2026-04-16 — check: `julia/Project.toml:10` lists `Plots` in `[deps]`; no `[weakdeps]` or `[extensions]` sections present; `ls julia/ext/` shows an empty directory.]_

- **G2 — MATLAB kernels migrate to 4-arg contract.** Change the MATLAB kernel signature to `kernel(x, m, tau0, x_cs)` to match the Julia engine's O(N) prefix-sum sharing. Completes the CHANGELOG claim of a unified kernel contract. Scope: `matlab/+sigmatau/+dev/engine.m` and the 5 kernels that currently recompute their own cumsum.
  _[Verified 2026-04-16 — check: `matlab/+sigmatau/+dev/engine.m:72` still calls `kernel(x, m, tau0)` (3-arg); AUDIT_02 §5 notes 5 of 10 MATLAB kernels currently recompute their own `cumsum([0; x(:)])`. Goal is open until engine dispatches `x_cs` and kernels accept it.]_

- **G3 — KF cross-validation across languages.** Add a test that runs the same phase data through Julia (`kalman_filter`) and MATLAB (`sigmatau.kf.kalman_filter`) and compares the `phase_est`, `freq_est`, and `drift_est` trajectories within a quantified tolerance. Requires first reconciling the two paradigms (type-dispatch vs struct-config) or settling on a thin adapter.

- **G4 — Oversized MATLAB functions split.** Split the six over-100-line files listed in §6.1. AST-based function-length audit should run first so the three possibly-compliant files (`noise_id.m`, `ci.m`, `calculate_edf.m`) can be excluded from refactor if their top-level functions are already short.

- **G5 — Resolve NEFF_RELIABLE discrepancy (direction open).** The mandate cites SP1065 §5.5.6 with a floor of 30; the code uses 50 with an inline empirical rationale (`julia/src/noise.jl:20-23`, `matlab/+sigmatau/+noise/noise_id.m:25-27`). Resolution options:
  (i) accept 50 as the product choice and amend SP1065 references with the empirical justification;
  (ii) migrate code to 30, accepting higher long-τ variance in the lag-1 ACF estimator;
  (iii) measure ACF estimator variance over the 30–50 range on realistic datasets and pick the defensible point.
  Resolution requires either a new measurement or an explicit product decision. Deferred until post-PH-551. Do not silently migrate the constant to 30 without sign-off.
  _[Verified 2026-04-16 — check: `julia/src/noise.jl:24` (`const NEFF_RELIABLE = 50`), `matlab/+sigmatau/+noise/noise_id.m:28` (`NEFF_RELIABLE = 50`); inline empirical comments at `julia/src/noise.jl:20-23` and `matlab/+sigmatau/+noise/noise_id.m:25-27`; SP1065 §5.6 cited floor is 30.]_

- **G6 — MATLAB KF migration to model-type shape.** MATLAB currently uses struct-config for the Kalman filter (`sigmatau.kf.kalman_filter(data, cfg)` with a flat 14-field struct). Julia uses model-type dispatch (`kalman_filter(data, ::ClockModel3)` with `ClockNoiseParams` nested inside). Unified-design decision: MATLAB moves to match Julia. Implementation: create MATLAB classes `ClockModel2`, `ClockModel3`, `ClockModelDiurnal` with `ClockNoiseParams` as a class or methodized struct. Preserve math and tests. Scope: `matlab/+sigmatau/+kf/` only. Estimated effort: ~1 week. Deferred until post-PH-551. Do not partial-migrate.
  _[Verified 2026-04-17 — check: `matlab/+sigmatau/+kf/kalman_filter.m:1` takes flat `(data, config)` struct; `julia/src/filter.jl:149` takes `(data, ::ClockModel; kwargs)`. AUDIT_02 §9 documents the shape divergence. Decision locked 2026-04-17 based on user's "unified design, two languages" requirement.]_

---

## 8. References

- **NIST SP1065** — Riley & Howe, *Handbook of Frequency Stability Analysis*. Local copy: `docs/papers/reference/sp1065.pdf`. Normative for σ(τ) equations, bias-correction tables, EDF formulas, and noise-ID thresholds.
- **Greenhall & Riley (2003)** — "Uncertainty of Stability Variances," PTTI. EDF derivations for modified/total families.
- **IEEE Std 1139-2022** — Standard Definitions of Physical Quantities for Fundamental Frequency and Time Metrology.
- **Matsakis & Banerjee (2023)** — *Timekeeping* (book). Local copy: `docs/papers/reference/2023_banerjee_matsakis_timekeeping_book.pdf`. Conceptual / derivation tiebreaker for stability and KF equations. Matsakis was Ian's PhD advisor; treat as authoritative for disambiguation.
- **Wu (2023)** — "KF Performance for an LTI Atomic Clock," IEEE TAES. Local copy: `docs/papers/state_estimation/2023_wu_kf_performance_lti_atomic_clock_ieee_taes.pdf`. Canonical source for the h↔q parameterization used in `clock_model.jl`.
