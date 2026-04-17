# Kalman Filter Architecture Spec

**Status:** Draft v2 (2026-04-15)
**Scope:** Redesign of SigmaTau's Julia KF stack to cleanly support the
current clock-estimation use case and extend to future research
directions: EKF, adaptive KF (AKF), ALS-based tuning, smoothers,
square-root forms, and alternative state-space models.

Public API is not preserved. The old `KalmanConfig`/`OptimizeConfig`
entry points are replaced outright once the new API lands; no
deprecation shim.

MATLAB (`matlab/+sigmatau/+kf/`) is **not** migrated as part of this
redesign. It continues to exist as a numerical cross-validation target
(for confirming Julia results match the legacy reference implementation
on canonical inputs). Any new feature lands in Julia only.

This spec is a sibling of [docs/equations/kalman.md](../equations/kalman.md),
which documents the mathematical model. This document covers **software
design**: how the Julia code is organized, what the extension points
are, and how implementation should proceed.

---

## 1. Purpose

### 1.1 Why redesign

The current KF implementation works correctly for the 3-state clock model
but has three structural issues that block extension:

1. **Config bloat, weak types.** `KalmanConfig` holds 14 fields mixing
   clock parameters (`q_wpm`, `q_wfm`, `q_rwfm`, `q_irwfm`, `q_diurnal`,
   `R`), controller gains (`g_p`, `g_i`, `g_d`), state-space choice
   (`nstates`), sampling (`tau`, `period`), and initialization (`P0`,
   `x0`). Nothing prevents inconsistent combinations (e.g., `q_diurnal > 0`
   with `nstates = 3`) at construction time; validation is after-the-fact.

2. **Formulas duplicated.** `build_Q` exists in three places in Julia:
   [julia/src/filter.jl:69-85](../../julia/src/filter.jl#L69-L85),
   [julia/src/optimize.jl:221-238](../../julia/src/optimize.jl#L221-L238),
   and inline in [julia/src/optimize.jl:338-345](../../julia/src/optimize.jl#L338-L345)
   (static-array fast path). Three copies of the same continuous-time
   integration — any correction must be applied three times.

3. **Steering baked into the filter loop.** `kalman_filter` runs
   predict → update → residual → PID → steer as one inseparable unit.
   For pure estimation (NLL evaluation, real-data labeling, offline
   reanalysis), the NLL path in `optimize.jl` had to re-implement a
   steering-free variant. Any future filter variant (EKF, UKF, adaptive)
   would have to re-implement both.

### 1.2 Design principles

- **Julian dispatch, not OOP hierarchy.** Concrete model structs, with
  generic functions (`build_phi`, `build_Q`, `build_H`, `sigma_y_theory`,
  `filter_step!`) dispatched on model type. No abstract base classes.
- **Separation of concerns.** Model (physics) ⊥ filter (algorithm) ⊥
  controller (steering) ⊥ objective (parameter tuning).
- **Single source of truth for each formula.** `build_Q` lives in
  exactly one Julia file; every call site (filter, optimizer,
  theoretical σ_y) goes through it.
- **Clean cut-over, not incremental migration.** When the new API
  lands, the old `KalmanConfig`/`OptimizeConfig` entry points are
  deleted. Internal call sites are updated in the same commit. No
  backward-compatibility shim, no deprecation warnings.
- **MATLAB is a cross-validation target, not a parallel implementation.**
  The existing `matlab/+sigmatau/+kf/` stays as a reference for
  numerical cross-checks on canonical inputs. It is not refactored.

---

## 2. Mathematical foundations

This section fixes the formulas, conventions, and cross-references that
the code must satisfy. The primary sources are:

- **NIST SP1065** (Riley & Howe, 2008) — h_α ↔ σ_y(τ) normative tables
- **Zucca & Tavella 2005** (IEEE UFFC) — 3-state SDE clock model,
  diffusion coefficients ↔ Allan variance
- **Wu 2023** (IEEE T-AES) — 2-state KF canonical form, LTI equivalence
- **Matsakis & Banerjee 2023** — alternate conventions, cross-reference

### 2.1 Clock model (3-state SDE)

Following Zucca-Tavella 2005 Eq 1, the clock phase error obeys

```
dX₁(t) = (X₂(t) + μ₁) dt + σ₁ dW₁(t)     # phase
dX₂(t) = (X₃(t) + μ₂) dt + σ₂ dW₂(t)     # frequency
dX₃(t) =           μ₃ dt + σ₃ dW₃(t)     # drift
```

where W_i are independent Wiener processes, μ_i are deterministic drift
terms, and σ_i² are diffusion coefficients. In the discrete sampled
form (step τ), this integrates to the transition matrix Φ and process
noise covariance Q given in §2.2.

**Measurement model** (Wu Eq 2):

```
z_k = X₁(t_k) + ε_k,    ε_k ~ N(0, R)
```

R is measurement (WPM) variance. SigmaTau's legacy field name `q_wpm`
refers to this R, **not** a state-level WPM. This is consistent with
Zucca-Tavella, Wu, and Kubczak — none of whom put WPM at the state level.

### 2.2 Discrete-time Φ and Q (nstates = 3)

```
       ┌ 1   τ   τ²/2 ┐
Φ  =   │ 0   1    τ   │
       └ 0   0    1   ┘

       ┌ σ₁²τ + σ₂²τ³/3 + σ₃²τ⁵/20    σ₂²τ²/2 + σ₃²τ⁴/8    σ₃²τ³/6 ┐
Q  =   │      symm                    σ₂²τ + σ₃²τ³/3        σ₃²τ²/2 │
       └      symm                         symm              σ₃²τ   ┘
```

With the identifications

- σ₁² = `q_wfm`   (state WFM diffusion)
- σ₂² = `q_rwfm`  (state RWFM diffusion)
- σ₃² = `q_irwfm` (state IRWFM / drift-random-walk diffusion)

**Observation:** H = [1, 0, 0] for phase-only measurements.

**Measurement noise:** R = `q_wpm` (scalar; extend to matrix for
vector observations).

**For nstates = 2:** drop the third row/column of Φ and Q; deterministic
drift `d` becomes an externally provided constant (or estimated separately
via polynomial fit; Wu §IV-B).

### 2.3 Allan variance ↔ clock model bridge

Given (R, σ₁², σ₂²), the theoretical Allan variance is
(Wu Eq 5, Zucca-Tavella Eq 36):

```
σ_y²(τ) = 3·R/τ² + σ₁²/τ + σ₂²·τ/3 + σ₃²·τ³/20
         └───────┘ └──────┘ └────────┘ └─────────┘
          WPM        WFM      RWFM       IRWFM
         (meas.)    (state)  (state)    (state)
```

Drift contributions (deterministic) appear as `(τ²/2)·d²` if the drift
is treated as a deterministic mean rather than a state.

### 2.4 h_α ↔ (R, σ²) conversions

SP1065 Table 3 / Eq. 5 give the Allan variance from spectral coefficients
h_α of S_y(f) = Σ_α h_α·f^α (one-sided PSD, f ∈ [0, f_h]):

| α  | Noise | σ_y²(τ) contribution |
|----|-------|----------------------|
| +2 | WPM   | 3·f_h·h₊₂ / (4π²·τ²)      (f_h·τ ≫ 1) |
|  0 | WFM   | h₀ / (2τ)                               |
| −2 | RWFM  | (2π²/3)·h₋₂·τ                           |

For sampled data with τ₀ sampling interval, f_h = 1/(2τ₀) (Nyquist).

Matching against §2.3:

```
R      = h₊₂ · f_h / (4π²)   =  h₊₂ / (8π²·τ₀)    ← WPM ↔ R
σ₁²    = h₀  / 2                                    ← WFM
σ₂²    = (2π² / 3) · h₋₂ · 3 = 2π² · h₋₂            ← RWFM
```

**Known bug (Tier 1 fix):** [julia/src/optimize.jl:452](../../julia/src/optimize.jl#L452)
currently computes `q_wpm = h₊₂·f_h/(2π²)`, which is 2× the correct R.
This is only a warm-start guess, so Nelder-Mead converges anyway, but
the new `h_to_q` API (§4.2) must use the correct formula
`R = h₊₂·f_h/(4π²)`.

### 2.5 Steady-state Kalman gain (Wu §III)

In steady state, Wu 2023 shows the KF is equivalent to 3 LTI systems
whose transfer functions are completely determined by the Kalman gain
coefficients K_s1, K_s2 (2-state) or K_s1, K_s2, K_s3 (3-state). These
come from the discrete algebraic Riccati equation (DARE):

```
P_∞ = solve_dare(Φ, H, Q, R)
K_∞ = P_∞ H^T / (H P_∞ H^T + R)
```

SigmaTau already computes `_dare_scalar_H` internally in
[julia/src/optimize.jl:259-307](../../julia/src/optimize.jl#L259-L307); the spec
promotes it to a public API (§4.3).

---

## 3. Current state audit

```
julia/src/
├── filter.jl       KalmanConfig (14 fields), KalmanResult, kalman_filter,
│                   build_phi!, build_Q!, update_pid!,
│                   _build_design_matrix, _initialize_state!
├── optimize.jl     OptimizeConfig, OptimizeResult, optimize_kf,
│                   _kf_nll, _kf_nll_static, _build_Q (duplicate #2),
│                   _build_A, _dare_scalar_H, _dare_scalar_H_static,
│                   optimize_kf_nll (h-warm-start wrapper)
├── predict.jl      PredictConfig, PredictResult, kf_predict
└── noise_fit.jl    mhdev_fit (legacy MHDEV-based noise-ID estimator)
```

MATLAB (`matlab/+sigmatau/+kf/`) has an equivalent set of files and is
used as a cross-validation reference. It is not touched by this spec.

### 3.1 Pain points (prioritized)

| # | Issue | Blocker for | Tier |
|---|-------|-------------|------|
| 1 | `q_wpm` scattered as `q_wpm` (KF+NLL fields) and `R` (KF-only field); no shared type | ML-label harmonization, ALS integration | 1 |
| 2 | `build_Q` in 4 places; any correction needs 4 edits | Correctness, audit trail | 1 |
| 3 | `h_to_q` only exists inline (with known 2× bug on h₊₂) | h-label ML pivot, user-facing analysis | 1 |
| 4 | `sigma_y_theory` doesn't exist at all | Validation plots, round-trip tests | 1 |
| 5 | Steering hard-coded in filter loop | Any future filter variant, cleaner NLL | 2 |
| 6 | No model abstraction — nstates switches encode physics | EKF, AKF, alternative state dimensions | 2 |
| 7 | No public steady-state gain accessor | Wu §III LTI analysis, diagnostics | 2 |
| 8 | No ALS estimator | Liu-style peer-reviewed Q/R labels | 2 |
| 9 | Result type stores P_history unconditionally | Memory cost on long runs; square-root forms need different shape | 3 |
| 10 | No smoother (RTS backward pass) | Improved offline estimates, retrospective analysis | 3 |

---

## 4. Object model

Julia side, with MATLAB mirror below. The design is **concrete types
with dispatched generic functions**, not abstract-class hierarchies.

### 4.1 Core type: `ClockNoiseParams`

```julia
"""
    ClockNoiseParams

The 3-to-4 noise diffusion coefficients of the canonical clock SDE
(Zucca-Tavella 2005, Eq 1). WPM enters as measurement noise R;
WFM/RWFM/IRWFM enter as state-noise diffusions.
"""
Base.@kwdef struct ClockNoiseParams
    q_wpm::Float64                  # R = σ₀² (WPM, measurement)
    q_wfm::Float64                  # σ₁² (WFM, state)
    q_rwfm::Float64      = 0.0      # σ₂² (RWFM, state); 0 to disable
    q_irwfm::Float64     = 0.0      # σ₃² (IRWFM / drift RW, state)
end
```

This replaces the `q_wpm, q_wfm, q_rwfm, q_irwfm, R` quintet scattered
across `KalmanConfig` and `OptimizeConfig`. It is the single label for
ML pipelines, the single argument to `sigma_y_theory`, and the output
of every estimator (`mhdev_fit`, `optimize_kf_nll`, future ALS).

### 4.2 Clock-model structs (dispatched functions)

```julia
"""
    ClockModel3

3-state linear-Gaussian clock model: (phase, frequency, drift).
Canonical case; matches Zucca-Tavella 2005 Eq 1 with σ₃² enabled.
"""
Base.@kwdef struct ClockModel3
    noise::ClockNoiseParams
    tau::Float64
end

"""
    ClockModel2

2-state: (phase, frequency). Drift is deterministic or absent.
"""
Base.@kwdef struct ClockModel2
    noise::ClockNoiseParams         # q_irwfm ignored
    tau::Float64
end

"""
    ClockModelDiurnal

5-state: (phase, frequency, drift, diurnal-sin, diurnal-cos).
"""
Base.@kwdef struct ClockModelDiurnal
    noise::ClockNoiseParams
    tau::Float64
    period::Float64       = 86400.0
    q_diurnal::Float64    = 0.0
end
```

Dispatched generic functions (one method per model):

```julia
build_phi(m::ClockModel2)        :: Matrix{Float64}          # 2×2
build_phi(m::ClockModel3)        :: Matrix{Float64}          # 3×3
build_phi(m::ClockModelDiurnal)  :: Matrix{Float64}          # 5×5

build_Q(m::ClockModel2)          :: Matrix{Float64}
build_Q(m::ClockModel3)          :: Matrix{Float64}
build_Q(m::ClockModelDiurnal)    :: Matrix{Float64}

build_H(m::ClockModel2)          :: Matrix{Float64}          # 1×2
build_H(m::ClockModel3)          :: Matrix{Float64}          # 1×3
build_H(m::ClockModelDiurnal, k) :: Matrix{Float64}          # 1×5, k-dependent

nstates(m::ClockModel2)          = 2
nstates(m::ClockModel3)          = 3
nstates(m::ClockModelDiurnal)    = 5

sigma_y_theory(m, tau)           :: Float64    # Wu Eq 5 / ZT Eq 36
```

`build_Q(m::ClockModel3)` is the **one and only** place the
continuous-integration formulas live. All existing call sites
(`kalman_filter`, `_kf_nll`, `_kf_nll_static`) call through it.

### 4.3 Steady-state accessors (Wu §III)

```julia
"""
    steady_state_covariance(model) -> Matrix{Float64}

Solve DARE for the posterior covariance P∞. Wraps `_dare_scalar_H`.
"""
steady_state_covariance(m::ClockModel3) :: Matrix{Float64}

"""
    steady_state_gain(model) -> Vector{Float64}

Scalar-H steady-state Kalman gain vector K∞ (Wu §III Ks1, Ks2, Ks3).
"""
steady_state_gain(m::ClockModel3) :: Vector{Float64}
```

### 4.4 h ↔ q conversion

```julia
"""
    h_to_q(h::NamedTuple, tau0::Float64) -> ClockNoiseParams

Map SP1065 power-law coefficients h_α to clock-SDE diffusion coefficients.
Formulas: §2.4 of this spec.

- h.h2  → q_wpm  = h2 * f_h / (4π²)   where f_h = 1 / (2·tau0)
- h.h0  → q_wfm  = h0 / 2
- h.h_2 → q_rwfm = 2π² · h_2

Missing fields default to 0.
"""
h_to_q(h, tau0) :: ClockNoiseParams

"""
    q_to_h(q::ClockNoiseParams, tau0) -> NamedTuple

Inverse mapping. Returns (h2=, h0=, h_2=). Flicker h₋₁, h₊₁ are not
derivable from (R, q_wfm, q_rwfm) alone — return as missing / 0.
"""
q_to_h(q, tau0) :: NamedTuple
```

### 4.5 Filter: pure estimation primitive

```julia
"""
    FilterState

Carries the running mean, covariance, and optional diagnostics for a
single filter step. Square-root forms replace P with U, D factors
(Tier 3).
"""
Base.@kwdef struct FilterState
    x::Vector{Float64}              # state mean
    P::Matrix{Float64}              # state covariance
    k::Int                 = 0      # step index
end

"""
    filter_step!(state, model, z_k) -> (state, innovation, innovation_variance)

One KF predict+update cycle. Pure estimation — no steering, no control
side-effects. The model provides Φ, Q, H via dispatched accessors.
Mutates `state` in place; returns diagnostic scalars.
"""
filter_step!(s::FilterState, m::ClockModel3, z_k) -> (s, ν, S)
```

Higher-level loops that need steering call `filter_step!` inside and
apply PID externally. This puts the control layer outside the
estimation layer, where it belongs.

### 4.6 High-level filter wrapper (unchanged API)

```julia
"""
    kalman_filter(data, model; steering=nothing, init=AutoLS()) -> KalmanResult

Run the filter over a time series. If `steering::PIDController` is
provided, apply it after each update. `init` controls state
initialization.
"""
kalman_filter(data, model; kwargs...) :: KalmanResult
```

The existing `kalman_filter(data, cfg::KalmanConfig)` signature becomes
a thin wrapper: build `model` from `cfg`, forward. Zero breakage.

### 4.7 Objectives (parameter tuning)

```julia
"""
    innovation_nll(data, model) -> Float64

Innovation negative log-likelihood:
    NLL = 0.5 · Σ_k [ log S_k + ν_k² / S_k ]
"""
innovation_nll(data, m::ClockModel3) :: Float64

"""
    optimize_nll(data, noise_init, tau0; kwargs...) -> ClockNoiseParams

Nelder-Mead in log10 space over (q_wpm, q_wfm, q_rwfm). Returns the
optimal `ClockNoiseParams`. Optional `h_init::NamedTuple` warm-starts
via `h_to_q`.
"""
optimize_nll(...) :: ClockNoiseParams
```

Future objectives slot in without touching the filter code:

```julia
als_fit(data, model_template) :: ClockNoiseParams    # Åkesson 2008 Eq 23
```

### 4.8 MATLAB

Out of scope for this redesign. The existing `matlab/+sigmatau/+kf/`
files stay in place as a numerical reference. Cross-validation workflow:
given the same (noise params, seed), Julia and MATLAB Φ, Q, and one-step
innovation should agree to machine precision. Any divergence is a
Julia-side regression to investigate.

---

## 5. Implementation tiers

### Tier 1 — Foundational (implement first)

Clean cut-over of the formula surface. ~300 LOC Julia. Old
`KalmanConfig`/`OptimizeConfig` entry points are replaced, not
shimmed.

1. **`julia/src/clock_model.jl`** (new):
   - `ClockNoiseParams` struct
   - `ClockModel2`, `ClockModel3`, `ClockModelDiurnal` structs
   - Dispatched `build_phi`, `build_Q`, `build_H`, `nstates`
   - `sigma_y_theory(model, tau)`
   - `h_to_q`, `q_to_h` (with the corrected h₊₂ formula from §2.4)
   - `steady_state_covariance`, `steady_state_gain`
2. **Deduplicate**: delete `build_Q!` in `filter.jl`, `_build_Q` in
   `optimize.jl`, and the inline build in `_kf_nll_static`. All three
   call sites go through `build_Q(model)`. Keep the static-array
   specialization as a performance overload dispatched on
   `ClockModel3` — same formulas, generated from the same source.
3. **Rewrite `filter.jl` and `optimize.jl` around the new types**.
   `KalmanConfig`/`OptimizeConfig` removed. `kalman_filter(data,
   model; kwargs...)` is the only signature.
4. **Update all call sites in-repo** in the same commit: ML dataset
   generators, `kf_predict`, `optimize_kf_nll`, tests. No stale
   references after the commit lands.
5. **Export new API**: `ClockNoiseParams`, `ClockModel{2,3,Diurnal}`,
   `build_phi`, `build_Q`, `build_H`, `sigma_y_theory`, `h_to_q`,
   `q_to_h`, `steady_state_covariance`, `steady_state_gain` added to
   `SigmaTau.jl` exports. Old exports removed.
6. **Tests**:
   - `test_clock_model.jl`: build_Q consistency across nstates=2/3,
     sigma_y_theory round-trips via h_to_q ∘ q_to_h, DARE convergence,
     agreement with MATLAB reference Φ/Q on canonical seeds
   - Rewrite existing `test_filter.jl` to use the new API; keep the
     behavior assertions (residual zero-mean, covariance convergence).

### Tier 2 — Separation of concerns

~400 LOC Julia. Each item is independent; do in any order.

1. **`filter_step!` primitive**: extract pure predict+update from
   `kalman_filter`'s inner loop. `kalman_filter` calls it and adds
   steering / bookkeeping as an outer layer.
2. **Steering as external type**: `PIDController` struct with its own
   `step!(controller, state) -> steer` method. `kalman_filter`
   receives it as an optional kwarg.
3. **Public `innovation_nll(data, model)`**: replaces private
   `_kf_nll`. `optimize_nll` becomes a thin wrapper over
   `innovation_nll` + Nelder-Mead.
4. **ALS estimator**: `als_fit(data, ClockModel3; lags=30)` implementing
   Åkesson 2008 Eq 23 (unconstrained linear LS on autocovariance).
   Complements `optimize_nll` and `mhdev_fit`; returns
   `ClockNoiseParams`.
5. **ML pipeline alignment**: ML dataset h-labels flow through
   `h_to_q` to produce target `ClockNoiseParams`; real-data validation
   converts predicted `ClockNoiseParams` back through `sigma_y_theory`
   for overlay plots.

### Tier 3 — Extension points (design only; implement when needed)

Document the contracts; don't implement until a driving use case
appears. Each of these is a thin layer on top of Tiers 1–2 and does
not require touching existing code.

#### 3.1 Extended Kalman Filter (EKF)

For nonlinear dynamics or observation (e.g., phase-wrap tracking,
nonlinear PLL coupling, joint state-parameter estimation):

```julia
struct NonlinearClockModel
    f::Function          # (x, t, tau) -> x_next
    h::Function          # (x, t) -> z
    F_jac::Function      # tangent of f at x
    H_jac::Function      # tangent of h at x
    Q::Matrix{Float64}
    R::Float64
end

filter_step!(s::FilterState, m::NonlinearClockModel, z_k) -> ...
```

`filter_step!` dispatches on `NonlinearClockModel` and uses Jacobians.
No change to linear path.

#### 3.2 Adaptive Kalman Filter (AKF)

For time-varying Q/R (temperature drift, aging, regime changes):

```julia
struct AdaptiveClockModel
    base::ClockModel3
    window::Int                  # innovation window for adaptation
    adapt::Symbol                # :mehra_1970, :sage_husa, :online_als
end
```

The adaptive layer maintains a running innovation-covariance estimate
and re-solves for Q/R each `window` steps. The base model's `build_Q`
is called with updated `ClockNoiseParams` at each adaptation point.

#### 3.3 RTS smoother

A backward pass over a filter trajectory:

```julia
rts_smooth(result::KalmanResult, model) -> SmoothedResult
```

Requires `KalmanResult` to record predicted state/covariance (already
does via `P_history`). No new model machinery needed.

#### 3.4 Square-root / UD forms

For numerical stability on poorly conditioned problems:

```julia
struct FilterStateUD
    x::Vector{Float64}
    U::Matrix{Float64}          # unit upper triangular
    d::Vector{Float64}          # diagonal factors
end

filter_step!(s::FilterStateUD, m, z_k) -> ...
```

Dispatch on the state type; the model is the same.

#### 3.5 Unscented / sigma-point

Same dispatch pattern: `SigmaPointState <: FilterState` with
`filter_step!(::SigmaPointState, ::NonlinearModel, z)` using the
unscented transform.

---

## 6. Replacement plan

### 6.1 One commit per tier

Tier 1 is a single atomic commit that:

1. Adds `julia/src/clock_model.jl`.
2. Rewrites `filter.jl`, `optimize.jl`, `predict.jl` around the new
   types.
3. Deletes `KalmanConfig`, `OptimizeConfig`, `_build_Q`, `build_Q!`,
   `_build_design_matrix`.
4. Updates all in-repo call sites (ML drivers, notebook, tests).
5. Updates exports in `SigmaTau.jl`.
6. Lands new and rewritten tests in `julia/test/`.

After this commit, there is exactly one KF API. No legacy path.

Tier 2 items land as independent commits on top.

### 6.2 Before-after sketch

```julia
# Before:
cfg = KalmanConfig(q_wfm=1e-22, q_rwfm=1e-30, R=1e-18, nstates=3, tau=1.0)
kalman_filter(data, cfg)

# After:
noise = ClockNoiseParams(q_wpm=1e-18, q_wfm=1e-22, q_rwfm=1e-30)
model = ClockModel3(noise=noise, tau=1.0)
kalman_filter(data, model)
```

### 6.3 Test harness

Existing `julia/test/test_filter.jl` is rewritten to use the new API.
Behavioral assertions (residual zero-mean, covariance convergence,
prediction RMS) are preserved.

New test files:

```
test_clock_model.jl       # §5 Tier 1 tests
test_filter_step.jl       # §5 Tier 2 item 1
test_als_fit.jl           # §5 Tier 2 item 4 (when implemented)
```

Each estimator test must include a round-trip:

1. Generate synthetic data with known `ClockNoiseParams`.
2. Run the estimator (`optimize_nll` / `als_fit` / `mhdev_fit`).
3. Assert recovered params match within expected tolerance.

### 6.4 MATLAB cross-validation

A `julia/test/test_matlab_parity.jl` (optional; gated by a MATLAB
availability check) exercises canonical seeds in both languages and
asserts agreement to ~1e-12 relative error on Φ, Q, one-step
innovation, and 100-step filter output. This guards against Julia
regressions without requiring the MATLAB code to evolve.

---

## 7. Known corrections to apply during refactor

1. **h₊₂ → R formula.** Current code at
   [julia/src/optimize.jl:452](../../julia/src/optimize.jl#L452):
   ```julia
   q_wpm0 = h_init[2.0] * f_h / (2π^2)      # WRONG — 2× too large
   ```
   Correct (per §2.4):
   ```julia
   q_wpm0 = h_init[2.0] * f_h / (4π^2)      # = h₊₂ / (8π²·τ₀)
   ```
   The new `h_to_q` uses the correct formula. The old warm-start is
   replaced by calling `h_to_q`; same code path after refactor.

2. **`KalmanConfig` redundancy.** `q_wpm` and `R` are separate fields
   with a comment "R = q_wpm typically." Users can set them
   independently, which is a footgun. Gone in the new API: only
   `ClockNoiseParams.q_wpm` exists; filter code reads it as R at a
   single point.

3. **`q_diurnal` with nstates≠5**: currently raises a validation
   warning at runtime. In the new API, `ClockModelDiurnal` is a
   separate type and the combination is structurally impossible.

---

## 8. Out of scope (explicitly)

- **Deep-learning / CNN-based state estimation.** Not a KF variant;
  separate project.
- **Joint state-parameter filtering (dual KF, JPDA).** Possible future
  extension via EKF-augmented state, but not prescribed here.
- **Distributed/ensemble Kalman filter.** Different problem domain
  (DA, spatial).
- **Particle filters.** Not relevant for linear-Gaussian clock models.

---

## 9. References

- NIST SP1065 §5, Tables 3–4 — h_α ↔ σ_y(τ)
  ([docs/papers/reference/sp1065.pdf](../papers/reference/sp1065.pdf))
- Zucca & Tavella 2005, IEEE UFFC — 3-state SDE clock model
  ([docs/papers/state_estimation/2005_zucca_tavella_clock_model_allan_ieee_uffc.pdf](../papers/state_estimation/2005_zucca_tavella_clock_model_allan_ieee_uffc.pdf))
- Wu 2023, IEEE T-AES — KF ≡ LTI in steady state, Eqs 5, 12
  ([docs/papers/state_estimation/2023_wu_kf_performance_lti_atomic_clock_ieee_taes.pdf](../papers/state_estimation/2023_wu_kf_performance_lti_atomic_clock_ieee_taes.pdf))
- Kubczak et al. 2019, IEEE SPA — dimensional sweep for Rb KF
  ([docs/papers/state_estimation/2019_kubczak_kf_fast_sync_rubidium_ieee_spa.pdf](../papers/state_estimation/2019_kubczak_kf_fast_sync_rubidium_ieee_spa.pdf))
- Liu et al. 2024, Sensors — adaptive KF + ALS for Rb
  ([docs/papers/state_estimation/2024_liu_adaptive_kf_rubidium_sensors.pdf](../papers/state_estimation/2024_liu_adaptive_kf_rubidium_sensors.pdf))
- Åkesson et al. 2008, J. Process Control — generalized ALS
  ([docs/papers/state_estimation/2008_akesson_generalized_als_kf_tuning_jprocontrol.pdf](../papers/state_estimation/2008_akesson_generalized_als_kf_tuning_jprocontrol.pdf))
- Matsakis & Banerjee 2023, Ch. 13 — KF conventions cross-reference
  ([docs/papers/reference/2023_banerjee_matsakis_timekeeping_book.pdf](../papers/reference/2023_banerjee_matsakis_timekeeping_book.pdf))
- [docs/equations/kalman.md](../equations/kalman.md) — equation reference (companion to this spec)
