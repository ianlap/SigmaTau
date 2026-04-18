---
name: kalman-filter-julia
description: >
  Use when building, modifying, or debugging Julia Kalman filter code —
  type-dispatch API with ClockModel{2,3,Diurnal}, ClockNoiseParams,
  optimize_nll, als_fit, predict_holdover, innovation_nll, or the
  OptimizeNLLResult return struct. Trigger when working in
  julia/src/{clock_model,filter,predict,optimize,als_fit}.jl.
---

# Kalman Filter — Julia (type-dispatch API)

Julia KF is type-dispatch on `ClockModel{2,3,Diurnal}` carrying a
`ClockNoiseParams`. No config struct. For the cross-language math (Q matrix,
Φ matrix, PID, P update, Wu 2023 q↔h) see `kalman-filter-shared`.

## Type-dispatch API

```julia
kalman_filter(data::Vector{Float64}, model;
              x0 = Float64[], P0 = 1e6,
              g_p = 0.1, g_i = 0.01, g_d = 0.05) -> KalmanResult
```

`model` ∈ {`ClockModel2`, `ClockModel3`, `ClockModelDiurnal`}, each carrying a
`noise::ClockNoiseParams` and a `tau::Float64`. Method dispatches on the model
type — there is no `nstates` keyword (read it off the model via
`nstates(model)`).

`julia/src/filter.jl:149` is the canonical entry point. Also exported as
`kf_filter` (`SigmaTau.jl:48,75`).

## Model constructors

```julia
ClockNoiseParams(; q_wpm, q_wfm, q_rwfm=0.0, q_irwfm=0.0)
ClockModel2(noise, tau)
ClockModel3(noise, tau)
ClockModelDiurnal(noise, tau, period=86400.0, q_diurnal=0.0)
```

`nstates(::ClockModel2) == 2`, `ClockModel3 == 3`, `ClockModelDiurnal == 5`.

## KalmanResult fields

```julia
struct KalmanResult
    phase_est   ::Vector{Float64}
    freq_est    ::Vector{Float64}
    drift_est   ::Vector{Float64}     # zeros for 2-state
    residuals   ::Vector{Float64}
    innovations ::Vector{Float64}
    steers      ::Vector{Float64}
    sumsteers   ::Vector{Float64}
    sum2steers  ::Vector{Float64}
    P_history   ::Array{Float64, 3}   # ns × ns × N (dense 3-D; MATLAB uses cell)
    model                              # echo back
end
```

Julia's `P_history` is a dense 3-D array indexed `P_history[:, :, k]`. MATLAB
stores it as `cell(N,1)` indexed `P_history{k}(i,j)` — distinct from Julia's
shape.

## optimize_nll — Nelder-Mead on innovation NLL

```julia
optimize_nll(data, tau0;
             h_init           = nothing,
             noise_init       = nothing,
             optimize_qwpm    = false,       # Fix 2 default
             optimize_irwfm   = false,
             verbose          = true,
             max_iter         = 500,
             tol              = 1e-6) -> OptimizeNLLResult
```

Fits Zucca-Tavella diffusion parameters by minimizing the Gaussian innovation
NLL on a log10-space simplex. Seeding:

- `h_init` (SP1065 power-law coefficients): routed through `h_to_q` (Wu 2023
  convention; see `kalman-filter-shared`) to produce a `ClockNoiseParams` seed.
- `noise_init` (a `ClockNoiseParams`): used directly.
- Neither: fallback `ClockNoiseParams(q_wpm=1e-26, q_wfm=1e-25, q_rwfm=1e-26)`.

`optimize_qwpm=false` is the default post-commits `1d6b49b` (optimize_nll) and
`32f42e1` (als_fit). Matches MATLAB `sigmatau.kf.optimize` and the textbook KF
tuning problem where `R` is known from short-τ ADEV or calibration. Pass
`optimize_qwpm=true` to sweep `R` jointly; this is the "hyperparameter sweep"
case, not the canonical tune.

## OptimizeNLLResult

```julia
struct OptimizeNLLResult
    noise     :: ClockNoiseParams
    nll       :: Float64
    n_evals   :: Int
    converged :: Bool   # std(fvals) < tol fired before max_iter
end
```

Added in commit `f06d346`. `converged` unblocks honest dataset-level
diagnostics (commit `e7728f6` replaced a hardwired `true` in
`ml/dataset/generate_dataset.jl` with `opt_res.converged`).

## als_fit — Autocovariance Least Squares (Åkesson 2008 / Odelson 2006)

```julia
als_fit(data, tau0;
        h_init         = nothing,
        noise_init     = nothing,
        optimize_qwpm  = false,    # symmetric with optimize_nll
        optimize_irwfm = false,
        lags           = 30,
        burn_in        = 50,
        max_iter       = 5,
        verbose        = true) -> ClockNoiseParams
```

Runs the KF once, extracts the innovation sequence (after `burn_in`), computes
empirical autocovariance out to `lags`, and solves a non-negative LS problem on
log10 parameters for each outer iteration. Convergence test: relative change in
`q_wfm` < 1e-3.

## predict_holdover — state propagation only

```julia
predict_holdover(x0, P0, model, horizon) -> HoldoverResult
predict_holdover(kf::KalmanResult, horizon; model=kf.model) -> HoldoverResult
```

Pure KF prediction (no measurement updates). Propagates `x ← Φ·x`, `P ← Φ·P·Φ'
+ Q` for `horizon` steps. The convenience method extracts final state from a
`KalmanResult` — errors for `ClockModelDiurnal` because `KalmanResult` doesn't
carry the diurnal states; use the raw `(x0, P0, model, horizon)` form instead.

## innovation_nll — scalar-H fast path

`julia/src/optimize.jl:89-133` has a `ClockModel3`-specialized method using
`StaticArrays` (`@SMatrix`, `@SVector`) and the scalar-H shortcut (`S =
P[1,1] + R`, `K = P[:,1] / S`) — significantly faster than the generic
method for 3-state. The generic method (`optimize.jl:135-177`) handles 2-state
and diurnal.

## Composition — full pipeline

```julia
h = Dict(2.0 => 1e-18, 0.0 => 1e-21, -2.0 => 1e-30)
tau0 = 1.0
opt = optimize_nll(phase, tau0; h_init=h)   # OptimizeNLLResult
model = ClockModel3(noise=opt.noise, tau=tau0)
kf    = kalman_filter(phase, model)          # KalmanResult
hold  = predict_holdover(kf, 3600)           # HoldoverResult (1 hour)
```

`scratch_holdover.jl` at the repo root is the only in-tree example that
exercises this chain end-to-end (AUDIT_01 §3 flagged this reachability gap;
not fixed in this session).
