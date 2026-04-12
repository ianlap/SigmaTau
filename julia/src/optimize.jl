# optimize.jl — Q-parameter grid search for Kalman filter
# Refactor of legacy matlab/legacy/kflab/optimize_kf.m

# ── Structs ───────────────────────────────────────────────────────────────────

"""
    OptimizeConfig

Configuration for Kalman filter Q-parameter optimization.

# Fields
- `search_range::Int`: Search ±N decades around initial values (default: 2)
- `n_grid_per_decade::Int`: Grid points per decade (default: 5)
- `nstates::Int`: Number of KF states — 2, 3, or 5 (default: 3)
- `target_horizons::Vector{Int}`: Prediction horizons to minimize [samples]
- `horizon_weights::Vector{Float64}`: Weight per horizon (empty → equal)
- `maturity::Int`: Filter warm-up before evaluating predictions
- `period::Float64`: Diurnal period [s] (only used when nstates=5)
"""
Base.@kwdef struct OptimizeConfig
    search_range::Int            = 2
    n_grid_per_decade::Int       = 5
    nstates::Int                 = 3
    target_horizons::Vector{Int} = [10, 100, 1000]
    horizon_weights::Vector{Float64} = Float64[]   # empty → equal weights
    maturity::Int                = 50_000
    period::Float64              = 86400.0
end

"""
    OptimizeResult

Output of `kf_optimize`.

# Fields
- `q_wpm::Float64`: Fixed WPM noise variance (held constant during optimization)
- `q_wfm::Float64`: Optimal WFM noise variance
- `q_rwfm::Float64`: Optimal RWFM noise variance
- `q_irwfm::Float64`: Optimal IRWFM noise variance
- `rms_opt::Vector{Float64}`: RMS error at each target horizon for optimal params
- `weighted_rms::Float64`: Weighted cost at optimal params
- `search_history::Matrix{Float64}`: All evaluated params + cost, columns = [q_wpm q_wfm q_rwfm q_irwfm cost]
- `n_evaluations::Int`: Total grid evaluations
"""
struct OptimizeResult
    q_wpm::Float64
    q_wfm::Float64
    q_rwfm::Float64
    q_irwfm::Float64
    rms_opt::Vector{Float64}
    weighted_rms::Float64
    search_history::Matrix{Float64}
    n_evaluations::Int
end

# ── Internal helpers ───────────────────────────────────────────────────────────

# Build a KalmanConfig for a given noise parameter set (no PID — optimization only)
# Legacy optimize_kf.m lines 220-223
function _opt_kf_config(tau::Float64, q_wpm::Float64, q_wfm::Float64, q_rwfm::Float64,
                        q_irwfm::Float64, nstates::Int, period::Float64)
    return KalmanConfig(
        q_wpm     = q_wpm,
        q_wfm     = q_wfm,
        q_rwfm    = q_rwfm,
        q_irwfm   = q_irwfm,
        q_diurnal = 0.0,
        R         = q_wpm,
        g_p       = 0.0,   # no PID for optimization — legacy optimize_kf.m line 220
        g_i       = 0.0,
        g_d       = 0.0,
        nstates   = nstates,
        tau       = tau,
        P0        = 1e30,
        period    = period,
    )
end

# Evaluate the weighted RMS cost for one set of Q parameters
# Legacy optimize_kf.m lines 206-250
function _eval_cost(data::Vector{Float64}, tau::Float64,
                    q_wpm::Float64, q_wfm::Float64, q_rwfm::Float64, q_irwfm::Float64,
                    target_horizons::Vector{Int}, weights::Vector{Float64},
                    maturity::Int, nstates::Int, period::Float64)
    HIGH_PENALTY = 1e10
    nH = length(target_horizons)
    rms_vals = fill(HIGH_PENALTY, nH)

    try
        kf_cfg   = _opt_kf_config(tau, q_wpm, q_wfm, q_rwfm, q_irwfm, nstates, period)
        pred_cfg = PredictConfig(maturity = maturity,
                                 max_horizon = maximum(target_horizons))
        pr = kf_predict(data, tau, kf_cfg, pred_cfg)

        for (k, h) in enumerate(target_horizons)
            idx = findfirst(==(h), pr.horizons)
            idx !== nothing && (rms_vals[k] = pr.rms_error[idx])
        end
    catch
        # Keep penalty values
    end

    cost = sum(weights .* rms_vals)
    return cost, rms_vals
end

# ── Main function ─────────────────────────────────────────────────────────────

"""
    kf_optimize(data, tau, q_wpm, q_wfm0, q_rwfm0, cfg) -> OptimizeResult

Find optimal Kalman filter Q parameters via logarithmic grid search.

WPM (`q_wpm`) is held fixed (measurement noise). WFM and RWFM are searched
over `cfg.search_range` decades in each direction from initial guesses `q_wfm0`
and `q_rwfm0`. If `q_irwfm0 > 0` and `cfg.nstates >= 3`, IRWFM is also searched.

The cost function is the weighted RMS prediction error at `cfg.target_horizons`.
Grid search uses `2 * search_range * n_grid_per_decade + 1` log-spaced points
per dimension — legacy optimize_kf.m lines 137-161.
"""
function kf_optimize(data::Vector{Float64}, tau::Float64,
                     q_wpm::Float64, q_wfm0::Float64, q_rwfm0::Float64,
                     cfg::OptimizeConfig = OptimizeConfig();
                     q_irwfm0::Float64 = 0.0)
    nH      = length(cfg.target_horizons)
    weights = isempty(cfg.horizon_weights) ? fill(1.0 / nH, nH) :
              cfg.horizon_weights ./ sum(cfg.horizon_weights)
    length(weights) == nH ||
        error("kf_optimize: horizon_weights length $(length(weights)) ≠ target_horizons length $nH")

    # Build log-spaced grid — legacy optimize_kf.m lines 137-161
    nPts   = 2 * cfg.search_range * cfg.n_grid_per_decade + 1
    factor = 10.0^cfg.search_range

    g_wfm  = exp10.(range(log10(q_wfm0  / factor), log10(q_wfm0  * factor); length = nPts))
    g_rwfm = exp10.(range(log10(q_rwfm0 / factor), log10(q_rwfm0 * factor); length = nPts))

    search_irwfm = cfg.nstates >= 3 && q_irwfm0 > 0.0
    g_irwfm = search_irwfm ?
        exp10.(range(log10(q_irwfm0 / factor), log10(q_irwfm0 * factor); length = nPts)) :
        [0.0]

    # Enumerate all grid points
    params = Vector{NTuple{4, Float64}}()
    for qw in g_wfm, qr in g_rwfm, qi in g_irwfm
        push!(params, (q_wpm, qw, qr, qi))
    end
    nTotal = length(params)

    costs    = zeros(Float64, nTotal)
    rms_all  = zeros(Float64, nTotal, nH)

    for k in 1:nTotal
        q1, q2, q3, q4 = params[k]
        costs[k], rms_all[k, :] = _eval_cost(data, tau, q1, q2, q3, q4,
                                               cfg.target_horizons, weights,
                                               cfg.maturity, cfg.nstates, cfg.period)
    end

    # Find best — legacy optimize_kf.m lines 179-183
    best_idx = argmin(costs)
    q1b, q2b, q3b, q4b = params[best_idx]

    # Build search history matrix: [q_wpm q_wfm q_rwfm q_irwfm cost] (nTotal × 5)
    history_mat = Matrix{Float64}(undef, nTotal, 5)
    for k in 1:nTotal
        history_mat[k, 1] = params[k][1]
        history_mat[k, 2] = params[k][2]
        history_mat[k, 3] = params[k][3]
        history_mat[k, 4] = params[k][4]
        history_mat[k, 5] = costs[k]
    end

    return OptimizeResult(q1b, q2b, q3b, q4b,
                          rms_all[best_idx, :],
                          costs[best_idx],
                          history_mat,
                          nTotal)
end
