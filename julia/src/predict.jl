# predict.jl — Multi-step prediction analysis for Kalman filter
# Refactor of legacy matlab/legacy/kflab/kf_predict.m

# ── Structs ───────────────────────────────────────────────────────────────────

"""
    PredictConfig

Configuration for Kalman filter prediction analysis.

# Fields
- `maturity::Int`: Samples before starting predictions (filter warm-up)
- `max_horizon::Int`: Maximum prediction horizon [samples]
"""
Base.@kwdef struct PredictConfig
    maturity::Int    = 50_000  # samples to warm up filter
    max_horizon::Int = 80_000  # max prediction horizon [samples]
end

"""
    PredictResult

Output of `kf_predict`.

# Fields
- `horizons::Vector{Int}`: Prediction horizons [samples]
- `rms_error::Vector{Float64}`: RMS prediction error at each horizon
- `n_samples::Vector{Int}`: Number of samples averaged per horizon
- `kf_result::KalmanResult`: Full Kalman filter output
- `config::PredictConfig`: Echo of input prediction config
"""
struct PredictResult
    horizons::Vector{Int}
    rms_error::Vector{Float64}
    n_samples::Vector{Int}
    kf_result::KalmanResult
    config::PredictConfig
end

# ── Main function ─────────────────────────────────────────────────────────────

"""
    kf_predict(data, tau, kf_cfg, pred_cfg) -> PredictResult

Run the Kalman filter on `data`, then evaluate multi-step prediction accuracy
from `pred_cfg.maturity` to the end of the data.

For each starting epoch `np ≥ maturity`, predicts `h` steps ahead using the
posterior state and computes RMS prediction errors at all horizons 1…max_horizon.

Prediction formula (legacy kf_predict.m lines 243–251):
- 2-state: `phase(np) + freq(np) * h * tau`
- 3+ state: add `0.5 * drift(np) * (h * tau)^2`

Diurnal states (nstates=5) contribute only to the filtered state used to
initialize the linear/quadratic extrapolation; the extrapolation itself uses
the three kinematic states only.
"""
function kf_predict(data::Vector{Float64}, tau::Float64,
                    kf_cfg::KalmanConfig,
                    pred_cfg::PredictConfig = PredictConfig())
    N = length(data)
    nstates  = kf_cfg.nstates
    maturity = min(pred_cfg.maturity, N - 2)
    max_hor  = min(pred_cfg.max_horizon, N - maturity - 1)
    max_hor < 1 && error("kf_predict: insufficient data — need maturity + max_horizon + 1 ≤ N")

    # Run the Kalman filter on all data — legacy kf_predict.m lines 62-78
    kf_res = kalman_filter(data, kf_cfg)

    phase_h = kf_res.phase_est
    freq_h  = kf_res.freq_est
    drift_h = kf_res.drift_est   # zero if nstates < 3

    # Accumulate squared errors per horizon — legacy kf_predict.m lines 228-263
    var_accum = zeros(Float64, max_hor)
    n_count   = zeros(Int,     max_hor)

    for np in maturity:(N - 1)
        h_max = min(max_hor, N - np)
        x1 = phase_h[np]
        x2 = freq_h[np]
        x3 = nstates >= 3 ? drift_h[np] : 0.0

        for h in 1:h_max
            ht = h * tau  # horizon in time units
            xpred = if nstates == 2
                x1 + x2 * ht
            else  # nstates >= 3 (or 5: kinematic extrapolation only)
                x1 + x2 * ht + 0.5 * x3 * ht^2
            end
            err = data[np + h] - xpred
            var_accum[h] += err * err
            n_count[h]   += 1
        end
    end

    valid     = findall(n_count .> 0)
    horizons  = valid
    rms_error = sqrt.(var_accum[valid] ./ n_count[valid])
    n_samples = n_count[valid]

    return PredictResult(horizons, rms_error, n_samples, kf_res, pred_cfg)
end
