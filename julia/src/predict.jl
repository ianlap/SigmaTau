# predict.jl — Multi-step prediction analysis for Kalman filter

"""
    PredictConfig

Configuration for Kalman filter prediction analysis.
"""
Base.@kwdef struct PredictConfig
    maturity::Int    = 50_000  # samples to warm up filter
    max_horizon::Int = 80_000  # max prediction horizon [samples]
end

"""
    PredictResult

Output of `kf_predict`.
"""
struct PredictResult
    horizons::Vector{Int}
    rms_error::Vector{Float64}
    n_samples::Vector{Int}
    kf_result::KalmanResult
    config::PredictConfig
end

"""
    kf_predict(data, model, pred_cfg = PredictConfig()) -> PredictResult

Run the Kalman filter on `data`, then evaluate multi-step prediction accuracy.
"""
function kf_predict(data::Vector{Float64}, model,
                    pred_cfg::PredictConfig = PredictConfig())
    N = length(data)
    ns = nstates(model)
    tau = model.tau

    maturity = min(pred_cfg.maturity, N - 2)
    max_hor  = min(pred_cfg.max_horizon, N - maturity - 1)
    max_hor < 1 && error("kf_predict: insufficient data")

    kf_res = kalman_filter(data, model)

    phase_h = kf_res.phase_est
    freq_h  = kf_res.freq_est
    drift_h = kf_res.drift_est

    var_accum = zeros(Float64, max_hor)
    n_count   = zeros(Int,     max_hor)

    for np in maturity:(N - 1)
        h_max = min(max_hor, N - np)
        x1 = phase_h[np]
        x2 = freq_h[np]
        x3 = ns >= 3 ? drift_h[np] : 0.0

        for h in 1:h_max
            ht = h * tau
            xpred = if ns == 2
                x1 + x2 * ht
            else
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
