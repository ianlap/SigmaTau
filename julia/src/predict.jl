# predict.jl — Holdover prediction via Kalman filter state propagation

using LinearAlgebra

"""
    HoldoverResult

Output of `predict_holdover`. Contains predicted state trajectories and
covariance history over the prediction horizon.

# Fields
- `phase_pred` — predicted phase at each horizon step
- `freq_pred`  — predicted frequency
- `drift_pred` — predicted drift (zeros for 2-state models)
- `P_pred`     — ns x ns x horizon covariance history
- `model`      — the clock model used for propagation
"""
struct HoldoverResult{M<:AbstractStateModel}
    phase_pred::Vector{Float64}
    freq_pred::Vector{Float64}
    drift_pred::Vector{Float64}
    P_pred::Array{Float64, 3}
    model::M
end

"""
    predict_holdover(x0, P0, model, horizon) -> HoldoverResult

Propagate state `x0` with covariance `P0` forward `horizon` steps under `model`.
Pure Kalman prediction (no updates). Works for ClockModel2, ClockModel3, and
ClockModelDiurnal.

# Arguments
- `x0::Vector{Float64}` — initial state vector (length = nstates(model))
- `P0::Matrix{Float64}` — initial covariance (nstates x nstates)
- `model` — a ClockModel2, ClockModel3, or ClockModelDiurnal
- `horizon::Int` — number of prediction steps
"""
function predict_holdover(x0::Vector{Float64}, P0::Matrix{Float64},
                          model::AbstractStateModel, horizon::Int)
    horizon >= 1 || error("predict_holdover: horizon must be >= 1")
    ns = nstates(model)
    length(x0) == ns || error("predict_holdover: x0 length $(length(x0)) != nstates $ns")
    size(P0) == (ns, ns) || error("predict_holdover: P0 size $(size(P0)) != ($ns, $ns)")

    Phi = build_phi(model)
    Q   = build_Q(model)

    phase_pred = zeros(Float64, horizon)
    freq_pred  = zeros(Float64, horizon)
    drift_pred = zeros(Float64, horizon)
    P_pred     = Array{Float64, 3}(undef, ns, ns, horizon)

    x = copy(x0)
    P = copy(P0)

    for i in 1:horizon
        x = Phi * x
        P = Phi * P * Phi' + Q
        P = 0.5 .* (P .+ P')  # enforce symmetry

        phase_pred[i] = x[1]
        freq_pred[i]  = ns >= 2 ? x[2] : 0.0
        drift_pred[i] = ns >= 3 ? x[3] : 0.0
        P_pred[:, :, i] .= P
    end

    return HoldoverResult(phase_pred, freq_pred, drift_pred, P_pred, model)
end

"""
    predict_holdover(kf::KalmanResult, horizon; model=kf.model) -> HoldoverResult

Convenience method: extract the final state and covariance from a `KalmanResult`
and propagate forward. The model defaults to `kf.model` but can be overridden.

Note: for ClockModelDiurnal (5-state), the full state cannot be reconstructed
from KalmanResult fields alone. Use the raw `(x0, P0, model, horizon)` method
for diurnal models.
"""
function predict_holdover(kf::KalmanResult, horizon::Int;
                          model = kf.model)
    ns = nstates(model)

    if ns > 3
        error("predict_holdover(KalmanResult, ...): KalmanResult does not store " *
              "states beyond (phase, freq, drift). For $(ns)-state models, use " *
              "predict_holdover(x0, P0, model, horizon) directly.")
    end

    N = length(kf.phase_est)
    x0 = if ns == 2
        Float64[kf.phase_est[N], kf.freq_est[N]]
    else
        Float64[kf.phase_est[N], kf.freq_est[N], kf.drift_est[N]]
    end
    P0 = Matrix{Float64}(kf.P_history[:, :, N])

    return predict_holdover(x0, P0, model, horizon)
end
