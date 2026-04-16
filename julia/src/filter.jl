# filter.jl — Kalman filter for frequency stability / clock steering

using LinearAlgebra
using Statistics

"""
    KalmanResult

Output of `kalman_filter`. `P_history[k]` is the posterior covariance at step k.
"""
struct KalmanResult
    phase_est::Vector{Float64}
    freq_est::Vector{Float64}
    drift_est::Vector{Float64}
    residuals::Vector{Float64}
    innovations::Vector{Float64}
    steers::Vector{Float64}
    sumsteers::Vector{Float64}
    sum2steers::Vector{Float64}
    P_history::Array{Float64, 3}
    model::Union{ClockModel2, ClockModel3, ClockModelDiurnal}
end

# ── Internal helpers ───────────────────────────────────────────────────────────

safe_sqrt(x::Float64) = abs(x) < 1e-10 ? 0.0 : x >= 0.0 ? sqrt(x) : -sqrt(-x)

# ── Separation of Concerns: Tier 2 ───────────────────────────────────────────

"""
    FilterState

Carries the running mean, covariance, and optional diagnostics for a single filter step.
"""
Base.@kwdef mutable struct FilterState
    x::Vector{Float64}
    P::Matrix{Float64}
    k::Int = 0
end

"""
    filter_step!(s::FilterState, m::Union{ClockModel2, ClockModel3, ClockModelDiurnal}, z_k::Float64, u_corr::Vector{Float64}=Float64[]) -> (state, innovation, innovation_variance)

One KF predict+update cycle. Pure estimation.
"""
function filter_step!(s::FilterState, m::Union{ClockModel2, ClockModel3, ClockModelDiurnal}, z_k::Float64, u_corr::Vector{Float64}=Float64[])
    s.k += 1
    Φ = build_phi(m)
    Q = build_Q(m)
    
    twopi = 2π
    if m isa ClockModelDiurnal
        H = zeros(Float64, 1, 5)
        H[1, 1] = 1.0
        H[1, 4] = sin(twopi * s.k / m.period)
        H[1, 5] = cos(twopi * s.k / m.period)
    else
        H = build_H(m)
    end
    
    x = s.x
    P = s.P
    R = m.noise.q_wpm
    
    if s.k > 1
        x = Φ * x
        if !isempty(u_corr)
            x[1] += u_corr[1]
            if length(x) >= 2 && length(u_corr) >= 2
                x[2] += u_corr[2]
            end
        end
        P = Symmetric(Φ * Matrix(P) * Φ' + Q)
    end
    
    ν = z_k - (H * x)[1]
    
    Pm = Matrix(P)
    S = (H * Pm * H')[1,1] + R
    K = (Pm * H') / S
    x = x + K[:,1] .* ν
    Pm = (I - K * H) * Pm
    Pm = (Pm + Pm') ./ 2.0   # Symmetrize to match MATLAB exactly
    
    for i in 1:length(x)
        Pm[i, i] = safe_sqrt(Pm[i, i])^2
    end
    
    s.x = x
    s.P = Matrix(Symmetric(Pm))
    
    return s, ν, S
end

"""
    PIDController
"""
Base.@kwdef mutable struct PIDController
    g_p::Float64 = 0.1
    g_i::Float64 = 0.01
    g_d::Float64 = 0.05
    sumx::Float64 = 0.0
    last_steer::Float64 = 0.0
end

function step!(c::PIDController, x::Vector{Float64})
    c.sumx += x[1]
    steer = -c.g_p * x[1] - c.g_i * c.sumx
    length(x) >= 2 && (steer -= c.g_d * x[2])
    c.last_steer = steer
    return steer
end

function _build_design_matrix(t::AbstractVector{Float64}, ns::Int, period::Float64)
    ns in (2, 3, 5) || error("nstates must be 2, 3, or 5; got $ns")
    n = length(t)
    A = ones(Float64, n, ns)
    ns >= 2 && (A[:, 2] = t)
    ns >= 3 && (A[:, 3] = (t .^ 2) ./ 2)
    if ns == 5
        A[:, 4] = sin.((2π / period) .* t)
        A[:, 5] = cos.((2π / period) .* t)
    end
    return A
end

function _initialize_state!(data::Vector{Float64}, ns::Int, period::Float64, tau::Float64)
    N = length(data)
    n_fit = min(100, N - 1)
    n_fit = max(n_fit, ns)
    n_fit >= N && error("Not enough data to initialize: need > $n_fit samples, got $N")
    t = Float64.((0:n_fit-1)) .* tau
    y = data[1:n_fit]
    A = _build_design_matrix(t, ns, period)
    coeffs = A \ y
    x0 = coeffs[1:ns]
    resid = y - A * coeffs
    P0_mat = var(resid) * Hermitian(inv(A' * A))
    return x0, P0_mat
end

# ── Main function ─────────────────────────────────────────────────────────────

"""
    kalman_filter(data, model; x0=Float64[], P0=1e6, g_p=0.1, g_i=0.01, g_d=0.05) -> KalmanResult

Run the Kalman filter on `data` using `model`.
"""
function kalman_filter(data::Vector{Float64}, model;
                       x0::Vector{Float64} = Float64[],
                       P0::Union{Float64, Matrix{Float64}} = 1e6,
                       g_p::Float64 = 0.1, g_i::Float64 = 0.01, g_d::Float64 = 0.05)
    isempty(data) && error("data must be non-empty")

    ns = nstates(model)
    τ = model.tau
    period = model isa ClockModelDiurnal ? model.period : 86400.0

    # Initialize state
    actual_x0 = if isempty(x0)
        init_x0, init_P0_mat = _initialize_state!(data, ns, period, τ)
        if isa(P0, Number) && P0 == 1e6  # If P0 not customized, use LS cov
            P0 = Matrix(init_P0_mat)
        end
        init_x0
    else
        copy(x0)
    end

    length(actual_x0) == ns || error("x0 length $(length(actual_x0)) ≠ nstates $ns")

    P_init = isa(P0, Matrix) ? copy(P0) : P0 * Matrix{Float64}(I, ns, ns)
    size(P_init) == (ns, ns) || error("P0 size must be ($ns,$ns)")

    N = length(data)
    R = model.noise.q_wpm
    twopi = 2π

    state     = FilterState(x=copy(actual_x0), P=Matrix(Symmetric(P_init)), k=0)
    pid       = PIDController(g_p=g_p, g_i=g_i, g_d=g_d)

    phase_est   = zeros(Float64, N)
    freq_est    = zeros(Float64, N)
    drift_est   = zeros(Float64, N)
    residuals_v = zeros(Float64, N)
    innov_v     = zeros(Float64, N)
    steers_v    = zeros(Float64, N)
    sumsteers_v = zeros(Float64, N)
    sum2steer_v = zeros(Float64, N)
    P_history   = Array{Float64, 3}(undef, ns, ns, N)

    phase = copy(data)

    for k in 1:N
        u_corr = Float64[]
        if k > 1
            u_corr = ns >= 2 ? Float64[pid.last_steer * τ, pid.last_steer] : Float64[pid.last_steer * τ]
            phase[k] = phase[k-1] + data[k] - data[k-1] + sumsteers_v[k-1]
        end

        _, innov, _ = filter_step!(state, model, phase[k], u_corr)
        x = state.x
        P = state.P

        resid = phase[k] - x[1]
        steer = step!(pid, x)

        if k == 1
            sumsteers_v[1] = pid.last_steer
            sum2steer_v[1] = sumsteers_v[1]
        else
            sumsteers_v[k] = sumsteers_v[k-1] + pid.last_steer
            sum2steer_v[k] = sum2steer_v[k-1] + sumsteers_v[k]
        end

        phase_est[k]   = x[1]
        freq_est[k]    = ns >= 2 ? x[2] : 0.0
        drift_est[k]   = ns >= 3 ? x[3] : 0.0
        residuals_v[k] = resid
        innov_v[k]     = innov
        steers_v[k]    = steer
        P_history[:, :, k] .= P
    end

    return KalmanResult(phase_est, freq_est, drift_est, residuals_v, innov_v,
                        steers_v, sumsteers_v, sum2steer_v, P_history, model)
end
