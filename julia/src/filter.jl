# filter.jl — Kalman filter for frequency stability / clock steering
# Struct-in, struct-out refactor of legacy filter.jl + utils.jl.
# Legacy refs: julia/legacy_kalmanlab/filter.jl, utils.jl, matlab/legacy/kflab/kalman_filter.m

using LinearAlgebra
using Statistics

# ── Structs ───────────────────────────────────────────────────────────────────

"""
    KalmanConfig

Configuration for the Kalman filter. All fields have defaults; override as needed.
`x0 = Float64[]` triggers automatic least-squares initialization from data.
`R` is the measurement noise variance; conventionally R = q_wpm.
"""
Base.@kwdef mutable struct KalmanConfig
    q_wpm::Float64     = 100.0      # White PM noise variance
    q_wfm::Float64     = 0.01       # White FM noise variance
    q_rwfm::Float64    = 1e-6       # Random walk FM noise variance
    q_irwfm::Float64   = 0.0        # Integrated RWFM noise variance
    q_diurnal::Float64 = 0.0        # Diurnal noise variance (requires nstates=5)
    R::Float64         = 100.0      # Measurement noise (= q_wpm typically)
    g_p::Float64       = 0.1        # PID proportional gain
    g_i::Float64       = 0.01       # PID integral gain
    g_d::Float64       = 0.05       # PID derivative gain
    nstates::Int       = 3          # State dimension: 2, 3, or 5
    tau::Float64       = 1.0        # Sampling interval [s]
    P0::Union{Float64, Matrix{Float64}} = 1e6  # Initial covariance (scalar or matrix)
    x0::Vector{Float64}             = Float64[]  # Initial state ([] → auto-init)
    period::Float64    = 86400.0    # Diurnal period [s]
end

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
    config::KalmanConfig
end

# ── Internal helpers ───────────────────────────────────────────────────────────

# Legacy kalman_filter.m lines 305-316
safe_sqrt(x::Float64) = abs(x) < 1e-10 ? 0.0 : x >= 0.0 ? sqrt(x) : -sqrt(-x)

# Legacy filter.jl: update_state_transition!
# Φ encodes constant-velocity/acceleration kinematics over interval τ
function build_phi!(Φ::Matrix{Float64}, nstates::Int, τ::Float64)
    fill!(Φ, 0.0)
    for i in 1:nstates; Φ[i,i] = 1.0; end
    nstates >= 2 && (Φ[1,2] = τ)
    nstates >= 3 && (Φ[1,3] = τ^2 / 2; Φ[2,3] = τ)
    # nstates == 5: diurnal states 4,5 are identity (no kinematic coupling)
end

# Legacy filter.jl: update_process_noise!
# SP1065: continuous-time noise model integrated over [0, τ]
function build_Q!(Q::Matrix{Float64}, nstates::Int,
                  q_wfm::Float64, q_rwfm::Float64, q_irwfm::Float64,
                  q_diurnal::Float64, τ::Float64)
    fill!(Q, 0.0)
    τ2 = τ^2; τ3 = τ^3; τ4 = τ^4; τ5 = τ^5
    Q[1,1] = q_wfm*τ + q_rwfm*τ3/3 + q_irwfm*τ5/20
    if nstates >= 2
        Q[1,2] = Q[2,1] = q_rwfm*τ2/2 + q_irwfm*τ4/8
        Q[2,2] = q_rwfm*τ + q_irwfm*τ3/3
    end
    if nstates >= 3
        Q[1,3] = Q[3,1] = q_irwfm*τ3/6
        Q[2,3] = Q[3,2] = q_irwfm*τ2/2
        Q[3,3] = q_irwfm*τ
    end
    nstates == 5 && (Q[4,4] = q_diurnal; Q[5,5] = q_diurnal)
end

# Legacy filter.jl lines 138-146; MATLAB PID convention: integral on phase
# Returns steer; mutates pid_state[1] (sumx) and pid_state[2] (last_steer)
function update_pid!(pid_state::Vector{Float64}, x::Vector{Float64},
                     nstates::Int, g_p::Float64, g_i::Float64, g_d::Float64)
    pid_state[1] += x[1]   # sumx accumulates phase error
    steer = -g_p * x[1] - g_i * pid_state[1]
    nstates >= 2 && (steer -= g_d * x[2])
    pid_state[2] = steer
    return steer
end

# Legacy utils.jl: build_design_matrix
function _build_design_matrix(t::AbstractVector{Float64}, nstates::Int, period::Float64)
    nstates in (2, 3, 5) || error("nstates must be 2, 3, or 5; got $nstates")
    n = length(t)
    A = ones(Float64, n, nstates)
    nstates >= 2 && (A[:, 2] = t)
    nstates >= 3 && (A[:, 3] = (t .^ 2) ./ 2)
    if nstates == 5
        A[:, 4] = sin.((2π / period) .* t)
        A[:, 5] = cos.((2π / period) .* t)
    end
    return A
end

# Legacy utils.jl: initialize! — least-squares fit to first n_fit samples
# Mutates cfg.x0 and cfg.P0 in-place.
function _initialize_state!(cfg::KalmanConfig, data::Vector{Float64})
    N = length(data)
    n_fit = min(100, N - 1)
    n_fit = max(n_fit, cfg.nstates)
    n_fit >= N && error("Not enough data to initialize: need > $n_fit samples, got $N")
    t = Float64.((0:n_fit-1)) .* cfg.tau
    y = data[1:n_fit]
    A = _build_design_matrix(t, cfg.nstates, cfg.period)
    coeffs = A \ y                      # least squares
    cfg.x0 = coeffs[1:cfg.nstates]
    resid = y - A * coeffs
    cfg.P0 = var(resid) * inv(A' * A)   # scaled inverse Hessian
end

# Legacy utils.jl: validate_filter
function _validate_config(cfg::KalmanConfig)
    ws = String[]
    cfg.nstates in (2, 3, 5)                          || push!(ws, "nstates must be 2, 3, or 5")
    length(cfg.x0) == cfg.nstates                     || push!(ws, "x0 length $(length(cfg.x0)) ≠ nstates $(cfg.nstates)")
    if isa(cfg.P0, Matrix)
        size(cfg.P0) == (cfg.nstates, cfg.nstates)    || push!(ws, "P0 matrix size ≠ (nstates, nstates)")
    else
        cfg.P0 > 0                                     || push!(ws, "P0 scalar must be > 0")
    end
    cfg.q_wpm     >= 0 || push!(ws, "q_wpm < 0")
    cfg.q_wfm     >= 0 || push!(ws, "q_wfm < 0")
    cfg.q_rwfm    >= 0 || push!(ws, "q_rwfm < 0")
    cfg.q_irwfm   >= 0 || push!(ws, "q_irwfm < 0")
    cfg.q_diurnal >= 0 || push!(ws, "q_diurnal < 0")
    cfg.q_diurnal > 0 && cfg.nstates != 5 && push!(ws, "q_diurnal > 0 requires nstates=5")
    cfg.tau > 0        || push!(ws, "tau must be > 0")
    return ws
end

# ── Main function ─────────────────────────────────────────────────────────────

"""
    kalman_filter(data, cfg) -> KalmanResult

Run the Kalman filter on `data` (phase measurements) with configuration `cfg`.
If `cfg.x0` is empty, the state is auto-initialized via least-squares on the
first min(100, N-1) samples. Returns a `KalmanResult`.
"""
function kalman_filter(data::Vector{Float64}, cfg::KalmanConfig)
    isempty(data) && error("data must be non-empty")

    # Auto-initialize state if x0 not provided — legacy utils.jl: initialize!
    isempty(cfg.x0) && _initialize_state!(cfg, data)

    ws = _validate_config(cfg)
    isempty(ws) || error("KalmanConfig invalid:\n  " * join(ws, "\n  "))

    N       = length(data)
    ns      = cfg.nstates
    τ       = cfg.tau
    R       = cfg.R
    g_p, g_i, g_d = cfg.g_p, cfg.g_i, cfg.g_d
    period  = cfg.period
    twopi   = 2π

    # Initialize matrices and state — legacy filter.jl: initialize_filter_state
    x = copy(cfg.x0)
    P_init = isa(cfg.P0, Matrix) ? copy(cfg.P0) : cfg.P0 * Matrix{Float64}(I, ns, ns)
    P = Symmetric(P_init)
    Φ = Matrix{Float64}(I, ns, ns)
    Q = zeros(Float64, ns, ns)
    H = zeros(Float64, 1, ns); H[1, 1] = 1.0  # measures phase only
    pid_state = zeros(Float64, 2)  # [sumx, last_steer]

    # Pre-allocate outputs
    phase_est   = zeros(Float64, N)
    freq_est    = zeros(Float64, N)
    drift_est   = zeros(Float64, N)
    residuals_v = zeros(Float64, N)
    innov_v     = zeros(Float64, N)
    steers_v    = zeros(Float64, N)
    sumsteers_v = zeros(Float64, N)
    sum2steer_v = zeros(Float64, N)
    P_history   = Array{Float64, 3}(undef, ns, ns, N)

    phase = copy(data)  # working copy — updated each step

    for k in 1:N
        build_phi!(Φ, ns, τ)
        build_Q!(Q, ns, cfg.q_wfm, cfg.q_rwfm, cfg.q_irwfm, cfg.q_diurnal, τ)

        # Diurnal measurement update — legacy filter.jl lines 75-79
        if ns == 5
            H[1, 4] = sin(twopi * k / period)
            H[1, 5] = cos(twopi * k / period)
        end

        # Predict — skipped at k=1, legacy filter.jl lines 82-97
        if k > 1
            x = Φ * x
            x[1] += pid_state[2] * τ   # phase correction from last steer
            ns >= 2 && (x[2] += pid_state[2])  # freq correction
            Pm = Matrix(P)
            Pm = Φ * Pm * Φ' + Q
            P  = Symmetric(Pm)

            # Phase accumulation — legacy filter.jl line 104 / kalman_filter.m line 152
            phase[k] = phase[k-1] + data[k] - data[k-1] + sumsteers_v[k-1]
        end

        # Innovation — legacy filter.jl line 108
        innov = phase[k] - (H * x)[1]

        # Kalman update — P = (I - K*H)*P; legacy filter.jl lines 111-123
        Pm = Matrix(P)
        S  = (H * Pm * H')[1, 1] + R
        K  = (Pm * H') / S          # ns×1 matrix
        x  = x + K[:, 1] .* innov
        Pm = (I - K * H) * Pm
        
        # Guard diagonal against numerical drift — GEMINI.md §2.2
        for i in 1:ns
            Pm[i, i] = safe_sqrt(Pm[i, i])^2
        end
        P  = Symmetric(Pm)

        # Residual — legacy filter.jl line 135
        resid = phase[k] - x[1]

        # PID — legacy filter.jl lines 138-146; MATLAB: sumx += x(1)
        steer = update_pid!(pid_state, x, ns, g_p, g_i, g_d)

        # Cumulative steering — legacy filter.jl lines 149-155
        if k == 1
            sumsteers_v[1]  = pid_state[2]
            sum2steer_v[1]  = sumsteers_v[1]
        else
            sumsteers_v[k]  = sumsteers_v[k-1] + pid_state[2]
            sum2steer_v[k]  = sum2steer_v[k-1] + sumsteers_v[k]
        end

        # Store — legacy filter.jl lines 158-169
        phase_est[k]   = x[1]
        freq_est[k]    = ns >= 2 ? x[2] : 0.0
        drift_est[k]   = ns >= 3 ? x[3] : 0.0
        residuals_v[k] = resid
        innov_v[k]     = innov
        steers_v[k]    = steer
        P_history[:, :, k] .= P
    end

    return KalmanResult(phase_est, freq_est, drift_est, residuals_v, innov_v,
                        steers_v, sumsteers_v, sum2steer_v, P_history, cfg)
end
