# optimize.jl — KF parameter optimization via NLL + Nelder-Mead

using LinearAlgebra: I, norm
using Statistics
using StaticArrays: SMatrix, SVector, @SMatrix, @SVector

const _INVALID_NLL       = 1e15
const _NM_PERTURB_EPS    = 1e-10
const _NM_PERTURB_FRAC   = 0.05
Configuration for `optimize_kf`. All fields have defaults; override as needed.
`q_irwfm = 0` (default) excludes the IRWFM parameter from optimization.
"""
Base.@kwdef struct OptimizeConfig
    q_wpm::Float64   = 100.0   # Measurement noise variance R (fixed)
    q_wfm::Float64   = 0.01    # Initial WFM guess
    q_rwfm::Float64  = 1e-6    # Initial RWFM guess
    q_irwfm::Float64    = 0.0     # Initial IRWFM guess (0 = not optimized)
    nstates::Int        = 3       # KF state dimension: 2 or 3
    tau::Float64        = 1.0     # Sampling interval [s]
    verbose::Bool       = true    # Print progress
    max_iter::Int       = 500     # Max Nelder-Mead iterations
    tol::Float64        = 1e-6    # Convergence tolerance (std of simplex f-values)
    optimize_qwpm::Bool = false   # If true, walk log10(q_wpm) too
end

const _NM_ALPHA = 1.0
const _NM_GAMMA = 2.0
const _NM_RHO   = 0.5
const _NM_SIGMA = 0.5

function _nelder_mead(f, x0::Vector{Float64}; max_iter::Int = 500, tol::Float64 = 1e-6)
    n = length(x0)
    verts = [copy(x0) for _ in 1:n+1]
    for i in 1:n
        δ = abs(x0[i]) > _NM_PERTURB_EPS ? _NM_PERTURB_FRAC * x0[i] : _NM_PERTURB_FRAC
        verts[i+1][i] += δ
    end

    fvals   = [f(v) for v in verts]
    n_evals = n + 1
    converged = false

    for _ in 1:max_iter
        ord    = sortperm(fvals)
        verts  = verts[ord]
        fvals  = fvals[ord]

        if std(fvals) < tol
            converged = true
            break
        end

        xbar = mean(verts[1:n])
        xr = xbar .+ _NM_ALPHA .* (xbar .- verts[end])
        fr = f(xr); n_evals += 1

        if fr < fvals[1]
            xe = xbar .+ _NM_GAMMA .* (xr .- xbar)
            fe = f(xe); n_evals += 1
            if fe < fr
                verts[end] = xe; fvals[end] = fe
            else
                verts[end] = xr; fvals[end] = fr
            end
        elseif fr < fvals[end-1]
            verts[end] = xr; fvals[end] = fr
        else
            if fr < fvals[end]
                xc = xbar .+ _NM_RHO .* (xr .- xbar)
                fc = f(xc); n_evals += 1
                if fc <= fr
                    verts[end] = xc; fvals[end] = fc
                else
                    _shrink!(verts, fvals, f, n); n_evals += n
                end
            else
                xc = xbar .+ _NM_RHO .* (verts[end] .- xbar)
                fc = f(xc); n_evals += 1
                if fc < fvals[end]
                    verts[end] = xc; fvals[end] = fc
                else
                    _shrink!(verts, fvals, f, n); n_evals += n
                end
            end
        end
    end
    ord = sortperm(fvals)
    return verts[ord[1]], fvals[ord[1]], n_evals, converged
end

function _shrink!(verts, fvals, f, n::Int)
    best = verts[1]
    for i in 2:n+1
        verts[i] = best .+ _NM_SIGMA .* (verts[i] .- best)
        fvals[i] = f(verts[i])
    end
end

"""
    innovation_nll(data, model) -> Float64

Evaluate the negative log-likelihood of `data` under `model`.
"""
function innovation_nll(data::Vector{Float64}, m::ClockModel3)::Float64
    N = length(data)
    τ = m.tau

    noise = m.noise
    R = noise.q_wpm
    q_wfm = noise.q_wfm
    q_rwfm = noise.q_rwfm
    q_irwfm = noise.q_irwfm

    Φ = @SMatrix [1.0  τ     τ^2/2;
                  0.0  1.0   τ;
                  0.0  0.0   1.0]

    τ2 = τ^2; τ3 = τ^3; τ4 = τ^4; τ5 = τ^5
    Q11 = q_wfm*τ + q_rwfm*τ3/3.0 + q_irwfm*τ5/20.0
    Q12 = q_rwfm*τ2/2.0 + q_irwfm*τ4/8.0
    Q13 = q_irwfm*τ3/6.0
    Q22 = q_rwfm*τ + q_irwfm*τ3/3.0
    Q23 = q_irwfm*τ2/2.0
    Q33 = q_irwfm*τ
    Q   = @SMatrix [Q11 Q12 Q13; Q12 Q22 Q23; Q13 Q23 Q33]

    P = _dare_scalar_H_static(Φ, Q, R)
    x = @SVector zeros(Float64, 3)
    nll = 0.0

    @inbounds for k in 1:N
        if k > 1
            x = Φ * x
            P = Φ * P * Φ' + Q
        end
        # Scalar-H shortcut: H = [1 0 0]
        ν = data[k] - x[1]
        S = P[1,1] + R
        S <= 0.0 && return _INVALID_NLL
        nll += 0.5 * (log(S) + ν*ν/S)
        
        K = SVector{3,Float64}(P[1,1]/S, P[2,1]/S, P[3,1]/S)
        x = x + K .* ν
        pr1 = SVector{3,Float64}(P[1,1], P[1,2], P[1,3])
        P = P - K * pr1'
    end
    return nll
end

function innovation_nll(data::Vector{Float64}, m::Union{ClockModel2, ClockModelDiurnal})::Float64
    N = length(data)
    ns = nstates(m)
    τ = m.tau
    
    Φ = build_phi(m)
    Q = build_Q(m)
    R = m.noise.q_wpm

    H_base = build_H(m) # For diurnal, we handle k later. Wait, for _dare, diurnal is problematic if H varies.
    # Actually, dare doesn't support diurnal. We use Q as init placeholder.
    P = m isa ClockModel2 ? _dare_scalar_H(Φ, Q, R) : Matrix{Float64}(Q)
    x = zeros(Float64, ns)
function _kf_nll(theta::Vector{Float64}, data::Vector{Float64},
                 cfg::OptimizeConfig)::Float64
    N   = length(data)
    ns  = cfg.nstates
    τ   = cfg.tau

    # Unpack theta — layout depends on cfg.optimize_qwpm and cfg.q_irwfm
    idx = 1
    R = cfg.q_wpm
    if cfg.optimize_qwpm
        R = 10.0^theta[idx]; idx += 1
    end
    q_wfm   = 10.0^theta[idx]; idx += 1
    q_rwfm  = 10.0^theta[idx]; idx += 1
    q_irwfm = length(theta) >= idx ? 10.0^theta[idx] : 0.0

    # State-transition Φ — matches filter.jl build_phi!
    Φ = Matrix{Float64}(I, ns, ns)
    ns >= 2 && (Φ[1, 2] = τ)
    ns >= 3 && (Φ[1, 3] = τ^2 / 2; Φ[2, 3] = τ)

    H = zeros(Float64, 1, ns)
    H[1, 1] = 1.0  # observe phase only

    # Process noise Q — SP1065 continuous-time model, matches filter.jl build_Q!
    Q = _build_Q(ns, q_wfm, q_rwfm, q_irwfm, τ)

    # LS initialization on first min(100, N-1) samples
    n_fit = max(ns, min(_MAX_LS_SAMPLES, N - 1))
    t_fit = Float64.(0:n_fit-1) .* τ
    A_fit = _build_A(t_fit, ns)
    x     = A_fit \ data[1:n_fit]

    P   = _P0_SCALE .* Matrix{Float64}(I, ns, ns)
    nll = 0.0
    twopi = 2π
    period = m isa ClockModelDiurnal ? m.period : 86400.0

    for k in 1:N
        if m isa ClockModelDiurnal
            H = zeros(Float64, 1, 5)
            H[1,1] = 1.0; H[1,4] = sin(twopi*k/period); H[1,5] = cos(twopi*k/period)
        else
            H = H_base
        end

        if k > 1
            x = Φ * x
            P = Φ * P * Φ' + Q
        end

        ν = data[k] - (H * x)[1]
        S = (H * P * H')[1,1] + R

        S <= 0.0 && return _INVALID_NLL

        nll += 0.5 * (log(S) + ν^2 / S)

        K = (P * H') ./ S
        x = x + K[:, 1] .* ν
        P = (I - K * H) * P
    end
    return nll
end

function _build_Q(ns::Int, q_wfm::Float64, q_rwfm::Float64,
                  q_irwfm::Float64, τ::Float64)
    # SP1065: continuous-time noise model integrated over τ.
    # Matches filter.jl build_Q! exactly.
    Q  = zeros(Float64, ns, ns)
    τ2 = τ^2; τ3 = τ^3; τ4 = τ^4; τ5 = τ^5
    Q[1, 1] = q_wfm*τ + q_rwfm*τ3/3 + q_irwfm*τ5/20
    if ns >= 2
        Q[1, 2] = Q[2, 1] = q_rwfm*τ2/2 + q_irwfm*τ4/8
        Q[2, 2] = q_rwfm*τ + q_irwfm*τ3/3
    end
    if ns >= 3
        Q[1, 3] = Q[3, 1] = q_irwfm*τ3/6
        Q[2, 3] = Q[3, 2] = q_irwfm*τ2/2
        Q[3, 3] = q_irwfm*τ
    end
    return Q
end

function _build_A(t::Vector{Float64}, ns::Int)
    # Design matrix [1, t, t²/2] for LS state initialization.
    A = ones(Float64, length(t), ns)
    ns >= 2 && (A[:, 2] = t)
    ns >= 3 && (A[:, 3] = t .^ 2 ./ 2)
    return A
end

# ── Static fast-path NLL kernel (nstates == 3) ────────────────────────────────

using StaticArrays: SMatrix, SVector, @SMatrix, @SVector

"""
    _kf_nll_static(theta, data, cfg) → Float64

Faster NLL kernel for `nstates == 3` using StaticArrays and the scalar-H
shortcut.  Produces identical results to `_kf_nll` to numerical precision.
"""
function _kf_nll_static(theta::Vector{Float64}, data::Vector{Float64},
                        cfg::OptimizeConfig)::Float64
    cfg.nstates == 3 || return _kf_nll(theta, data, cfg)

    N = length(data)
    τ = cfg.tau

    idx = 1
    R = cfg.q_wpm
    if cfg.optimize_qwpm
        R = 10.0^theta[idx]; idx += 1
    end
    q_wfm   = 10.0^theta[idx]; idx += 1
    q_rwfm  = 10.0^theta[idx]; idx += 1
    q_irwfm = length(theta) >= idx ? 10.0^theta[idx] : 0.0

    # Φ (3×3) and Q (3×3) as static matrices
    Φ = @SMatrix [1.0  τ     τ^2/2;
                  0.0  1.0   τ;
                  0.0  0.0   1.0]

    τ2 = τ^2; τ3 = τ^3; τ4 = τ^4; τ5 = τ^5
    Q11 = q_wfm*τ + q_rwfm*τ3/3 + q_irwfm*τ5/20
    Q12 = q_rwfm*τ2/2 + q_irwfm*τ4/8
    Q13 = q_irwfm*τ3/6
    Q22 = q_rwfm*τ + q_irwfm*τ3/3
    Q23 = q_irwfm*τ2/2
    Q33 = q_irwfm*τ
    Q   = @SMatrix [Q11 Q12 Q13; Q12 Q22 Q23; Q13 Q23 Q33]

    # LS initialization — hoisted into static form
    n_fit = max(3, min(_MAX_LS_SAMPLES, N - 1))
    t_fit = Float64.(0:n_fit-1) .* τ
    A     = hcat(ones(n_fit), t_fit, t_fit.^2 ./ 2)
    xls   = A \ data[1:n_fit]
    x     = SVector{3,Float64}(xls[1], xls[2], xls[3])

    P   = SMatrix{3,3,Float64}(_P0_SCALE*I)
    nll = 0.0

    @inbounds for k in 1:N
        if k > 1
            x = Φ * x
            P = Φ * P * Φ' + Q
        end
        # Scalar-H shortcut: H = [1 0 0]
        ν = data[k] - x[1]
        S = P[1,1] + R
        S <= 0.0 && return _INVALID_NLL
        nll += 0.5 * (log(S) + ν*ν/S)
        # Kalman gain (3×1)
        K = SVector{3,Float64}(P[1,1]/S, P[2,1]/S, P[3,1]/S)
        x = x + K .* ν
        # P = (I - K H) P = P - K * P[1,:]'
        pr1 = SVector{3,Float64}(P[1,1], P[1,2], P[1,3])
        P = P - K * pr1'
    end
    return nll
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    optimize_nll(data, tau0; h_init, noise_init, ...) -> ClockNoiseParams

Optimize parameters over NelderMead. Returns optimal `ClockNoiseParams`.
"""
function optimize_nll(data::AbstractVector{<:Real}, tau0::Real;
                      h_init::Union{Nothing,AbstractDict{<:Real,<:Real}}=nothing,
                      noise_init::Union{ClockNoiseParams, Nothing}=nothing,
                      optimize_qwpm::Bool = true,
                      optimize_irwfm::Bool = false,
                      verbose::Bool = true,
                      max_iter::Int = 500,
                      tol::Float64 = 1e-6)
    
    if noise_init === nothing
        if h_init !== nothing
            noise_init = h_to_q(h_init, tau0)
        else
            # Default fallbacks
            noise_init = ClockNoiseParams(q_wpm=1e-26, q_wfm=1e-25, q_rwfm=1e-26)
        end
    end

    if verbose
        println("\n=== KF NLL OPTIMIZATION (Nelder-Mead) ===")
        println("  q_wpm   = $(round(noise_init.q_wpm,   sigdigits=3))  ($(optimize_qwpm ? "initial guess" : "fixed"), R)")
        println("  q_wfm0  = $(round(noise_init.q_wfm,   sigdigits=3))")
        println("  q_rwfm0 = $(round(noise_init.q_rwfm,  sigdigits=3))")
        optimize_irwfm && println("  q_irwfm0= $(round(noise_init.q_irwfm, sigdigits=3))")
    end

    theta0 = Float64[]
    optimize_qwpm && push!(theta0, log10(max(noise_init.q_wpm, 1e-40)))
    push!(theta0, log10(max(noise_init.q_wfm, 1e-40)))
    push!(theta0, log10(max(noise_init.q_rwfm, 1e-40)))
    optimize_irwfm && push!(theta0, log10(max(noise_init.q_irwfm, 1e-40)))

    function obj(th)
        idx = 1
        q_wpm = optimize_qwpm ? 10.0^th[idx] : noise_init.q_wpm; optimize_qwpm && (idx += 1)
        q_wfm = 10.0^th[idx]; idx += 1
        q_rwfm = 10.0^th[idx]; idx += 1
        q_irwfm = optimize_irwfm ? 10.0^th[idx] : noise_init.q_irwfm

        m = ClockModel3(noise=ClockNoiseParams(q_wpm, q_wfm, q_rwfm, q_irwfm), tau=tau0)
        return innovation_nll(Vector{Float64}(data), m)
        println("  q_wpm   = $(round(cfg.q_wpm,   sigdigits=3))  ($(cfg.optimize_qwpm ? "initial guess" : "fixed"), R)")
        println("  q_wfm0  = $(round(cfg.q_wfm,   sigdigits=3))")
        println("  q_rwfm0 = $(round(cfg.q_rwfm,  sigdigits=3))")
        cfg.q_irwfm > 0 && println("  q_irwfm0= $(round(cfg.q_irwfm, sigdigits=3))")
    end

    # Initial guess in log10 space
    theta0 = Float64[]
    cfg.optimize_qwpm && push!(theta0, log10(cfg.q_wpm))
    push!(theta0, log10(cfg.q_wfm))
    push!(theta0, log10(cfg.q_rwfm))
    cfg.q_irwfm > 0 && push!(theta0, log10(cfg.q_irwfm))

    obj = if cfg.nstates == 3
        th -> _kf_nll_static(th, data, cfg)
    else
        th -> _kf_nll(th, data, cfg)
    end

    theta_opt, nll_opt, n_evals, converged = _nelder_mead(obj, theta0; max_iter=max_iter, tol=tol)

    idx_opt = 1
    q_wpm_opt = optimize_qwpm ? 10.0^theta_opt[idx_opt] : noise_init.q_wpm; optimize_qwpm && (idx_opt += 1)
    q_wfm_opt = 10.0^theta_opt[idx_opt]; idx_opt += 1
    q_rwfm_opt = 10.0^theta_opt[idx_opt]; idx_opt += 1
    q_irwfm_opt = optimize_irwfm ? 10.0^theta_opt[idx_opt] : noise_init.q_irwfm
    idx = 1
    q_wpm_opt = cfg.q_wpm
    if cfg.optimize_qwpm
        q_wpm_opt = 10.0^theta_opt[idx]; idx += 1
    end
    q_wfm_opt   = 10.0^theta_opt[idx]; idx += 1
    q_rwfm_opt  = 10.0^theta_opt[idx]; idx += 1
    q_irwfm_opt = length(theta_opt) >= idx ? 10.0^theta_opt[idx] : 0.0

    if verbose
        println("  NLL = $(round(nll_opt, sigdigits=6))  ($n_evals evals)")
        optimize_qwpm && println("  q_wpm   = $(round(q_wpm_opt,   sigdigits=3))")
        cfg.optimize_qwpm && println("  q_wpm   = $(round(q_wpm_opt,   sigdigits=3))")
        println("  q_wfm   = $(round(q_wfm_opt,   sigdigits=3))")
        println("  q_rwfm  = $(round(q_rwfm_opt,  sigdigits=3))")
        optimize_irwfm && println("  q_irwfm = $(round(q_irwfm_opt, sigdigits=3))")
    end

    return ClockNoiseParams(q_wpm_opt, q_wfm_opt, q_rwfm_opt, q_irwfm_opt)
    return OptimizeResult(q_wpm_opt, q_wfm_opt, q_rwfm_opt, q_irwfm_opt,
                          nll_opt, n_evals, converged)
end

"""
    optimize_kf_nll(phase, τ₀; h_init=nothing, verbose=true, max_iter=500) → OptimizeResult

High-level wrapper used by the ML dataset driver.  Optimizes
`(q_wpm, q_wfm, q_rwfm)` in log10 space via Nelder-Mead on `_kf_nll_static`.
If `h_init::Dict` is provided, uses the SP1065 analytical mapping h_α → q
as the simplex centroid — drastically reduces iteration count.
"""
function optimize_kf_nll(phase::AbstractVector{<:Real}, τ₀::Real;
                         h_init::Union{Nothing,AbstractDict{<:Real,<:Real}}=nothing,
                         verbose::Bool = true,
                         max_iter::Int = 500,
                         tol::Float64  = 1e-6)
    # Defaults cover the typical Rb range; overridden by h_init when supplied.
    q_wpm0, q_wfm0, q_rwfm0 = 1e-26, 1e-25, 1e-26
    if h_init !== nothing
        f_h = 1.0 / (2τ₀)
        haskey(h_init,  2.0) && (q_wpm0  = h_init[ 2.0] * f_h / (2π^2))
        haskey(h_init,  0.0) && (q_wfm0  = h_init[ 0.0] / 2)
        haskey(h_init, -2.0) && (q_rwfm0 = (2π^2 / 3) * h_init[-2.0])
    end
    cfg = OptimizeConfig(
        q_wpm   = q_wpm0,
        q_wfm   = q_wfm0,
        q_rwfm  = q_rwfm0,
        nstates = 3,
        tau     = τ₀,
        verbose = verbose,
        max_iter = max_iter,
        tol      = tol,
        optimize_qwpm = (h_init === nothing),
    )
    return optimize_kf(Vector{Float64}(phase), cfg)
end
