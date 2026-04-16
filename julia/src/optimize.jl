# optimize.jl — KF parameter optimization via NLL + Nelder-Mead

using LinearAlgebra: I, norm
using Statistics
using StaticArrays: SMatrix, SVector, @SMatrix, @SVector

const _INVALID_NLL       = 1e15
const _NM_PERTURB_EPS    = 1e-10
const _NM_PERTURB_FRAC   = 0.05

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
    end

    theta_opt, nll_opt, n_evals, converged = _nelder_mead(obj, theta0; max_iter=max_iter, tol=tol)

    idx_opt = 1
    q_wpm_opt = optimize_qwpm ? 10.0^theta_opt[idx_opt] : noise_init.q_wpm; optimize_qwpm && (idx_opt += 1)
    q_wfm_opt = 10.0^theta_opt[idx_opt]; idx_opt += 1
    q_rwfm_opt = 10.0^theta_opt[idx_opt]; idx_opt += 1
    q_irwfm_opt = optimize_irwfm ? 10.0^theta_opt[idx_opt] : noise_init.q_irwfm

    if verbose
        println("  NLL = $(round(nll_opt, sigdigits=6))  ($n_evals evals)")
        optimize_qwpm && println("  q_wpm   = $(round(q_wpm_opt,   sigdigits=3))")
        println("  q_wfm   = $(round(q_wfm_opt,   sigdigits=3))")
        println("  q_rwfm  = $(round(q_rwfm_opt,  sigdigits=3))")
        optimize_irwfm && println("  q_irwfm = $(round(q_irwfm_opt, sigdigits=3))")
    end

    return ClockNoiseParams(q_wpm_opt, q_wfm_opt, q_rwfm_opt, q_irwfm_opt)
end
