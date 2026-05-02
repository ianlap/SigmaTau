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

function innovation_nll(data::Vector{Float64}, m::AbstractStateModel,
                        meas::AbstractMeasurementModel)::Float64
    N = length(data)
    ns = nstates(m)

    Φ = build_phi(m)
    Q = build_Q(m)
    R = measurement_R(meas, m)

    H_base = build_H(meas, m, 0)
    # _dare only converges for time-invariant H; diurnal uses Q as a placeholder seed.
    P = m isa ClockModel2 ? _dare_scalar_H(Φ, Q, R) : Matrix{Float64}(Q)
    x = zeros(Float64, ns)
    nll = 0.0

    for k in 1:N
        H = m isa ClockModelDiurnal ? build_H(meas, m, k) : H_base

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

# 2-arg back-compat: defaults to PhaseOnlyMeasurement.
innovation_nll(data::Vector{Float64}, m::AbstractStateModel)::Float64 =
    innovation_nll(data, m, PhaseOnlyMeasurement())

# Perf shim: the ClockModel3 + PhaseOnly fast path stays the StaticArrays
# kernel, even when callers pass meas explicitly (e.g. optimize_nll inner loop).
innovation_nll(data::Vector{Float64}, m::ClockModel3, ::PhaseOnlyMeasurement)::Float64 =
    innovation_nll(data, m)

"""
    OptimizeNLLResult

Return value of [`optimize_nll`](@ref). Wraps the fitted noise parameters
together with optimizer diagnostics so callers do not have to re-run the
likelihood or guess at convergence status.

# Fields
- `noise::ClockNoiseParams`: fitted q's at the NLL optimum.
- `nll::Float64`: value of the innovation NLL at the optimum.
- `n_evals::Int`: number of `innovation_nll` evaluations consumed by Nelder-Mead.
- `converged::Bool`: `true` iff the simplex `std(fvals) < tol` criterion fired
  before `max_iter` exhausted (i.e., the optimizer stopped on convergence,
  not on the iteration cap).
"""
struct OptimizeNLLResult
    noise::ClockNoiseParams
    nll::Float64
    n_evals::Int
    converged::Bool
end

"""
    optimize_nll(data, tau0; h_init, noise_init, optimize_qwpm=false, ...) -> OptimizeNLLResult

Fit Zucca-Tavella clock-SDE diffusion parameters by Nelder-Mead on the
Gaussian innovation NLL. Returns an [`OptimizeNLLResult`](@ref) carrying the
fitted `ClockNoiseParams` plus the optimizer's NLL, evaluation count, and
convergence flag.

By default `optimize_qwpm=false`: `q_wpm` (= measurement noise `R`) is held at
its seed value, matching MATLAB `sigmatau.kf.optimize` and the textbook tuning
problem where `R` is known from the short-τ WPM floor or the measurement-chain
calibration. Pass `optimize_qwpm=true` to sweep `R` jointly with the diffusion
parameters.
"""
function optimize_nll(data::AbstractVector{<:Real}, tau0::Real;
                      h_init::Union{Nothing,AbstractDict{<:Real,<:Real}}=nothing,
                      noise_init::Union{ClockNoiseParams, Nothing}=nothing,
                      optimize_qwpm::Bool = false,
                      optimize_irwfm::Bool = false,
                      verbose::Bool = true,
                      max_iter::Int = 500,
                      tol::Float64 = 1e-6,
                      state_ctor::Union{Function,Nothing} = nothing,
                      meas::AbstractMeasurementModel = PhaseOnlyMeasurement())
    
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

        n = ClockNoiseParams(q_wpm, q_wfm, q_rwfm, q_irwfm)
        m = state_ctor === nothing ? ClockModel3(noise=n, tau=tau0) : state_ctor(n)
        return innovation_nll(Vector{Float64}(data), m, meas)
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

    noise_opt = ClockNoiseParams(q_wpm_opt, q_wfm_opt, q_rwfm_opt, q_irwfm_opt)
    return OptimizeNLLResult(noise_opt, nll_opt, n_evals, converged)
end
