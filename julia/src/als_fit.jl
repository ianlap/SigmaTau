# als_fit.jl — Autocovariance Least Squares (ALS) Estimator
# Implements the tuning algorithm described in Åkesson et al. (2008) and Odelson et al. (2006).

using LinearAlgebra



"""
    als_fit(data, tau0; kwargs...) -> ClockNoiseParams

Optimize parameters iteratively via the Autocovariance Least Squares (ALS) method
(Åkesson 2008 / Odelson 2006).

By default `optimize_qwpm=false`: `q_wpm` (= measurement noise `R`) is held at
its seed value, matching MATLAB `sigmatau.kf.optimize` and `optimize_nll` —
the textbook tuning problem where `R` is known from the short-τ WPM floor or
the measurement-chain calibration. Pass `optimize_qwpm=true` to sweep `R`
jointly with the diffusion parameters.

# Keyword Arguments
- `h_init`, `noise_init` — initial parameter seeds.
- `optimize_qwpm` (default: `false`) — include measurement noise R in optimization.
- `optimize_irwfm` (default: `false`) — include Random Run FM in optimization.
- `lags` (default: `30`) — number of autocovariance lags to consider.
- `burn_in` (default: `50`) — number of steady-state relaxation steps to ignore.
- `max_iter` (default: `5`) — maximum iterations of the ALS parameter-update loop.
- `verbose` (default: `true`) — print iteration diagnostics.
"""
function als_fit(data::AbstractVector{<:Real}, tau0::Real;
                 h_init::Union{Nothing,AbstractDict{<:Real,<:Real}}=nothing,
                 noise_init::Union{ClockNoiseParams, Nothing}=nothing,
                 optimize_qwpm::Bool = false,
                 optimize_irwfm::Bool = false,
                 lags::Int = 30, max_iter::Int = 5, burn_in::Int = 50, verbose::Bool = true)
    
    if noise_init === nothing
        if h_init !== nothing
            noise_init = h_to_q(h_init, tau0)
        else
            noise_init = ClockNoiseParams(q_wpm=1e-26, q_wfm=1e-25, q_rwfm=1e-26)
        end
    end

    if verbose
        println("\n=== ALS TUNING ESTIMATOR ===")
        println("  lags = $(lags), burn_in = $(burn_in)")
        println("  q_wpm   = $(round(noise_init.q_wpm,   sigdigits=3))  ($(optimize_qwpm ? "initial guess" : "fixed"), R)")
        println("  q_wfm0  = $(round(noise_init.q_wfm,   sigdigits=3))")
        println("  q_rwfm0 = $(round(noise_init.q_rwfm,  sigdigits=3))")
        optimize_irwfm && println("  q_irwfm0= $(round(noise_init.q_irwfm, sigdigits=3))")
    end

    current_noise = noise_init
    for iter in 1:max_iter
        if verbose
            println("\n--- ALS Iteration $iter ---")
        end
        n_out = _als_iteration(data, tau0, current_noise, lags, burn_in, optimize_qwpm, optimize_irwfm)
        
        if verbose
            optimize_qwpm && println("  q_wpm   = $(round(n_out.q_wpm, sigdigits=3))")
            println("  q_wfm   = $(round(n_out.q_wfm, sigdigits=3))")
            println("  q_rwfm  = $(round(n_out.q_rwfm, sigdigits=3))")
            optimize_irwfm && println("  q_irwfm = $(round(n_out.q_irwfm, sigdigits=3))")
        end

        # Convergence test
        diff = abs(n_out.q_wfm - current_noise.q_wfm) / max(current_noise.q_wfm, 1e-40)
        current_noise = n_out
        
        if diff < 1e-3
            verbose && println("ALS Converged.")
            break
        end
    end
    
    return current_noise
end


function _als_iteration(data::AbstractVector{<:Real}, tau0::Real, noise_current::ClockNoiseParams,
                        lags::Int, burn_in::Int, opt_wpm::Bool, opt_irwfm::Bool)
    N = length(data)
    m = ClockModel3(noise=noise_current, tau=tau0)
    
    # Run KF to extract innovation sequence
    res = kalman_filter(data, m)
    inns = @view res.innovations[min(burn_in+1, N):end]
    Nk = length(inns)
    
    # Empirical autocovariance sequence
    C_hat = zeros(lags + 1)
    for j in 0:lags
        s = 0.0
        for k in 1:(Nk - j)
            s += inns[k] * inns[k+j]
        end
        C_hat[j+1] = s / (Nk - j)
    end
    
    # Base KF properties
    P_inf = steady_state_covariance(m)
    Phi = build_phi(m)
    H = build_H(m)
    R_current = noise_current.q_wpm
    
    S = (H * P_inf * H')[1,1] + R_current
    K = (P_inf * H') ./ S
    Abar = Phi - K * H
    
    n_s = nstates(m)
    pinv_A = pinv(I(n_s^2) - kron(Abar, Abar))
    
    opt_keys = Symbol[]
    opt_wpm && push!(opt_keys, :q_wpm)
    push!(opt_keys, :q_wfm)
    push!(opt_keys, :q_rwfm)
    opt_irwfm && push!(opt_keys, :q_irwfm)
    
    all_keys = [:q_wpm, :q_wfm, :q_rwfm, :q_irwfm]
    C_basis = Dict{Symbol, Vector{Float64}}()
    
    for k in all_keys
        p_dict = Dict(:q_wpm => 0.0, :q_wfm => 0.0, :q_rwfm => 0.0, :q_irwfm => 0.0)
        p_dict[k] = 1.0
        
        Q_b = build_Q(ClockModel3(noise=ClockNoiseParams(p_dict[:q_wpm], p_dict[:q_wfm], p_dict[:q_rwfm], p_dict[:q_irwfm]), tau=tau0))
        R_b = p_dict[:q_wpm]
        
        vec_P = pinv_A * vec(Q_b .+ K * R_b * K')
        P_b = reshape(vec_P, n_s, n_s)
        
        c = zeros(lags + 1)
        c[1] = (H * P_b * H')[1,1] + R_b
        
        if lags > 0
            for j in 1:lags
                c[j+1] = (H * (Abar^j) * P_b * H')[1,1] - (H * (Abar^(j-1)) * K * R_b)[1]
            end
        end
        C_basis[k] = c
    end
    
    b = copy(C_hat)
    if !opt_wpm
        b .-= noise_current.q_wpm .* C_basis[:q_wpm]
    end
    if !opt_irwfm
        b .-= noise_current.q_irwfm .* C_basis[:q_irwfm]
    end
    
    A = zeros(lags + 1, length(opt_keys))
    for (i, k) in enumerate(opt_keys)
        A[:, i] = C_basis[k]
    end
    
    # Solve non-negative LS using Nelder-Mead on log10 parameters
    th0 = zeros(length(opt_keys))
    for (i, k) in enumerate(opt_keys)
        v = k == :q_wpm ? noise_current.q_wpm :
            k == :q_wfm ? noise_current.q_wfm :
            k == :q_rwfm ? noise_current.q_rwfm : noise_current.q_irwfm
        th0[i] = log10(max(v, 1e-40))
    end
    
    function obj(log_th)
        th = 10.0 .^ log_th
        return log10(sum(abs2, A * th .- b) + 1e-60)
    end
    
    opt_log_th, _, _, _ = _nelder_mead(obj, th0; max_iter=300, tol=1e-8)
    theta = 10.0 .^ opt_log_th
    
    ret_dict = Dict(:q_wpm => noise_current.q_wpm, 
                    :q_wfm => noise_current.q_wfm, 
                    :q_rwfm => noise_current.q_rwfm, 
                    :q_irwfm => noise_current.q_irwfm)
    for (i, k) in enumerate(opt_keys)
        ret_dict[k] = theta[i]
    end
    
    return ClockNoiseParams(ret_dict[:q_wpm], ret_dict[:q_wfm], ret_dict[:q_rwfm], ret_dict[:q_irwfm])
end
