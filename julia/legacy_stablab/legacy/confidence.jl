# Confidence interval calculations
# Translated from MATLAB stablab/+stablab/compute_ci.m and calculate_edf.m

using Statistics, Distributions

"""
    compute_ci(result::DeviationResult{T}, confidence_level::T=0.683; method::String="full") where T<:Real

Compute confidence intervals for stability deviations.

# Arguments
- `result`: DeviationResult struct containing deviation values and metadata
- `confidence_level`: Confidence level (default: 0.683 for 68.3%)
- `method`: Computation method ("full" or "simple")
  - "full": EDF-based chi-squared intervals with noise identification (default)
  - "simple": Basic statistical errors σ/√n (matches AllanTools)

# Returns
Updated DeviationResult with confidence intervals computed.

# Method (full)
1. Calculate equivalent degrees of freedom (EDF) for each tau point
2. Use chi-squared confidence intervals when EDF is available
3. Fall back to Gaussian intervals with Kn factors when EDF unavailable

# Method (simple)
1. Use simple statistical standard error: σ/√n
2. Apply Gaussian confidence intervals with z-scores

# References
- NIST SP1065 Appendix A (EDF calculation)
- Riley & Howe frequency stability analysis
"""
function compute_ci(result::DeviationResult{T}, confidence_level::T=0.683; method::String="full") where T<:Real
    dev = result.deviation
    alpha = result.alpha
    N = fill(result.N, length(dev))  # Original data length for all tau
    
    L = length(dev)
    ci = Matrix{T}(undef, L, 2)
    edf = Vector{T}(undef, L)
    
    if method == "simple"
        # Simple statistical errors like AllanTools: σ/√n
        z = quantile(Normal(), 1 - (1 - confidence_level)/2)
        
        for k in 1:L
            # Use effective sample count (neff) like AllanTools
            n_eff = result.neff[k]
            if n_eff <= 0
                n_eff = 1  # Prevent division by zero
            end
            
            # Simple statistical standard error
            stderr = dev[k] / sqrt(n_eff)
            margin = z * stderr
            
            ci[k, 1] = dev[k] - margin
            ci[k, 2] = dev[k] + margin
            edf[k] = Float64(n_eff)  # Use effective sample count as "EDF"
        end
        
        # Return updated result
        return DeviationResult(
            result.tau, dev, edf, ci, alpha, result.neff,
            result.tau0, result.N, result.method, confidence_level
        )
    end
    
    # Full EDF-based method (original implementation)
    
    # Loop over each tau point
    for k in 1:L
        # Calculate EDF based on deviation type and parameters
        edf_val = calculate_edf_for_method(result.method, alpha[k], result.tau[k], 
                                          result.tau0, N[k])
        edf[k] = edf_val
        
        if isnan(edf_val) || edf_val <= 0
            # Fallback: Gaussian CI using Kn factor and sample count
            Kn = kn_from_alpha(alpha[k])
            z = quantile(Normal(), 1 - (1 - confidence_level)/2)
            margin = Kn * dev[k] * z / sqrt(N[k])
            ci[k, 1] = dev[k] - margin
            ci[k, 2] = dev[k] + margin
        else
            # Chi-squared confidence intervals using EDF
            alpha_chi = 1 - confidence_level
            chi2_lo = quantile(Chisq(edf_val), alpha_chi/2)
            chi2_hi = quantile(Chisq(edf_val), 1 - alpha_chi/2)
            ci[k, 1] = dev[k] * sqrt(edf_val / chi2_hi)
            ci[k, 2] = dev[k] * sqrt(edf_val / chi2_lo)
        end
    end
    
    # Return updated result with computed confidence intervals and EDF
    return DeviationResult(
        result.tau, dev, edf, ci, alpha, result.neff,
        result.tau0, result.N, result.method, confidence_level
    )
end

"""
    calculate_edf_for_method(method, alpha, tau, tau0, N)

Calculate equivalent degrees of freedom based on deviation method and parameters.
"""
function calculate_edf_for_method(method::String, alpha::Int, tau::Real, 
                                tau0::Real, N::Int)
    m = round(Int, tau / tau0)  # Averaging factor
    
    if method in ["adev"]
        # Allan deviation (non-overlapping): d=2, F=m (unmodified), S=1 (non-overlapped) 
        return calculate_edf(alpha, 2, m, m, 1, N)
        
    elseif method in ["oadev"] 
        # Overlapping Allan deviation: d=2, F=m (unmodified), S=m (overlapped)
        return calculate_edf(alpha, 2, m, m, m, N)
        
    elseif method in ["mdev"]  
        # Modified Allan deviation: d=2, F=1 (modified), S=1 (non-overlapped)
        return calculate_edf(alpha, 2, m, 1, 1, N)
        
    elseif method in ["hdev"]
        # Hadamard deviation (non-overlapping): d=3, F=m (unmodified), S=1 (non-overlapped)
        return calculate_edf(alpha, 3, m, m, 1, N)
        
    elseif method in ["ohdev"]
        # Overlapping Hadamard deviation: d=3, F=m (unmodified), S=m (overlapped)
        return calculate_edf(alpha, 3, m, m, m, N)
        
    elseif method in ["mhdev"]
        # Modified Hadamard deviation: d=3, F=1 (modified), S=1 (non-overlapped) 
        return calculate_edf(alpha, 3, m, 1, 1, N)
        
    elseif method in ["totdev"]
        # Total deviation - use special EDF formula
        T = (N - 1) * tau0  # Record duration
        return totaldev_edf("totvar", alpha, T, tau)
        
    elseif method in ["mtotdev"]
        # Modified total deviation
        T = (N - 1) * tau0
        return totaldev_edf("mtot", alpha, T, tau)
        
    elseif method in ["htotdev"] 
        # Hadamard total deviation
        T = (N - 1) * tau0
        return totaldev_edf("htot", alpha, T, tau)
        
    elseif method in ["mhtotdev"]
        # Modified Hadamard total deviation
        T = (N - 1) * tau0
        return totaldev_edf("mhtot", alpha, T, tau)
        
    elseif method in ["tdev", "ldev"]
        # Time-domain deviations: use same EDF as their base deviations
        base_method = method == "tdev" ? "mdev" : "mhdev"
        return calculate_edf_for_method(base_method, alpha, tau, tau0, N)
        
    else
        # Unknown method - return NaN to trigger fallback
        @warn "Unknown method: $method, using Gaussian fallback"
        return NaN
    end
end

"""
    calculate_edf(alpha, d, m, F, S, N)

Core EDF calculation using NIST SP1065 formulation.

# Arguments  
- `alpha`: Frequency noise exponent (-4 to 2)
- `d`: Order of phase difference (1, 2, or 3)
- `m`: Averaging factor (tau/tau0)
- `F`: Filter factor (1: modified, m: unmodified)  
- `S`: Stride factor (1: non-overlapped, m: overlapped)
- `N`: Number of phase data points
"""
function calculate_edf(alpha::Int, d::Int, m::Int, F::Int, S::Int, N::Int)
    # Check restriction
    if alpha + 2*d <= 1
        return NaN  # Invalid parameters
    end
    
    # Initial steps
    L = m/F + m*d  # Filter length
    if N < L
        return NaN  # Not enough data
    end
    
    M = 1 + floor(Int, S*(N - L)/m)  # Number of summands
    J = min(M, (d + 1)*S)            # Truncation parameter
    
    # Compute sz(0, F, alpha, d)
    sz0 = compute_sz(0, F, alpha, d)
    
    # Compute BasicSum
    basic_sum = compute_basic_sum(J, M, S, F, alpha, d)
    
    # Calculate EDF
    if basic_sum > 0
        return M * sz0^2 / basic_sum
    else
        return NaN
    end
end

"""
    totaldev_edf(var_type, alpha, T, tau)

Specialized EDF calculation for total deviation types.
"""
function totaldev_edf(var_type::String, alpha::Int, T::Real, tau::Real)
    if var_type == "totvar"
        b, c = coeff_totvar(alpha)
        return b * (T / tau) - c
    elseif var_type == "mtot"
        b, c = coeff_mtot(alpha)
        return b * (T / tau) - c
    elseif var_type == "htot"
        b0, b1 = coeff_htot(alpha)
        return (T / tau) / (b0 + b1 * (tau / T))
    elseif var_type == "mhtot"
        b, c = coeff_mhtot(alpha)
        return b * (T/tau) - c
    else
        return NaN
    end
end

"""
    kn_from_alpha(alpha)

Lookup Kn factor based on rounded α for Gaussian fallback.
"""
function kn_from_alpha(alpha::Int)
    if alpha == -2      # Random walk FM
        return 0.75
    elseif alpha == -1  # Flicker FM  
        return 0.77
    elseif alpha == 0   # White FM
        return 0.87
    elseif alpha == 1   # Flicker PM
        return 0.99
    elseif alpha == 2   # White PM
        return 0.99
    else
        return 1.10  # Conservative fallback
    end
end

# -------------------- EDF Helper Functions --------------------

function compute_sw(t::Real, alpha::Int)
    t_abs = abs(t)
    if alpha == 2
        return -t_abs
    elseif alpha == 1
        return t^2 * log(max(t_abs, eps()))
    elseif alpha == 0  
        return t_abs^3
    elseif alpha == -1
        return -t^4 * log(max(t_abs, eps()))
    elseif alpha == -2
        return -t_abs^5
    elseif alpha == -3
        return t^6 * log(max(t_abs, eps()))
    elseif alpha == -4
        return t_abs^7
    else
        return NaN
    end
end

function compute_sx(t::Real, F::Int, alpha::Int)
    if F > 100 && alpha <= 0  # Large F approximation
        return compute_sw(t, alpha + 2)
    else
        return F^2 * (2*compute_sw(t, alpha) -
                      compute_sw(t - 1/F, alpha) -
                      compute_sw(t + 1/F, alpha))
    end
end

function compute_sz(t::Real, F::Int, alpha::Int, d::Int)
    if d == 1
        return 2*compute_sx(t, F, alpha) - compute_sx(t-1, F, alpha) - compute_sx(t+1, F, alpha)
    elseif d == 2
        return 6*compute_sx(t, F, alpha) - 4*compute_sx(t-1, F, alpha) - 4*compute_sx(t+1, F, alpha) +
               compute_sx(t-2, F, alpha) + compute_sx(t+2, F, alpha)
    elseif d == 3
        return 20*compute_sx(t, F, alpha) - 15*compute_sx(t-1, F, alpha) - 15*compute_sx(t+1, F, alpha) +
               6*compute_sx(t-2, F, alpha) + 6*compute_sx(t+2, F, alpha) -
               compute_sx(t-3, F, alpha) - compute_sx(t+3, F, alpha)
    else
        return NaN
    end
end

function compute_basic_sum(J::Int, M::Int, S::Int, F::Int, alpha::Int, d::Int)
    sz0 = compute_sz(0, F, alpha, d)
    basic_sum = sz0^2
    
    for j in 1:(J-1)
        szj = compute_sz(j/S, F, alpha, d)
        basic_sum += 2 * (1 - j/M) * szj^2
    end
    
    if J <= M
        szJ = compute_sz(J/S, F, alpha, d)
        basic_sum += (1 - J/M) * szJ^2
    end
    
    return basic_sum
end

# -------------------- Coefficient Tables --------------------

function coeff_totvar(alpha::Int)
    if alpha == 0; return (1.50, 0.00)      # White FM
    elseif alpha == -1; return (1.17, 0.22) # Flicker FM  
    elseif alpha == -2; return (0.93, 0.36) # Random walk FM
    else; return (NaN, NaN); end
end

function coeff_mtot(alpha::Int)
    if alpha == 2; return (1.90, 2.10)      # White PM
    elseif alpha == 1; return (1.20, 1.40)  # Flicker PM
    elseif alpha == 0; return (1.10, 1.20)  # White FM
    elseif alpha == -1; return (0.85, 0.50) # Flicker FM
    elseif alpha == -2; return (0.75, 0.31) # Random walk FM
    else; return (NaN, NaN); end
end

function coeff_mhtot(alpha::Int)
    if alpha == 2; return (3.904, 9.640)    # White PM
    elseif alpha == 1; return (2.656, 11.093) # Flicker PM
    elseif alpha == 0; return (2.275, 8.701)  # White FM
    elseif alpha == -1; return (1.964, 4.908) # Flicker FM
    elseif alpha == -2; return (1.572, 4.534) # Random walk FM
    else; return (NaN, NaN); end
end

function coeff_htot(alpha::Int)
    if alpha == 0; return (0.546, 1.41)     # White FM
    elseif alpha == -1; return (0.667, 2.00) # Flicker FM
    elseif alpha == -2; return (0.909, 1.00) # RWFM
    else; return (NaN, NaN); end
end

# -------------------- Bias Correction --------------------

"""
    bias_correction(alpha, var_type, tau, T)

Bias factor B(α) for TOTVAR, MTOT, and HTOT corrections.

Translated from MATLAB stablab/+stablab/bias_correction.m

# Arguments
- `alpha`: noise exponent α (scalar or vector)
- `var_type`: "totvar", "mtot", or "htot"
- `tau`: averaging time τ (same shape as alpha)
- `T`: record duration T = N·τ₀ (scalar)

# Returns
Bias factor for each τ, same size as alpha

# References
- NIST SP1065 §§5.11–5.13
- Greenhall & Riley, "Uncertainty of Stability Variances", PTTI 2003
"""
function bias_correction(alpha::Union{Int,Vector{Int}}, var_type::String, 
                        tau::Union{Real,Vector{<:Real}}, T::Real)
    
    # Ensure alpha and tau are the same size
    if isa(alpha, Int)
        alpha = [alpha]
    end
    if isa(tau, Real)
        tau = [tau]
    end
    
    B = ones(Float64, max(length(alpha), length(tau)))
    
    # Expand to same length if needed
    if length(alpha) == 1 && length(tau) > 1
        alpha = fill(alpha[1], length(tau))
    elseif length(tau) == 1 && length(alpha) > 1
        tau = fill(tau[1], length(alpha))
    end
    
    var_type_lower = lowercase(var_type)
    
    if var_type_lower == "totvar"
        # TOTVAR: B(τ) = 1 - a * (τ / T)
        for k in eachindex(alpha)
            if alpha[k] == -1
                a = 1 / (3 * log(2))  # ≈ 0.481
                B[k] = 1 - a * (tau[k] / T)
            elseif alpha[k] == -2
                a = 0.75
                B[k] = 1 - a * (tau[k] / T)
            else
                B[k] = 1.0  # no correction for other α
            end
        end
        
    elseif var_type_lower == "mtot"
        # MTOTDEV bias factors from SP1065 Table 11
        bias_table = Dict(
            2 => 1.06,
            1 => 1.17,
            0 => 1.27,
            -1 => 1.30,
            -2 => 1.31
        )
        for k in eachindex(alpha)
            if haskey(bias_table, alpha[k])
                B[k] = bias_table[alpha[k]]
            else
                B[k] = 1.0
            end
        end
        
    elseif var_type_lower == "htot"
        # HTOT bias: B = 1 / (1 + a), using a(α) from Table 1 (FCS 2001)
        a_table = Dict(
            0 => -0.005,
            -1 => -0.149,
            -2 => -0.229,
            -3 => -0.283,
            -4 => -0.321
        )
        for k in eachindex(alpha)
            if haskey(a_table, alpha[k])
                a = a_table[alpha[k]]
                B[k] = 1 / (1 + a)  # bias factor = 1 / (1 + a)
            else
                B[k] = 1.0  # fallback if α not listed
            end
        end
        
    else
        error("Unknown var_type \"$var_type\". Use \"totvar\", \"mtot\", or \"htot\".")
    end
    
    return B
end

"""
    apply_bias_correction!(result::DeviationResult)

Apply bias correction to total deviation results in-place.
"""
function apply_bias_correction!(result::DeviationResult)
    method = result.method
    
    # Only apply to total deviation types
    if method in ["totdev", "mtotdev", "htotdev", "mhtotdev"]
        T = (result.N - 1) * result.tau0  # Record duration
        
        # Map method names to var_type
        var_type_map = Dict(
            "totdev" => "totvar",
            "mtotdev" => "mtot", 
            "htotdev" => "htot",
            "mhtotdev" => "htot"  # Use htot correction for mhtotdev
        )
        
        var_type = var_type_map[method]
        
        # Calculate bias correction factors
        B = bias_correction(result.alpha, var_type, result.tau, T)
        
        # Apply correction: σ_corrected = σ_measured / B
        result.deviation ./= B
        
        # If confidence intervals exist, correct them too
        if !any(isnan, result.ci)
            result.ci ./= B
        end
    end
    
    return result
end