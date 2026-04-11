# Noise identification and characterization utilities
# Translated from MATLAB stablab/+stablab/noise_id.m

using Statistics
using FFTW: irfft, ifft, rfftfreq
using Random

"""
    noise_id(x, m_list, data_type, dmin=0, dmax=2)

Dominant power-law noise estimator from time series data.

# Arguments
- `x`: Phase or frequency data (column vector)
- `m_list`: List of averaging factors (τ = m·τ₀)
- `data_type`: "phase" or "freq"  
- `dmin`: Minimum differencing depth (default = 0)
- `dmax`: Maximum differencing depth (default = 2)

# Returns
- `alpha_list`: Estimated α values at each τ

# Method
For each m, use lag-1 autocorrelation estimator when N_eff ≥ 30,
otherwise use fallback via B1 ratio and R(n) test.

# References
- NIST SP1065 Section 5.6
- Riley & Howe frequency stability analysis
"""
function noise_id(x::Vector{T}, m_list::Vector{Int}, data_type::String="phase", 
                 dmin::Int=0, dmax::Int=2) where T<:Real
    # Preprocess data: remove outliers and detrend
    x_clean = preprocess_x(x)
    alpha_list = fill(NaN, length(m_list))
    
    for (k, m) in enumerate(m_list)
        # Estimate number of usable points after averaging
        N_eff = floor(Int, length(x_clean) / m)
        
        try
            if N_eff >= 30
                # Use lag-1 ACF method
                alpha, _, _, _ = noise_id_lag1acf(x_clean, m, data_type, dmin, dmax)
            else
                # Use B1 ratio + R(n) fallback method
                alpha, _, _ = noise_id_b1rn(x_clean, m, data_type)
            end
            alpha_list[k] = round(Int, alpha)
        catch err
            @warn "Noise ID failed for m = $m: $(err)"
            alpha_list[k] = NaN
        end
    end
    
    return alpha_list
end

"""
    preprocess_x(x)

Remove outliers (>5σ) and linear trend from data.
"""
function preprocess_x(x::Vector{T}) where T<:Real
    x = vec(x)  # Ensure column vector
    
    # Remove outliers >5σ
    x_mean = mean(x)
    x_std = std(x)
    z_scores = abs.((x .- x_mean) ./ x_std)
    x_clean = x[z_scores .< 5.0]
    
    # Remove linear trend (frequency drift)
    return detrend_linear(x_clean)
end

"""
    noise_id_lag1acf(x, m, data_type, dmin=0, dmax=2)

Lag-1 autocorrelation function method for noise identification.

Returns (alpha, alpha_int, d, rho) where:
- alpha: Estimated noise exponent
- alpha_int: Rounded integer estimate  
- d: Final differencing order used
- rho: Fractional integration index
"""
function noise_id_lag1acf(x::Vector{T}, m::Int, data_type::String, 
                         dmin::Int=0, dmax::Int=2) where T<:Real
    # Step 1: Preprocess by data type
    if lowercase(data_type) == "phase"
        if m > 1
            x = x[1:m:end]  # Decimate
        end
        x = detrend_quadratic(x)  # Remove quadratic drift
    elseif lowercase(data_type) == "freq"
        N = floor(Int, length(x) / m) * m
        x = x[1:N]
        x = reshape(x, m, :)
        x = vec(mean(x, dims=1))  # Average in blocks
        x = detrend_linear(x)
    else
        error("data_type must be 'phase' or 'freq'")
    end
    
    # Step 2: Differencing loop
    d = 0
    while true
        r1 = compute_lag1_acf(x)  # Lag-1 autocorrelation
        rho = r1 / (1 + r1)       # Fractional integration index
        
        if d >= dmin && (rho < 0.25 || d >= dmax)
            p = -2 * (rho + d)    # Spectral slope
            alpha = p + 2 * (lowercase(data_type) == "phase" ? 1 : 0)
            alpha_int = round(Int, alpha)
            return (alpha, alpha_int, d, rho)
        else
            x = diff(x)
            d = d + 1
            if length(x) < 5
                error("Data too short after differencing")
            end
        end
    end
end

"""
    compute_lag1_acf(x)

Compute lag-1 autocorrelation coefficient.
"""
function compute_lag1_acf(x::Vector{T}) where T<:Real
    x = vec(x) .- mean(x)
    
    if all(x .== 0)
        return NaN
    end
    
    x0 = x[1:end-1]
    x1 = x[2:end]
    
    return sum(x0 .* x1) / sum(x .^ 2)
end

"""
    noise_id_b1rn(x_full, m, data_type)

B1 ratio and R(n) fallback method for noise identification.

Returns (alpha_int, mu_best, B1_obs) where:
- alpha_int: Integer noise exponent estimate
- mu_best: Best-fit slope parameter
- B1_obs: Observed B1 ratio
"""
function noise_id_b1rn(x_full::Vector{T}, m::Int, data_type::String) where T<:Real
    x_full = vec(x_full)
    
    if lowercase(data_type) == "phase"
        # Decimate and detrend phase data
        x_dec = x_full[1:m:end]
        x_dec = detrend_quadratic(x_dec)
        tau = m
        avar_val = simple_avar(x_dec, tau)
        N_avar = floor(Int, length(x_dec) - 2)
        
        # Classical variance of averaged differences
        dx = diff(x_full)
        N = floor(Int, length(dx) / m) * m
        if N < m
            return (NaN, NaN, NaN)
        end
        dx = dx[1:N]
        y_blocks = reshape(dx, m, :)
        y_avg = vec(mean(y_blocks, dims=1))
        var_classical = var(y_avg; corrected=false)
        
    elseif lowercase(data_type) == "freq"
        N = floor(Int, length(x_full) / m) * m
        if N < 2*m
            return (NaN, NaN, NaN)
        end
        x = reshape(x_full[1:N], m, :)
        y_avg = vec(mean(x, dims=1))
        y_avg = detrend_linear(y_avg)
        dy = diff(y_avg)
        var_classical = var(y_avg; corrected=false)
        avar_val = sum(dy .^ 2) / (2 * (length(y_avg) - 1))
        N_avar = length(y_avg)
    else
        error("Unsupported data_type: use 'phase' or 'freq'")
    end
    
    # Compute observed B1 ratio
    B1_obs = var_classical / avar_val
    
    # Define noise types (ordered from high to low μ for checking)
    mu_list = [1, 0, -1, -2]  # RWFM, FLFM, WHFM, WHPM
    alpha_list = [-2, -1, 0, 2]
    
    # Calculate theoretical B1 values
    b1_vals = [b1_theory(N_avar, mu) for mu in mu_list]
    
    # Decision boundaries using geometric means (NIST approach)
    mu_best = mu_list[end]  # Default to lowest μ
    alpha_int = alpha_list[end]
    
    for i in 1:length(mu_list)-1
        boundary = sqrt(b1_vals[i] * b1_vals[i+1])
        
        if B1_obs > boundary
            mu_best = mu_list[i]
            alpha_int = alpha_list[i]
            break
        end
    end
    
    # Refine α = 2 vs 1 using R(n) when needed (FLPM vs WHPM)
    if mu_best == -2 && lowercase(data_type) == "phase"
        adev_val = sqrt(avar_val)
        mdev_val = simple_mdev(x_full, tau, 1.0)
        Rn_obs = (mdev_val / adev_val)^2
        R_hi = rn_theory(m, 0)   # α = 2 (WHPM)
        R_lo = rn_theory(m, -1)  # α = 1 (FLPM)
        if Rn_obs > sqrt(R_hi * R_lo)
            alpha_int = 1  # Flicker PM
        else
            alpha_int = 2  # White PM
        end
    end
    
    return (alpha_int, mu_best, B1_obs)
end

"""
    b1_theory(N, mu)

Theoretical B1 values from slope μ.
"""
function b1_theory(N::Int, mu::Int)
    if mu == 2
        return N * (N + 1) / 6
    elseif mu == 1
        return N / 2
    elseif mu == 0
        return N * log(N) / (2 * (N - 1) * log(2))
    elseif mu == -1
        return 1.0
    elseif mu == -2
        return (N^2 - 1) / (1.5 * N * (N - 1))
    else
        return (N * (1 - N^mu)) / (2 * (N - 1) * (1 - 2^mu))
    end
end

"""
    rn_theory(af, b)

Theoretical R(n) values for noise classification.
"""
function rn_theory(af::Int, b::Int)
    if b == 0
        return af^(-1)  # White PM
    elseif b == -1
        avar = (1.038 + 3 * log(2 * π * 0.5 * af)) / (4 * π^2)
        mvar = 3 * log(256 / 27) / (8 * π^2)
        return mvar / avar  # Flicker PM
    else
        return 1.0
    end
end

"""
    simple_avar(x, m)

Simple Allan variance calculation without noise identification.
"""
function simple_avar(x::Vector{T}, m::Int) where T<:Real
    N = length(x)
    L = N - 2*m + 1
    if L <= 0
        return NaN
    end
    
    # Second differences: x(n+2m) - 2x(n+m) + x(n)
    d2 = x[1+2*m:N] - 2*x[1+m:N-m] + x[1:L]
    return mean(d2.^2) / (2 * m^2)
end

"""
    simple_mdev(x, m, tau0)

Simple Modified Allan deviation without calling full mdev function.
"""
function simple_mdev(x::Vector{T}, m::Int, tau0::Real) where T<:Real
    N = length(x)
    L = N - 3*m + 1
    if L <= 0
        return NaN
    end
    
    # Moving averages via cumulative sum
    S = cumsum([0; x])  # Prefix sum
    s1 = S[1+m:L+m]     - S[1:L]
    s2 = S[1+2*m:L+2*m] - S[1+m:L+m] 
    s3 = S[1+3*m:L+3*m] - S[1+2*m:L+2*m]
    d = s3 - 2*s2 + s1
    
    mvar = mean(d.^2) / (2 * m^2 * tau0^2)
    return sqrt(mvar)
end

"""
    detrend_quadratic(x)

Remove quadratic trend from data (for phase data).
"""
function detrend_quadratic(x::Vector{T}) where T<:Real
    N = length(x)
    if N < 3
        return x
    end
    
    # Design matrix for quadratic fit: [1, t, t²]
    t = collect(1:N)
    A = hcat(ones(N), t, t.^2)
    
    # Least squares fit and removal
    coeffs = A \ x
    trend = A * coeffs
    
    return x - trend
end

"""
    timmer_koenig_from_psd(f_nodes, h, alpha, duration, timestep; output="phase", seed=nothing)

Generate time-domain clock noise from a piecewise power-law one-sided PSD ``S_y(f)`` using
    Timmer–Koenig-style Fourier synthesis. **Frequencies match the DFT**: ``f_k`` are
    ``rfftfreq(n, 1/\\tau_0)`` in Julia (equivalent to ``numpy.fft.rfftfreq(n, \\tau_0)``, which
    uses sample spacing), not an ad-hoc ``linspace``, so
    ``S_y`` and ``S_x`` are evaluated at the bins used by ``irfft`` / real FFT — see
    Timmer & Koenig (1995). AllanTools does not ship this routine; it uses Kasdin’s method
    elsewhere (`noise_kasdin`); this implementation follows the usual complex-Gaussian half-spectrum
    construction with a **real Nyquist bin** when ``n`` is even.

    The model uses one-sided fractional-frequency PSD ``S_y(f) = h_i f^{α_i}`` on each segment;
    phase PSD is ``S_x(f) = S_y(f) / (2πf)^2``. Interior bins:
    ``Z_k = \\sqrt{S_x(f_k)}/2 · (N(0,1) + i N(0,1))``; at Nyquist (even ``n`` only)
    ``Z_{N/2} = \\sqrt{S_x(f_{N/2})/2} · N(0,1)``. Then ``x = \\mathrm{irfft}(Z) \\sqrt{(n-1)/dt}``.

# Arguments
- `f_nodes`: Break frequencies (Hz), ascending; length typically `length(h) - 1` (or empty for one segment).
- `h`, `alpha`: Coefficients and slopes per segment; same length.
- `duration`: Total duration of the series (seconds).
- `timestep`: Sample interval `dt` (seconds); sample rate is `1/dt`.
- `output`: `"phase"` returns phase/time fluctuation `x(t)` [s]; `"freq"` returns fractional frequency `y(t) = dx/dt`.
- `seed`: Optional RNG seed (`Random.seed!`) for reproducible synthesis.

# Returns
- Vector of length `n = duration/timestep` (phase or frequency noise as selected).

# References
- J. Timmer & M. Koenig, "On generating power law noise," *Astronomy and Astrophysics* 300, 707–710 (1995).
"""
function timmer_koenig_from_psd(
    f_nodes::AbstractVector{<:Real},
    h::AbstractVector{<:Real},
    alpha::AbstractVector{<:Real},
    duration::Real,
    timestep::Real;
    output::AbstractString = "phase",
    seed::Union{Nothing,Integer} = nothing,
)
    duration > 0 || throw(ArgumentError("duration must be positive."))
    timestep > 0 || throw(ArgumentError("timestep must be positive."))
    length(h) == length(alpha) || throw(ArgumentError("h and alpha must have the same length."))
    nfn = length(f_nodes)
    nfn == 0 || nfn == length(h) - 1 ||
        throw(ArgumentError("Expected length(f_nodes) == length(h) - 1 (or 0 for a single segment)."))

    isnothing(seed) || Random.seed!(Int(seed))

    h_ext = vcat(collect(Float64, h), Float64(h[end]))
    alpha_ext = vcat(collect(Float64, alpha), Float64(alpha[end]))

    n = Int(duration / timestep)
    n < 4 && throw(ArgumentError("duration/timestep too small; need at least 4 samples."))

    # Match `numpy.fft.rfftfreq(n, dt)`: Julia's `rfftfreq(n, fs)` takes sampling rate fs=1/dt.
    frequencies = collect(rfftfreq(n, inv(timestep)))

    Sy = zeros(Float64, length(frequencies))
    @inbounds for i in eachindex(frequencies)
        f = frequencies[i]
        if abs(f) < 1e-30
            Sy[i] = 0.0
            continue
        end
        if nfn > 0 && f < f_nodes[1]
            Sy[i] = h_ext[1] * f^alpha_ext[1]
        elseif nfn > 0 && f > f_nodes[end]
            Sy[i] = h_ext[end] * f^alpha_ext[end]
        elseif nfn == 0
            Sy[i] = h_ext[1] * f^alpha_ext[1]
        else
            for j in 1:(nfn - 1)
                if f_nodes[j] <= f < f_nodes[j + 1]
                    Sy[i] = h_ext[j + 1] * f^alpha_ext[j + 1]
                    break
                end
            end
        end
    end

    Sx = zeros(Float64, length(frequencies))
    @inbounds for i in eachindex(frequencies)
        f = frequencies[i]
        if abs(f) < 1e-30
            Sx[i] = 0.0
        else
            Sx[i] = Sy[i] / (2.0 * π * f)^2
        end
    end

    nh = n ÷ 2
    Z = zeros(ComplexF64, nh + 1)
    Z[1] = 0.0 + 0.0im
    if iseven(n)
        @inbounds for k in 2:nh
            Z[k] = (sqrt(Sx[k]) / 2.0) * (randn() + im * randn())
        end
        Z[nh + 1] = sqrt(Sx[nh + 1] / 2.0) * randn()
    else
        @inbounds for k in 2:(nh + 1)
            Z[k] = (sqrt(Sx[k]) / 2.0) * (randn() + im * randn())
        end
    end

    x = irfft(Z, n) .* sqrt((n - 1) / timestep)

    out = lowercase(output)
    if out == "phase"
        return x
    elseif out == "freq"
        return vcat(0.0, diff(x) ./ timestep)
    else
        throw(ArgumentError("output must be \"phase\" or \"freq\"."))
    end
end

"""
    clock_noise(h_coeffs, duration, timestep; output=:phase, seed=nothing)

Generate a clock time series whose fractional-frequency PSD is
`Sᵧ(f) = Σᵢ hᵢ f^αᵢ` by Timmer–Koenig synthesis on the summed PSD.

# Arguments
- `h_coeffs`: `Dict{Int,Float64}` mapping exponent α → coefficient hα.
  e.g. `Dict(-2 => 2e-25, -1 => 1e-24, 0 => 5e-26, 1 => 0.0, 2 => 0.0)`
- `duration`: total time series length (s).
- `timestep`: sampling interval dt (s).
- `output`:   `:phase` for x(t) [s], `:freq` for y(t) = dx/dt.
- `seed`:     optional integer for reproducibility.

Each component uses `seed + α` so adding/removing a component doesn't
change the realisations of the others.
"""
function clock_noise(
    h_coeffs::Dict{Int,Float64},
    duration::Real,
    timestep::Real;
    output::Symbol = :phase,
    seed::Union{Int,Nothing} = nothing
)
    duration > 0 && timestep > 0 ||
        throw(ArgumentError("duration and timestep must be positive."))
    output ∈ (:phase, :freq) ||
        throw(ArgumentError("output must be :phase or :freq."))

    n = floor(Int, duration / timestep)
    n ≥ 4 || throw(ArgumentError("Need at least 4 samples."))

    f1 = 1.0 / ((n - 1) * timestep)
    fn = 1.0 / (2.0 * timestep)
    freqs = range(f1, fn; length = n ÷ 2 + 1)

    Sx_total = zeros(length(freqs))
    denom = (2π .* freqs).^2

    for (α, hα) in h_coeffs
        hα == 0.0 && continue
        Sx_total .+= hα .* freqs.^α ./ denom
    end
    Sx_total[1] = 0.0

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    X = zeros(ComplexF64, n)
    for j in 2:(n ÷ 2 + 1)
        X[j] = (sqrt(Sx_total[j]) / 2.0) * (randn(rng) + im * randn(rng))
    end
    for j in 2:(n ÷ 2)
        X[n - j + 2] = conj(X[j])
    end

    x = real.(ifft(X)) .* sqrt((n - 1) / timestep)

    if output == :phase
        return x
    else
        return vcat(0.0, diff(x) ./ timestep)
    end
end