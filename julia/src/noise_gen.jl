# noise_gen.jl — Composite power-law phase noise generator
# Kasdin & Walter (1992), "Discrete simulation of power law noise", IEEE FCS.
#
# Amplitude convention: bin-k one-sided spectral amplitude
#     |A_k|² = h_α · f_k^α · Δf
# where Δf = 1 / (N·τ₀).  IFFT of the conjugate-symmetric full spectrum
# produces frequency-domain data y(t) with two-sided PSD S_y(f) = h_α · |f|^α.
# Phase is cumsum(y) * τ₀ (in seconds).
#
# Works for any real α (integer or fractional, including FPM α=+1 and FFM α=−1).

using Random
using FFTW

"""
    generate_power_law_noise(α, h, N, τ₀; seed) → Vector{Float64}

Synthesise N phase samples (in seconds) with frequency PSD S_y(f) = h · |f|^α.

# Arguments
- `α`: power-law exponent (S_y(f) ∝ f^α)
- `h`: coefficient h_α (dimensional, depends on α; e.g. for α=0 units are Hz⁻¹)
- `N`: number of output samples (must be even)
- `τ₀`: sampling interval (seconds)
- `seed`: integer RNG seed

# Returns
Phase time series `x(t)` in seconds, length N.
"""
function generate_power_law_noise(α::Real, h::Real, N::Int, τ₀::Real; seed::Integer)
    iseven(N) || throw(ArgumentError("N must be even for FFT symmetry"))
    N >= 4     || throw(ArgumentError("N must be ≥ 4"))
    h > 0      || throw(ArgumentError("h must be positive"))
    τ₀ > 0     || throw(ArgumentError("τ₀ must be positive"))

    rng = Xoshiro(seed)
    Δf  = 1.0 / (N * τ₀)

    # One-sided frequency grid: k = 1..N/2 (DC excluded, Nyquist forced real)
    Nhalf = N ÷ 2
    half  = zeros(ComplexF64, Nhalf + 1)

    # DC bin — set to 0 (no finite-power at f=0 for these power laws)
    half[1] = 0.0

    # Positive-freq bins 1..(Nhalf-1) — complex Gaussian amplitude with variance h·f^α·Δf/2
    # Factor of 1/2: each bin appears at both +f and -f in the mirrored spectrum.
    # Total variance per frequency interval = 2 × (h·f^α·Δf/2) = h·f^α·Δf (one-sided).
    for k in 1:(Nhalf - 1)
        f_k     = k * Δf
        var_k   = h * f_k^α * Δf / 2     # half-power per side of spectrum
        σ_k     = sqrt(var_k / 2)         # split across real+imag for E[|A|²]=var_k
        half[k + 1] = σ_k * (randn(rng) + im * randn(rng))
    end

    # Nyquist bin — forced real so IFFT output is real-valued; also half-power
    f_ny        = Nhalf * Δf
    var_ny      = h * f_ny^α * Δf / 2
    half[end]   = sqrt(var_ny) * randn(rng)

    # Mirror to conjugate-symmetric full spectrum
    full              = Vector{ComplexF64}(undef, N)
    full[1:Nhalf + 1] = half
    @inbounds for k in 1:(Nhalf - 1)
        full[N - k + 1] = conj(half[k + 1])
    end

    # IFFT — FFTW convention: ifft(X)[n] = (1/N) Σ X[k] e^{+2πi nk/N}
    # Scale: to get continuous-time PSD match, multiply by N (undo the 1/N).
    y_freq = real.(ifft(full)) .* N

    # Integrate frequency → phase; multiply by τ₀ (discrete integration)
    return cumsum(y_freq) .* τ₀
end

"""
    generate_composite_noise(h_coeffs, N, τ₀; seed) → Vector{Float64}

Synthesise composite phase noise by summing independent power-law components.

# Arguments
- `h_coeffs::AbstractDict{<:Real,<:Real}`: α → h_α mapping.
- `N`: number of samples (even).
- `τ₀`: sampling interval (seconds).
- `seed`: base RNG seed; each component uses seed + offset so streams never collide.

# Returns
Phase time series `x(t)` in seconds, length N.
"""
function generate_composite_noise(h_coeffs::AbstractDict{<:Real,<:Real}, N::Int,
                                   τ₀::Real; seed::Integer)
    isempty(h_coeffs) && throw(ArgumentError("h_coeffs must be non-empty"))
    x = zeros(Float64, N)
    # Sort α for deterministic seed offset ordering
    for (offset, α) in enumerate(sort(collect(keys(h_coeffs))))
        h = h_coeffs[α]
        x .+= generate_power_law_noise(α, h, N, τ₀; seed = seed + offset - 1)
    end
    return x
end
