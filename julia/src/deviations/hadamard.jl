# deviations/hadamard.jl — Hadamard deviation (HDEV, MHDEV, LDEV)

# ── HDEV ──────────────────────────────────────────────────────────────────────

"""
    hdev(x, tau0; m_list=nothing, data_type=:phase) → DeviationResult

Overlapping Hadamard deviation (OHDEV). Third-difference phase variance, SP1065 §4.6.

The kernel computes the overlapping estimator:
    HVAR(τ) = ⟨(x[i+3m] - 3x[i+2m] + 3x[i+m] - x[i])²⟩ / (6τ²)   # SP1065 HVAR

Hadamard deviation suppresses linear frequency drift, making it useful for
oscillators with significant aging or for characterising flicker walk FM.

# Arguments
- `x`:         phase data vector (seconds), length N ≥ 5
- `tau0`:      sampling interval (seconds)
- `m_list`:    averaging factors; auto-generated (octave-spaced) when `nothing`
- `data_type`: `:phase` (default) or `:freq` (fractional-frequency samples)

# Returns
`DeviationResult` with fields `tau`, `deviation`, `edf`, `ci`, `alpha`, `neff`.
CI is filled by the engine (chi-squared where EDF is finite, Gaussian fallback).
"""
function hdev(
    x         :: AbstractVector{<:Real},
    tau0      :: Real;
    m_list    :: Union{Nothing, AbstractVector{<:Integer}} = nothing,
    data_type :: Symbol = :phase,
)
    params = DevParams(
        "hdev",         # name
        4,              # min_factor: N/m ≥ 4
        3,              # d: third-difference order (Hadamard)
        m -> m,         # F_fn: F = m for unmodified
        0,              # dmin for noise_id
        2,              # dmax for noise_id
    )
    return engine(x, tau0, m_list, _hdev_kernel, params; data_type)
end

# Kernel: returns (variance, neff) — engine takes sqrt internally
# SP1065 HVAR: HVAR(τ) = mean(d3²) / (6m²τ₀²)
function _hdev_kernel(x::AbstractVector{<:Real}, m::Int, tau0::Real, x_cs::AbstractVector{<:Real})
    N = length(x)
    L = N - 3m
    L <= 0 && return (NaN, 0)
    d3 = @view(x[1+3m:end]) .- 3 .* @view(x[1+2m:end-m]) .+
         3 .* @view(x[1+m:end-2m]) .- @view(x[1:L])
    v = sum(abs2, d3) / (L * 6.0 * Float64(m)^2 * tau0^2)   # SP1065 HVAR
    return (v, L)
end

# ── MHDEV ─────────────────────────────────────────────────────────────────────

"""
    mhdev(x, tau0; m_list=nothing, data_type=:phase) → DeviationResult

Modified Hadamard deviation (MHDEV). Third differences with moving average
(cumsum trick), analogous to MDEV vs ADEV.

# Arguments
- `x`:         phase data vector (seconds), length N ≥ 5
- `tau0`:      sampling interval (seconds)
- `m_list`:    averaging factors; auto-generated (octave-spaced) when `nothing`
- `data_type`: `:phase` (default) or `:freq` (fractional-frequency samples)

# Returns
`DeviationResult` with fields `tau`, `deviation`, `edf`, `ci`, `alpha`, `neff`.
CI is filled by the engine (chi-squared where EDF is finite, Gaussian fallback).
"""
function mhdev(
    x         :: AbstractVector{<:Real},
    tau0      :: Real;
    m_list    :: Union{Nothing, AbstractVector{<:Integer}} = nothing,
    data_type :: Symbol = :phase,
)
    params = DevParams(
        "mhdev",        # name
        4,              # min_factor: N/m ≥ 4
        3,              # d: third-difference order (Hadamard)
        m -> 1,         # F_fn: F = 1 for modified
        0,              # dmin for noise_id
        2,              # dmax for noise_id
    )
    return engine(x, tau0, m_list, _mhdev_kernel, params; data_type)
end

# Kernel: returns (variance, neff) — engine takes sqrt internally
# SP1065: MHVAR = ⟨(s4 - 3s3 + 3s2 - s1)²⟩ / (6 * m^4 * tau0^2)
function _mhdev_kernel(x::AbstractVector{<:Real}, m::Int, tau0::Real, x_cs::AbstractVector{<:Real})
    N = length(x)
    Ne = N - 4m + 1
    Ne <= 0 && return (NaN, 0)
    # Identity: s4 - 3s3 + 3s2 - s1 = x_cs[i+4m] - 4x_cs[i+3m] + 6x_cs[i+2m] - 4x_cs[i+m] + x_cs[i]
    d = @view(x_cs[1+4m:Ne+4m]) .- 4 .* @view(x_cs[1+3m:Ne+3m]) .+
        6 .* @view(x_cs[1+2m:Ne+2m]) .- 4 .* @view(x_cs[1+m:Ne+m]) .+
        @view(x_cs[1:Ne])
    v = sum(abs2, d) / (Ne * 6.0 * Float64(m)^4 * tau0^2)
    return (v, Ne)
end

# ── LDEV ──────────────────────────────────────────────────────────────────────

"""
    ldev(x, tau0; m_list=nothing, data_type=:phase) → DeviationResult

Hadamard time deviation (LDEV). Wraps `mhdev` and scales by τ/√(10/3):
    LDEV(τ) = τ · MHDEV(τ) / √(10/3)

LDEV characterises time error stability in the Hadamard sense (units: seconds),
complementing MHDEV in the frequency domain.

Does NOT call engine directly; derives from MHDEV for consistency.

# Arguments
- `x`:         phase data vector (seconds), length N ≥ 5
- `tau0`:      sampling interval (seconds)
- `m_list`:    averaging factors; auto-generated (octave-spaced) when `nothing`
- `data_type`: `:phase` (default) or `:freq` (fractional-frequency samples)

# Returns
`DeviationResult` with fields `tau`, `deviation`, `edf`, `ci`, `alpha`, `neff`.
CI is filled by the engine (chi-squared where EDF is finite, Gaussian fallback).
"""
function ldev(
    x         :: AbstractVector{<:Real},
    tau0      :: Real;
    m_list    :: Union{Nothing, AbstractVector{<:Integer}} = nothing,
    data_type :: Symbol = :phase,
)
    mr = mhdev(x, tau0; m_list, data_type)
    return _scale_result(mr, mr.tau ./ LDEV_MHDEV_PREFACTOR, "ldev")
end
