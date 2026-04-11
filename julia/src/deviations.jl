# deviations.jl — Thin deviation wrappers over the shared engine
# Each function defines a kernel + DevParams and delegates to engine().
# Architecture: deviation-engine skill / CLAUDE.md §Architecture

# ── ADEV ──────────────────────────────────────────────────────────────────────

"""
    adev(x, tau0; m_list=nothing) → DeviationResult

Overlapping Allan deviation (OADEV). Second-difference phase variance, SP1065 §3.

The kernel computes the overlapping estimator:
    AVAR(τ) = ⟨(x[i+2m] - 2x[i+m] + x[i])²⟩ / (2τ²)   # SP1065 Eq. 14

# Arguments
- `x`:     phase data vector (seconds), length N ≥ 3
- `tau0`:  sampling interval (seconds)
- `m_list`: averaging factors; auto-generated (octave-spaced) when `nothing`

# Returns
`DeviationResult` with fields `tau`, `deviation`, `edf`, `ci`, `alpha`, `neff`.
CI is NaN until `compute_ci` is called.
"""
function adev(
    x         :: AbstractVector{<:Real},
    tau0      :: Real;
    m_list    :: Union{Nothing, AbstractVector{<:Integer}} = nothing,
    data_type :: Symbol = :phase,
)
    params = DevParams(
        "adev",         # name
        2,              # min_factor: N/m ≥ 2
        2,              # d: second-difference order (Allan)
        m -> m,         # F_fn: F = m for unmodified
        0,              # dmin for noise_id
        2,              # dmax for noise_id
        false,          # is_total
        "",             # total_type (unused)
        false,          # needs_bias
        "",             # bias_type (unused)
    )
    return engine(x, tau0, m_list, _adev_kernel, params; data_type)
end

# Kernel: returns (variance, neff)  — engine takes sqrt internally
# SP1065 Eq. 14: AVAR(τ) = mean(d²) / (2m²τ₀²)
function _adev_kernel(x::AbstractVector{<:Real}, m::Int, tau0::Real)
    N = length(x)
    L = N - 2m
    L <= 0 && return (NaN, 0)
    d2 = @view(x[1+2m:end]) .- 2 .* @view(x[1+m:end-m]) .+ @view(x[1:L])
    v  = sum(abs2, d2) / (L * 2 * m^2 * tau0^2)    # SP1065 Eq. 14
    return (v, L)
end

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
CI is NaN until `compute_ci` is called.
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
        false,          # is_total
        "",             # total_type (unused)
        false,          # needs_bias
        "",             # bias_type (unused)
    )
    return engine(x, tau0, m_list, _hdev_kernel, params; data_type)
end

# Kernel: returns (variance, neff) — engine takes sqrt internally
# SP1065 HVAR: HVAR(τ) = mean(d3²) / (6m²τ₀²)
function _hdev_kernel(x::AbstractVector{<:Real}, m::Int, tau0::Real)
    N = length(x)
    L = N - 3m
    L <= 0 && return (NaN, 0)
    d3 = @view(x[1+3m:end]) .- 3 .* @view(x[1+2m:end-m]) .+
         3 .* @view(x[1+m:end-2m]) .- @view(x[1:L])
    v = sum(abs2, d3) / (L * 6 * m^2 * tau0^2)   # SP1065 HVAR
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
CI is NaN until `compute_ci` is called.
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
        false,          # is_total
        "",             # total_type (unused)
        false,          # needs_bias
        "",             # bias_type (unused)
    )
    return engine(x, tau0, m_list, _mhdev_kernel, params; data_type)
end

# Kernel: returns (variance, neff) — engine takes sqrt internally
# Third differences + moving average via cumsum trick
function _mhdev_kernel(x::AbstractVector{<:Real}, m::Int, tau0::Real)
    N = length(x)
    Ne = N - 4m + 1
    Ne <= 0 && return (NaN, 0)
    # Third differences of the phase data
    d4 = @view(x[1:Ne]) .- 3 .* @view(x[1+m:Ne+m]) .+
         3 .* @view(x[1+2m:Ne+2m]) .- @view(x[1+3m:Ne+3m])
    # Moving average via cumsum (length-m sums over d4)
    S = cumsum([zero(eltype(x)); d4])  # length Ne+1
    avg = @view(S[m+1:end]) .- @view(S[1:end-m])  # length Ne+1-m
    # Variance: meansq(avg) / (6 * m^4 * tau0^2)
    v = sum(abs2, avg) / (length(avg) * 6 * m^4 * tau0^2)
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
CI is NaN until `compute_ci` is called.
"""
function ldev(
    x         :: AbstractVector{<:Real},
    tau0      :: Real;
    m_list    :: Union{Nothing, AbstractVector{<:Integer}} = nothing,
    data_type :: Symbol = :phase,
)
    mr = mhdev(x, tau0; m_list, data_type)
    scale = mr.tau ./ sqrt(10 / 3)
    ci_scaled = mr.ci .* reshape(scale, :, 1)   # (L,2) .* (L,1) broadcast
    return DeviationResult(
        mr.tau,
        scale .* mr.deviation,
        mr.edf,
        ci_scaled,
        mr.alpha,
        mr.neff,
        mr.tau0,
        mr.N,
        "ldev",
        mr.confidence,
    )
end
