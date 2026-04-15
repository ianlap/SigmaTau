# deviations/allan.jl — Allan deviation (ADEV, MDEV, TDEV)

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
CI is filled by the engine (chi-squared where EDF is finite, Gaussian fallback).
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
        m -> m,         # F_fn: m for unmodified
        0,              # dmin for noise_id
        2,              # dmax for noise_id
    )
    return engine(x, tau0, m_list, _adev_kernel, params; data_type)
end

# Kernel: returns (variance, neff)  — engine takes sqrt internally
# SP1065 Eq. 14: AVAR(τ) = mean(d²) / (2m²τ₀²)
function _adev_kernel(x::AbstractVector{<:Real}, m::Int, tau0::Real, x_cs::AbstractVector{<:Real})
    N = length(x)
    L = N - 2m
    L <= 0 && return (NaN, 0)
    d2 = @view(x[1+2m:end]) .- 2 .* @view(x[1+m:end-m]) .+ @view(x[1:L])
    v  = sum(abs2, d2) / (L * 2.0 * Float64(m)^2 * tau0^2)    # SP1065 Eq. 14
    return (v, L)
end

# ── MDEV ──────────────────────────────────────────────────────────────────────

"""
    mdev(x, tau0; m_list=nothing, data_type=:phase) → DeviationResult

Modified Allan deviation (MDEV). Uses overlapping second-difference sums,
SP1065 §4 (Modified AVAR).

The kernel uses cumsum prefix sums (O(N) per m):
    MVAR(τ) = Σ(s3 - 2s2 + s1)² / (Ne · 2m² · τ) / τ²   # SP1065 Eq. 15

# Arguments
- `x`:         phase data vector (seconds), length N ≥ 4
- `tau0`:      sampling interval (seconds)
- `m_list`:    averaging factors; auto-generated (octave-spaced) when `nothing`
- `data_type`: `:phase` (default) or `:freq`

# Returns
`DeviationResult` with fields `tau`, `deviation`, `edf`, `ci`, `alpha`, `neff`.
CI is filled by the engine (chi-squared where EDF is finite, Gaussian fallback).
"""
function mdev(
    x         :: AbstractVector{<:Real},
    tau0      :: Real;
    m_list    :: Union{Nothing, AbstractVector{<:Integer}} = nothing,
    data_type :: Symbol = :phase,
)
    params = DevParams(
        "mdev",         # name
        3,              # min_factor: N/m ≥ 3
        2,              # d: second-difference order
        m -> 1,         # F_fn: F = 1 for modified
        0,              # dmin for noise_id
        2,              # dmax for noise_id
    )
    return engine(x, tau0, m_list, _mdev_kernel, params; data_type)
end

# Kernel: cumsum prefix-sum approach, O(N) per m
# SP1065 Eq. 15: MVAR(τ) = Σ(s3 - 2s2 + s1)² / (Ne · 2m² · τ₀²)
function _mdev_kernel(x::AbstractVector{<:Real}, m::Int, tau0::Real, x_cs::AbstractVector{<:Real})
    N  = length(x)
    Ne = N - 3m + 1
    Ne <= 0 && return (NaN, 0)
    # Identity: s3 - 2s2 + s1 = x_cs[i+3m] - 3x_cs[i+2m] + 3x_cs[i+m] - x_cs[i]
    d  = @view(x_cs[1+3m:Ne+3m]) .- 3 .* @view(x_cs[1+2m:Ne+2m]) .+
         3 .* @view(x_cs[1+m:Ne+m]) .- @view(x_cs[1:Ne])
    # Divide by 2*m^4*tau0^2 because d is Σx_i (unscaled block-sum difference).
    # Float64 promotion on the denominator prevents Int64 overflow for large m·Ne
    # (e.g. Ne·2·m^4 wraps silently at m≳1e4 with Int64 arithmetic).
    v  = sum(abs2, d) / (Ne * 2.0 * Float64(m)^4 * tau0^2)
    return (v, Ne)
end

# ── TDEV ──────────────────────────────────────────────────────────────────────

"""
    tdev(x, tau0; m_list=nothing, data_type=:phase) → DeviationResult

Time deviation (TDEV). Wraps `mdev` and scales:
    TDEV(τ) = τ · MDEV(τ) / √3   # SP1065 §4

Does NOT call engine directly; derives from MDEV for consistency.

# Arguments
- `x`:         phase data vector (seconds), length N ≥ 4
- `tau0`:      sampling interval (seconds)
- `m_list`:    averaging factors; auto-generated (octave-spaced) when `nothing`
- `data_type`: `:phase` (default) or `:freq`

# Returns
`DeviationResult` with fields `tau`, `deviation`, `edf`, `ci`, `alpha`, `neff`.
CI is filled by the engine (chi-squared where EDF is finite, Gaussian fallback).
"""
function tdev(
    x         :: AbstractVector{<:Real},
    tau0      :: Real;
    m_list    :: Union{Nothing, AbstractVector{<:Integer}} = nothing,
    data_type :: Symbol = :phase,
)
    mr = mdev(x, tau0; m_list, data_type)
    return _scale_result(mr, mr.tau ./ TDEV_MDEV_PREFACTOR, "tdev")
end
