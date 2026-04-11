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
