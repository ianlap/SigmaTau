# types.jl — Core data structures for SigmaTau
# SP1065: Riley & Howe, "Handbook of Frequency Stability Analysis"

"""
    DeviationResult{T<:Real}

Result of a deviation computation.

# Fields
- `tau::Vector{T}`: Averaging times τ = m·τ₀ (s)
- `deviation::Vector{T}`: Deviation values (sqrt of variance)
- `edf::Vector{T}`: Equivalent degrees of freedom (NaN until compute_ci called)
- `ci::Matrix{T}`: Confidence intervals, size (L,2): [lower upper] per row
- `alpha::Vector{Int}`: Power-law noise exponent α per averaging time
- `neff::Vector{Int}`: Effective sample counts per averaging time
- `tau0::T`: Sampling interval (s)
- `N::Int`: Original phase data length
- `method::String`: Deviation identifier (e.g. "adev")
- `confidence::T`: Confidence level (default 0.683 = 1σ)
"""
struct DeviationResult{T<:Real}
    tau::Vector{T}
    deviation::Vector{T}
    edf::Vector{T}
    ci::Matrix{T}
    alpha::Vector{Int}
    neff::Vector{Int}
    tau0::T
    N::Int
    method::String
    confidence::T
end

"""
    DevParams

Configuration struct passed to the shared deviation engine.
Each deviation wrapper creates one of these; the engine is parameterised by it.

# Fields
- `name::String`: Deviation identifier ("adev", "mdev", …)
- `min_factor::Int`: Minimum N/m ratio for default m_list generation (2, 3, or 4)
- `d::Int`: Phase difference order (2 = Allan/modified, 3 = Hadamard)
- `F_fn::Function`: m → F filter factor (identity `m` for unmodified, `1` for modified)
- `dmin::Int`: Minimum differencing depth for noise_id
- `dmax::Int`: Maximum differencing depth for noise_id
- Total-deviation EDF and bias behavior are inferred from `name` in the engine.

# Deviation quick reference (from SP1065)
| dev      | d | min_factor | F_fn |
|----------|---|------------|------|
| adev     | 2 | 2          | m->m |
| mdev     | 2 | 3          | m->1 |
| hdev     | 3 | 4          | m->m |
| mhdev    | 3 | 4          | m->1 |
| totdev   | 2 | 2          | m->m |
| mtotdev  | 2 | 3          | m->1 |
| htotdev  | 3 | 3          | m->m |
| mhtotdev | 3 | 4          | m->1 |
"""
struct DevParams
    name::String
    min_factor::Int
    d::Int
    F_fn::Function
    dmin::Int
    dmax::Int
end

# ── Private utilities ─────────────────────────────────────────────────────────

"""
    _default_mlist(N, min_factor)

Octave-spaced averaging factors 1,2,4,… ensuring `min_factor * m ≤ N`.
"""
_default_mlist(N::Int, min_factor::Int) =
    [2^k for k in 0:floor(Int, log2(N / min_factor))]

"""
    detrend_linear!(x)

Remove linear trend from data in-place via least-squares fit.
Uses a numerically stable two-pass algorithm.
"""
function detrend_linear!(x::AbstractVector{T}) where T<:Real
    n = length(x)
    n < 2 && return x
    x_bar = (n + 1) / 2.0
    sum_num = 0.0
    sum_y = 0.0
    for i in 1:n
        val = Float64(x[i])
        sum_num += (i - x_bar) * val
        sum_y += val
    end
    ss_xx = n * (Float64(n)^2 - 1) / 12.0
    slope = sum_num / ss_xx
    y_bar = sum_y / n
    for i in 1:n
        x[i] -= T(y_bar + slope * (i - x_bar))
    end
    return x
end

"""
    detrend_linear(x)

Remove linear trend from data via least-squares fit.
"""
function detrend_linear(x::AbstractVector{T}) where T<:Real
    return detrend_linear!(copy(x))
end

"""
    _meansq(v)

Mean of squared elements: Σv²/N. Avoids a separate allocation.
"""
@inline _meansq(v) = sum(abs2, v) / length(v)

# ── Result unpacking ──────────────────────────────────────────────────────────

"""
    unpack_result(r, Val(N))

Unpack a `DeviationResult` into a tuple of N fields:
- `Val(2)` → `(tau, deviation)`
- `Val(3)` → adds `edf`
- `Val(4)` → adds `ci`
- `Val(5)` → adds `alpha`
"""
unpack_result(r::DeviationResult, ::Val{2}) = (r.tau, r.deviation)
unpack_result(r::DeviationResult, ::Val{3}) = (r.tau, r.deviation, r.edf)
unpack_result(r::DeviationResult, ::Val{4}) = (r.tau, r.deviation, r.edf, r.ci)
unpack_result(r::DeviationResult, ::Val{5}) = (r.tau, r.deviation, r.edf, r.ci, r.alpha)
