# Core utilities for StabLab

function validate_phase_data(x::AbstractVector{T}) where T<:Real
    all(isfinite, x) || throw(ArgumentError("Phase data must be finite"))
    return vec(x)
end

function validate_tau0(tau0::Real)
    (tau0 > 0 && isfinite(tau0)) || throw(ArgumentError("tau0 must be positive and finite"))
    return tau0
end

"""
    _default_mlist(N, min_factor)

Octave-spaced averaging factors ensuring `min_factor * m <= N`.
"""
_default_mlist(N::Int, min_factor::Int) =
    [2^k for k in 0:floor(Int, log2(N / min_factor))]

# Kept for any external callers
default_m_list(N::Int) = _default_mlist(N, 2)

"""
    _make_result(tau, dev, alpha, neff, tau0, N, method, confidence)

Build a `DeviationResult` with the given computed fields.  EDF and CI are left
as NaN placeholders (filled on demand by `compute_ci`).
"""
function _make_result(tau, dev, alpha, neff, tau0, N::Int, method::String, confidence)
    L = length(dev)
    alpha_int = Vector{Int}(undef, L)
    for i in 1:L
        alpha_int[i] = isnan(alpha[i]) ? 0 : round(Int, alpha[i])
    end
    DeviationResult(Float64.(tau), Float64.(dev), fill(NaN, L), fill(NaN, L, 2),
                    alpha_int, Vector{Int}(neff), Float64(tau0), N, method, Float64(confidence))
end

"""
    detrend_linear(x)

Remove linear trend from data using least-squares fit.
"""
function detrend_linear(x::AbstractVector{T}) where T<:Real
    n = length(x)
    n < 2 && return copy(x)
    A = [ones(T, n) T.(1:n)]
    return x - A * (A \ x)
end

# ── DeviationResult tuple unpacking ──────────────────────────────────────────

"""
    unpack_result(r::DeviationResult, ::Val{N})

`Val(2)` → `(tau, deviation)`, `3` adds `edf`, `4` adds `ci`, `5` adds `alpha`.
"""
unpack_result(r::DeviationResult, ::Val{2}) = (r.tau, r.deviation)
unpack_result(r::DeviationResult, ::Val{3}) = (r.tau, r.deviation, r.edf)
unpack_result(r::DeviationResult, ::Val{4}) = (r.tau, r.deviation, r.edf, r.ci)
unpack_result(r::DeviationResult, ::Val{5}) = (r.tau, r.deviation, r.edf, r.ci, r.alpha)

function _tuple_return_check(N::Int, fname::String)
    2 <= N <= 5 || throw(ArgumentError("$fname(..., Val(N)) requires N ∈ 2:5, got $N"))
end

# ── mean-of-squares without temporary allocation ────────────────────────────

@inline _meansq(v) = sum(abs2, v) / length(v)
