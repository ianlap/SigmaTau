# stats.jl — EDF, bias correction, and confidence intervals
# Ported from legacy StabLab.jl confidence.jl
# References:
#   SP1065 Appendix A (EDF)
#   Greenhall & Riley, "Uncertainty of Stability Variances," PTTI 2003

# ── EDF dispatch ─────────────────────────────────────────────────────────────

"""
    edf_for_result(result) → Vector{Float64}

Compute equivalent degrees of freedom for every τ point in a `DeviationResult`.
Returns a vector of EDF values (NaN where parameters are invalid).
"""
function edf_for_result(result::DeviationResult)
    L   = length(result.tau)
    T   = (result.N - 1) * result.tau0   # record duration (s)
    edf = Vector{Float64}(undef, L)

    for k in 1:L
        m = round(Int, result.tau[k] / result.tau0)
        edf[k] = _edf_dispatch(result.method, result.alpha[k],
                               m, result.tau[k], result.tau0, result.N, T)
    end
    return edf
end

function _edf_dispatch(method::String, alpha::Int, m::Int,
                       tau::Real, tau0::Real, N::Int, T::Real)
    if method == "adev"
        return calculate_edf(alpha, 2, m, m, 1, N)        # SP1065: d=2, F=m, S=1
    elseif method == "mdev"
        return calculate_edf(alpha, 2, m, 1, 1, N)        # d=2, F=1, S=1
    elseif method == "hdev"
        return calculate_edf(alpha, 3, m, m, 1, N)        # d=3, F=m, S=1
    elseif method == "mhdev"
        return calculate_edf(alpha, 3, m, 1, 1, N)        # d=3, F=1, S=1
    elseif method == "totdev"
        return totaldev_edf("totvar", alpha, T, tau)
    elseif method == "mtotdev"
        return totaldev_edf("mtot",   alpha, T, tau)
    elseif method == "htotdev"
        return totaldev_edf("htot",   alpha, T, tau)
    elseif method == "mhtotdev"
        return totaldev_edf("mhtot",  alpha, T, tau)
    elseif method in ("tdev", "ldev")
        base = method == "tdev" ? "mdev" : "mhdev"
        return _edf_dispatch(base, alpha, m, tau, tau0, N, T)
    else
        @warn "edf_for_result: unknown method \"$method\", returning NaN"
        return NaN
    end
end

# ── Core EDF calculation (SP1065 Appendix A) ──────────────────────────────────

"""
    calculate_edf(alpha, d, m, F, S, N) → Float64

Equivalent degrees of freedom for an overlapping variance estimator.

# Arguments
- `alpha`: Power-law frequency noise exponent (-4 to 2)
- `d`: Phase difference order (2 = Allan, 3 = Hadamard)
- `m`: Averaging factor τ/τ₀
- `F`: Filter factor (m for unmodified, 1 for modified)
- `S`: Stride (1 non-overlapping, m overlapping)
- `N`: Number of phase data points
"""
function calculate_edf(alpha::Int, d::Int, m::Int, F::Int, S::Int, N::Int)
    # Convergent boundary: 2d + alpha > 1 (Greenhall & Riley 2003, Eq. 4)
    alpha + 2d <= 1 && return NaN

    L = m/F + m*d                    # filter length
    N < L && return NaN

    M = 1 + floor(Int, S * (N - L) / m)   # number of summands
    J = min(M, (d + 1) * S)               # truncation parameter (Greenhall 2003)

    sz0        = Float64(_compute_sz(0.0, F, alpha, d))   # 0.0 ensures Float64 throughout
    basic_sum  = _compute_basic_sum(J, M, S, F, alpha, d, sz0)

    basic_sum > 0 || return NaN
    return M * sz0^2 / basic_sum
end

# ── Total deviation EDF (SP1065 §§5.2.11–5.2.13) ─────────────────────────────

"""
    totaldev_edf(var_type, alpha, T, tau) → Float64

EDF for total deviation variants using lookup-table coefficients.
"""
function totaldev_edf(var_type::String, alpha::Int, T::Real, tau::Real)
    if var_type == "totvar"
        b, c = _coeff_totvar(alpha)
        return b * (T / tau) - c
    elseif var_type == "mtot"
        b, c = _coeff_mtot(alpha)
        return b * (T / tau) - c
    elseif var_type == "htot"
        b0, b1 = _coeff_htot(alpha)
        return (T / tau) / (b0 + b1 * (tau / T))
    elseif var_type == "mhtot"
        b, c = _coeff_mhtot(alpha)
        return b * (T / tau) - c
    else
        return NaN
    end
end

# ── EDF helper functions ──────────────────────────────────────────────────────

function _compute_sw(t::Real, alpha::Int)
    ta = abs(t)
    if alpha == 2;  return -ta
    elseif alpha == 1;  return t^2 * log(max(ta, eps()))
    elseif alpha == 0;  return ta^3
    elseif alpha == -1; return -t^4 * log(max(ta, eps()))
    elseif alpha == -2; return -ta^5
    elseif alpha == -3; return  t^6 * log(max(ta, eps()))
    elseif alpha == -4; return  ta^7
    else; return NaN
    end
end

function _compute_sx(t::Real, F::Int, alpha::Int)
    if F > 100 && alpha <= 0   # large-F approximation
        return _compute_sw(t, alpha + 2)
    end
    return F^2 * (2 * _compute_sw(t, alpha) -
                  _compute_sw(t - 1/F, alpha) -
                  _compute_sw(t + 1/F, alpha))
end

# Sz = squared d-th central difference of Sx — Greenhall & Riley (2003) Eq. 8.
# Weights are rows of the squared Pascal triangle:
#   d=1 → (2, -1, -1); d=2 → (6, -4, -4, 1, 1); d=3 → (20, -15, -15, 6, 6, -1, -1)
function _compute_sz(t::Real, F::Int, alpha::Int, d::Int)
    sx = (u) -> _compute_sx(u, F, alpha)
    if d == 1
        return 2sx(t) - sx(t-1) - sx(t+1)
    elseif d == 2
        return 6sx(t) - 4sx(t-1) - 4sx(t+1) + sx(t-2) + sx(t+2)
    elseif d == 3
        return 20sx(t) - 15sx(t-1) - 15sx(t+1) +
                6sx(t-2) +  6sx(t+2) - sx(t-3) - sx(t+3)
    else
        return NaN
    end
end

function _compute_basic_sum(J::Int, M::Int, S::Int, F::Int, alpha::Int, d::Int, sz0::Float64)
    bsum = sz0^2
    for j in 1:(J-1)
        szj   = _compute_sz(j/S, F, alpha, d)
        bsum += 2 * (1 - j/M) * szj^2
    end
    if J <= M
        szJ   = _compute_sz(J/S, F, alpha, d)
        bsum += (1 - J/M) * szJ^2
    end
    return bsum
end

# ── Coefficient tables ────────────────────────────────────────────────────────

# SP1065 Table 9
function _coeff_totvar(alpha::Int)
    alpha == 0  && return (1.50, 0.00)
    alpha == -1 && return (1.17, 0.22)
    alpha == -2 && return (0.93, 0.36)
    return (NaN, NaN)
end

# SP1065 Table 10
function _coeff_mtot(alpha::Int)
    alpha == 2  && return (1.90, 2.10)
    alpha == 1  && return (1.20, 1.40)
    alpha == 0  && return (1.10, 1.20)
    alpha == -1 && return (0.85, 0.50)
    alpha == -2 && return (0.75, 0.31)
    return (NaN, NaN)
end

# FCS 2001 coefficients (no published model for mhtotdev; approximate)
function _coeff_mhtot(alpha::Int)
    alpha == 2  && return (3.904,  9.640)
    alpha == 1  && return (2.656, 11.093)
    alpha == 0  && return (2.275,  8.701)
    alpha == -1 && return (1.964,  4.908)
    alpha == -2 && return (1.572,  4.534)
    return (NaN, NaN)
end

# Greenhall (2003) Table 1 — htot
function _coeff_htot(alpha::Int)
    alpha == 0  && return (0.546, 1.41)
    alpha == -1 && return (0.667, 2.00)
    alpha == -2 && return (0.909, 1.00)
    return (NaN, NaN)
end

# ── Gaussian fallback Kn factors ─────────────────────────────────────────────

function _kn_from_alpha(alpha::Int)
    alpha == -2 && return 0.75
    alpha == -1 && return 0.77
    alpha == 0  && return 0.87
    alpha == 1  && return 0.99
    alpha == 2  && return 0.99
    return 1.10   # conservative fallback
end

# ── Bias correction (SP1065 §§5.2.11–5.2.13) ────────────────────────────────

"""
    bias_correction(alpha, var_type, tau, T) → Vector{Float64}

Bias factor B(α) for TOTVAR, MTOT, and HTOT.
Divide raw deviation by B to get unbiased estimate.
"""
function bias_correction(alpha::Union{Int,Vector{Int}},
                         var_type::String,
                         tau::Union{Real,Vector{<:Real}},
                         T::Real)
    alpha_v = isa(alpha, Int) ? [alpha] : alpha
    tau_v   = isa(tau,   Real) ? [tau]  : collect(Float64, tau)

    # Broadcast to equal length
    if length(alpha_v) == 1 && length(tau_v) > 1
        alpha_v = fill(alpha_v[1], length(tau_v))
    elseif length(tau_v) == 1 && length(alpha_v) > 1
        tau_v = fill(tau_v[1], length(alpha_v))
    end

    L = length(alpha_v)
    B = ones(Float64, L)
    vt = lowercase(var_type)

    if vt == "totvar"
        for k in 1:L
            a = alpha_v[k] == -1 ? 1 / (3 * log(2)) :   # Flicker FM
                alpha_v[k] == -2 ? 0.75 : 0.0            # RWFM / no correction
            B[k] = 1 - a * (tau_v[k] / T)
        end

    elseif vt == "mtot"
        # SP1065 Table 11
        table = Dict(2=>1.06, 1=>1.17, 0=>1.27, -1=>1.30, -2=>1.31)
        for k in 1:L
            a = clamp(alpha_v[k], -2, 2)
            if a != alpha_v[k]
                @warn "bias_correction: alpha=$(alpha_v[k]) out of MTOT bounds [-2..2]. Clamping."
            end
            B[k] = get(table, a, 1.0)
        end

    elseif vt == "htot"
        # FCS 2001 Table 1: correction a(α), B = 1/(1+a)
        table = Dict(0=>-0.005, -1=>-0.149, -2=>-0.229, -3=>-0.283, -4=>-0.321)
        for k in 1:L
            a = clamp(alpha_v[k], -4, 0)
            if a != alpha_v[k]
                @warn "bias_correction: alpha=$(alpha_v[k]) out of HTOT bounds [-4..0]. Clamping."
            end
            B[k] = 1 / (1 + get(table, a, 0.0))
        end

    else
        throw(ArgumentError("var_type must be \"totvar\", \"mtot\", or \"htot\", got \"$var_type\""))
    end

    return B
end

# ── Confidence intervals ──────────────────────────────────────────────────────

"""
    compute_ci(result; confidence=result.confidence) → DeviationResult

Compute EDF and confidence intervals. Chi-squared CI where EDF is finite
and positive; Gaussian ±Kn·σ·z/√N fallback otherwise (SP1065 Eq. A-8).
"""
function compute_ci(result::DeviationResult; confidence::Real = result.confidence)
    edf = edf_for_result(result)
    L   = length(result.deviation)
    ci  = Matrix{Float64}(undef, L, 2)

    a_half = (1 - confidence) / 2
    z      = norminvcdf(1 - a_half)

    for k in 1:L
        d = result.deviation[k]
        if isnan(d)
            ci[k, 1] = NaN
            ci[k, 2] = NaN
            continue
        end
        ef = edf[k]
        if isfinite(ef) && ef >= 1.0
            # Chi-squared CI: dev·√(edf/χ²_{1-a/2}) to dev·√(edf/χ²_{a/2})
            chi_lo = chisqinvcdf(ef, a_half)
            chi_hi = chisqinvcdf(ef, 1 - a_half)
            ci[k, 1] = d * sqrt(ef / chi_hi)
            ci[k, 2] = d * sqrt(ef / chi_lo)
        else
            # Gaussian fallback for ef < 1 or ef is NaN (SP1065 §A.3)
            Kn   = _kn_from_alpha(result.alpha[k])
            half = Kn * d * z / sqrt(Float64(result.N))
            ci[k, 1] = d - half
            ci[k, 2] = d + half
        end
    end

    return DeviationResult(
        result.tau, result.deviation, edf, ci,
        result.alpha, result.neff,
        result.tau0, result.N, result.method, Float64(confidence)
    )
end
