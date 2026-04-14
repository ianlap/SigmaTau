# noise.jl — Noise identification for frequency stability analysis
# Ported from legacy StabLab.jl noise.jl
# References: NIST SP1065 Section 5.6; Riley & Howe

"""
    noise_id(x, m_list, data_type="phase", dmin=0, dmax=2) → Vector{Float64}

Dominant power-law noise estimator.  Returns a vector of estimated α exponents
(one per element of `m_list`), or NaN where every method fails.

Dispatch (SP1065 §5.6 / Stable32 manual):
  1. `N_eff ≥ NEFF_RELIABLE` → lag-1 ACF (primary)
  2. `N_eff <  NEFF_RELIABLE` → B1-ratio / R(n) (reliable once m² scaled)
  3. Both above return NaN   → carry forward the most recent reliable α

Carry-forward matches Stable32's "use the previous noise type estimate at the
longest averaging time" rule — the last-resort when neither ACF nor B1/R(n)
can produce an estimate for this τ.
"""
# NEFF_RELIABLE: minimum N_eff for lag-1 ACF to be numerically trustworthy.
# SP1065 §5.6 cites 30 as the theoretical minimum, but empirically the ACF
# estimator is still high-variance just above that boundary. 50 is a
# conservative working threshold that stabilises the long-τ tail.
const NEFF_RELIABLE = 50

function noise_id(x::Vector{Float64}, m_list::Vector{Int},
                  data_type::String = "phase",
                  dmin::Int = 0, dmax::Int = 2)
    x_clean = _preprocess(x)
    N = length(x_clean)
    alpha_list = fill(NaN, length(m_list))
    last_reliable = NaN

    for (k, m) in enumerate(m_list)
        N_eff = N ÷ m
        alpha = NaN
        try
            if N_eff >= NEFF_RELIABLE
                alpha, = _noise_id_lag1acf(x_clean, m, data_type, dmin, dmax)
            else
                alpha, = _noise_id_b1rn(x_clean, m, data_type)
            end
        catch err
            @warn "noise_id: estimation failed for m=$m — $(err)"
        end

        if !isnan(alpha)
            alpha_list[k]  = Float64(alpha)
            last_reliable  = alpha_list[k]
        elseif !isnan(last_reliable)
            alpha_list[k]  = last_reliable   # last-resort: carry forward
        end
    end

    return alpha_list
end

# ── Preprocessing ─────────────────────────────────────────────────────────────

function _preprocess(x::Vector{Float64})
    x_mean = mean(x)
    x_std  = std(x)
    # Guard against constant data
    if x_std < eps()
        return detrend_linear(x)
    end
    z = abs.((x .- x_mean) ./ x_std)
    return detrend_linear(x[z .< 5.0])
end

# ── Lag-1 ACF method ──────────────────────────────────────────────────────────

"""
    _noise_id_lag1acf(x, m, data_type, dmin, dmax)

Lag-1 autocorrelation function method for noise identification (SP1065 §5.6).
Returns `(alpha, alpha_int, d, rho)`.
"""
function _noise_id_lag1acf(x::Vector{Float64}, m::Int, data_type::String,
                            dmin::Int = 0, dmax::Int = 2)
    if lowercase(data_type) == "phase"
        x = m > 1 ? x[1:m:end] : x
        x = detrend_quadratic(x)
    elseif lowercase(data_type) == "freq"
        N = (length(x) ÷ m) * m
        x = vec(mean(reshape(x[1:N], m, :), dims=1))
        x = detrend_linear(x)
    else
        throw(ArgumentError("data_type must be \"phase\" or \"freq\""))
    end

    d = 0
    while true
        r1  = _lag1_acf(x)
        rho = r1 / (1 + r1)

        if d >= dmin && (rho < 0.25 || d >= dmax)
            p       = -2 * (rho + d)
            alpha   = p + 2 * (lowercase(data_type) == "phase" ? 1 : 0)
            return (alpha, round(Int, alpha), d, rho)
        end

        x = diff(x)
        d += 1
        length(x) >= 5 || throw(ArgumentError("Data too short after differencing"))
    end
end

function _lag1_acf(x::Vector{Float64})
    xm   = x .- mean(x)
    ssx  = sum(abs2, xm)
    # Guard for constant input: after mean-subtraction the detrend residuals
    # are O(ε)·N rather than exact zeros, so use a tolerance rather than ==0.
    ssx < eps(Float64) * length(x) && return NaN
    return sum(@view(xm[1:end-1]) .* @view(xm[2:end])) / ssx
end

# ── B1-ratio / R(n) fallback ──────────────────────────────────────────────────

"""
    _noise_id_b1rn(x, m, data_type)

B1-ratio and R(n) fallback method for small N_eff (SP1065 §5.6).
Returns `(alpha_int, mu_best, B1_obs)`.
"""
function _noise_id_b1rn(x::Vector{Float64}, m::Int, data_type::String)
    if lowercase(data_type) == "phase"
        x_dec = x[1:m:end]
        x_dec = detrend_quadratic(x_dec)

        # AVAR at τ = m·τ₀.  Computed from decimated phase so the detrend above
        # (which stabilises long-drift records) carries through; the m² factor
        # corrects `_simple_avar(..., 1)` to SP1065 Eq. 14's m²·τ₀² denominator.
        avar_val = _simple_avar(x_dec, 1) / Float64(m)^2
        N_avar   = length(x_dec) - 2

        dx = diff(x)
        Nd = (length(dx) ÷ m) * m
        Nd < m && return (NaN, -2, NaN)
        y_blocks   = reshape(dx[1:Nd], m, :)
        y_avg      = vec(mean(y_blocks, dims=1))
        var_class  = var(y_avg; corrected=false)

    elseif lowercase(data_type) == "freq"
        N = (length(x) ÷ m) * m
        N < 2m && return (NaN, -2, NaN)
        y_avg     = vec(mean(reshape(x[1:N], m, :), dims=1))
        y_avg     = detrend_linear(y_avg)
        dy        = diff(y_avg)
        var_class = var(y_avg; corrected=false)
        avar_val  = sum(abs2, dy) / (2 * (length(y_avg) - 1))
        N_avar    = length(y_avg)
    else
        throw(ArgumentError("data_type must be \"phase\" or \"freq\""))
    end

    (isnan(avar_val) || avar_val <= 0) && return (NaN, -2, NaN)

    B1_obs = var_class / avar_val

    mu_list    = [1, 0, -1, -2]
    alpha_list = [-2, -1, 0, 2]
    b1_vals    = [_b1_theory(N_avar, mu) for mu in mu_list]

    mu_best   = mu_list[end]
    alpha_int = alpha_list[end]

    for i in 1:(length(mu_list) - 1)
        boundary = sqrt(b1_vals[i] * b1_vals[i+1])
        if B1_obs > boundary
            mu_best   = mu_list[i]
            alpha_int = alpha_list[i]
            break
        end
    end

    # Refine α=2 vs α=1 using R(n) for White PM vs Flicker PM (phase data only)
    if mu_best == -2 && lowercase(data_type) == "phase"
        adev_val = sqrt(avar_val)
        mdev_val = _simple_mdev(x, m, 1.0)
        if !isnan(mdev_val) && adev_val > 0
            Rn_obs = (mdev_val / adev_val)^2
            R_hi = _rn_theory(m, 0)   # α=2 (White PM)
            R_lo = _rn_theory(m, -1)  # α=1 (Flicker PM)
            alpha_int = Rn_obs > sqrt(R_hi * R_lo) ? 1 : 2
        end
    end

    return (alpha_int, mu_best, B1_obs)
end

# ── B1 / R(n) theory tables ───────────────────────────────────────────────────

# Theoretical B1 ratio (classical-var / Allan-var) vs. noise slope μ
# where μ is defined by σ²_y(τ) ∝ τ^μ.
# Closed forms for integer μ; general formula (SP1065 Eq. 73 / Howe–Beard 1998) otherwise.
#   μ = +2 → FW FM (α=-3)        μ =  0 → FLFM  (α=-1)        μ = -2 → WHPM/FLPM (α=2,1)
#   μ = +1 → RWFM  (α=-2)        μ = -1 → WHFM  (α= 0)
function _b1_theory(N::Int, mu::Int)
    if mu == 2;  return N * (N + 1) / 6                            # FW FM
    elseif mu == 1;  return N / 2                                  # RWFM
    elseif mu == 0;  return N * log(N) / (2 * (N - 1) * log(2))    # FLFM
    elseif mu == -1; return 1.0                                    # WHFM (reference)
    elseif mu == -2; return (N^2 - 1) / (1.5 * N * (N - 1))        # WHPM/FLPM
    else;            return (N * (1 - N^mu)) / (2 * (N - 1) * (1 - 2^mu))  # SP1065 Eq. 73
    end
end

# Theoretical R(n) = MVAR/AVAR ratio vs. noise slope b (SP1065 §5.6 / Riley §5.2.6).
# Used to resolve WHPM (b=0) vs. FLPM (b=-1) ambiguity after the B1 ratio test.
function _rn_theory(af::Int, b::Int)
    if b == 0
        return 1.0 / af                                # WHPM asymptotic: R(n) → 1/m
    elseif b == -1
        # FLPM: closed-form ratio of MVAR/AVAR using leading-order expansions
        avar = (1.038 + 3 * log(2π * 0.5 * af)) / (4π^2)
        mvar = 3 * log(256 / 27) / (8π^2)
        return mvar / avar
    else
        return 1.0
    end
end

# ── Helpers used by B1 fallback ───────────────────────────────────────────────

"""
    _simple_avar(x, m)

Basic overlapping Allan variance at averaging factor `m` (no noise ID).
SP1065 Eq. 10.
"""
function _simple_avar(x::Vector{Float64}, m::Int)
    N = length(x)
    L = N - 2m
    L <= 0 && return NaN
    d2 = @view(x[1+2m:N]) .- 2 .* @view(x[1+m:N-m]) .+ @view(x[1:L])
    return _meansq(d2) / (2 * m^2)
end

"""
    _simple_mdev(x, m, tau0)

Basic modified Allan deviation (prefix-sum method) without noise ID.
"""
function _simple_mdev(x::Vector{Float64}, m::Int, tau0::Real)
    N  = length(x)
    Ne = N - 3m + 1
    Ne <= 0 && return NaN
    cs = cumsum(pushfirst!(copy(x), 0.0))
    s1 = @view(cs[1+m:Ne+m])   .- @view(cs[1:Ne])
    s2 = @view(cs[1+2m:Ne+2m]) .- @view(cs[1+m:Ne+m])
    s3 = @view(cs[1+3m:Ne+3m]) .- @view(cs[1+2m:Ne+2m])
    d  = (s3 .- 2 .* s2 .+ s1) ./ m
    return sqrt(_meansq(d) / (2 * m^2 * tau0^2))
end
