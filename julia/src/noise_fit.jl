# noise_fit.jl — MHDEV-to-q noise-component fit
# Non-interactive port of matlab/legacy/kflab/mhdev_fit.m:
# successive-subtraction, fixed-slope power-law fits over user-declared
# τ-index ranges. Each region operates on the residual σ² (or residual σ
# for flicker types) after earlier subtractions, matching the legacy flow.
#
# Model (σ²_MHDEV space, legacy constants):
#     σ²_MHDEV = (10/3)·q_wpm·τ^-3 + (7/16)·q_wfm·τ^-1
#              + (1/9)·q_rwfm·τ^+1 + (11/120)·q_rrfm·τ^+3
# Flicker types fit σ directly (not σ²):
#     FFM  σ = sig0·τ^0
#     FPM  σ = sig0·τ^-2
#
# References:
#   matlab/legacy/kflab/mhdev_fit.m    — interactive fit (this file's basis)
#   matlab/legacy/kflab/ci2weights.m   — CI→weight mapping
#   matlab/legacy/kflab/weightedMean.m — weighted mean with n_eff

"""
    MHDevFitRegion

Per-region fit diagnostics returned inside `MHDevFitResult.regions`.

- `noise_type` — `:wpm`, `:wfm`, `:rwfm`, `:rrfm`, `:ffm`, or `:fpm`
- `indices`    — τ-index range that was fit
- `value`      — fitted `q` (for power-law) or `sig0` (for flicker)
- `value_std`  — 1σ uncertainty from the weighted mean
- `skipped`    — true if the residual had no positive values in this range
"""
struct MHDevFitRegion
    noise_type :: Symbol
    indices    :: Vector{Int}
    value      :: Float64
    value_std  :: Float64
    skipped    :: Bool
end

"""
    MHDevFitResult

Output of `mhdev_fit`.

- `q_wpm`, `q_wfm`, `q_rwfm`, `q_rrfm` — power-law coefficients (sums over
  all regions of that type)
- `sig0_ffm`, `sig0_fpm` — flicker intercepts (σ at τ=1)
- `regions` — per-region diagnostics (see `MHDevFitRegion`)
- `var_residual` — residual σ² after all subtractions (length = length(tau))
"""
struct MHDevFitResult
    q_wpm        :: Float64
    q_wfm        :: Float64
    q_rwfm       :: Float64
    q_rrfm       :: Float64
    sig0_ffm     :: Float64
    sig0_fpm     :: Float64
    regions      :: Vector{MHDevFitRegion}
    var_residual :: Vector{Float64}
end

# Legacy coefficients and slopes from mhdev_fit.m
const _MHDEV_POWERLAW = (
    wpm  = (slope = -3, coeff = 10 / 3),
    wfm  = (slope = -1, coeff = 7 / 16),
    rwfm = (slope = +1, coeff = 1 / 9),
    rrfm = (slope = +3, coeff = 11 / 120),
)
const _MHDEV_FLICKER = (
    ffm = (slope =  0,),
    fpm = (slope = -2,),
)

_is_powerlaw(t::Symbol) = t in (:wpm, :wfm, :rwfm, :rrfm)
_is_flicker(t::Symbol)  = t in (:ffm, :fpm)

"""
    mhdev_fit(tau, sigma, regions; ci=nothing, weight_method=:equal)
        -> MHDevFitResult

Fit MHDEV noise components by successive residual subtraction.

# Arguments
- `tau::AbstractVector` — averaging times [s], length L
- `sigma::AbstractVector` — MHDEV values, length L
- `regions` — iterable of `(noise_type::Symbol, idx_range)` pairs, applied in
  order. `noise_type` must be one of `:wpm`, `:wfm`, `:rwfm`, `:rrfm`, `:ffm`,
  `:fpm`. `idx_range` is any integer range into `tau`.
- `ci::Union{Nothing, AbstractMatrix}` — optional L×2 confidence-interval
  matrix `[lower upper]`. When `nothing`, all points carry equal weight.
- `weight_method::Symbol` — how to convert CI → weights (used only when `ci`
  is given). Mirrors legacy `ci2weights.m`:
    * `:symmetric`    → w = 1 / (½·(hi−lo))²
    * `:conservative` → w = 1 / max(σ−lo, hi−σ)²  (MATLAB default)
    * `:inverse`      → w = 1 / (hi−lo)
- `:equal` (default when `ci===nothing`) — uniform weights

# Returns
`MHDevFitResult` with total q/sig0 coefficients, per-region diagnostics, and
the residual σ² after all subtractions.
"""
function mhdev_fit(tau::AbstractVector{<:Real},
                   sigma::AbstractVector{<:Real},
                   regions;
                   ci::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
                   weight_method::Symbol = :equal)
    length(tau) == length(sigma) ||
        throw(ArgumentError("mhdev_fit: tau and sigma must have matching length"))
    weights = if isnothing(ci)
        fill(1.0, length(sigma))
    else
        size(ci) == (length(sigma), 2) ||
            throw(ArgumentError("mhdev_fit: ci must be length(sigma)×2"))
        _ci_to_weights(Float64.(sigma), Float64.(@view(ci[:, 1])),
                                         Float64.(@view(ci[:, 2])),
                       weight_method === :equal ? :symmetric : weight_method)
    end

    q_total   = Dict(:wpm => 0.0, :wfm => 0.0, :rwfm => 0.0, :rrfm => 0.0)
    sig0      = Dict(:ffm => 0.0, :fpm => 0.0)
    var_resid = Float64.(sigma) .^ 2
    sig_resid = sqrt.(max.(var_resid, 0.0))
    region_log = MHDevFitRegion[]
    τ_vec = Float64.(tau)

    for (noise_type, idx_range) in regions
        idx = _to_indices(idx_range)
        if _is_powerlaw(noise_type)
            model = getfield(_MHDEV_POWERLAW, noise_type)
            v_sub = var_resid[idx]
            valid = v_sub .> 0
            if !any(valid)
                push!(region_log, MHDevFitRegion(noise_type, idx, 0.0, NaN, true))
                continue
            end
            τ_v = τ_vec[idx][valid]
            v_v = v_sub[valid]
            w_v = weights[idx][valid]
            # log(v) − slope·log(τ) = log(coeff·q)
            y = log.(v_v) .- model.slope .* log.(τ_v)
            μ, σμ = _weighted_mean(y, w_v)
            q_est = exp(μ) / model.coeff
            q_std = q_est * σμ                 # delta method: Var(exp μ) = exp(μ)²·Var(μ)
            q_total[noise_type] += q_est
            var_resid .-= model.coeff * q_est .* τ_vec .^ model.slope
            var_resid .= max.(var_resid, 0.0)
            sig_resid .= sqrt.(var_resid)
            push!(region_log, MHDevFitRegion(noise_type, idx, q_est, q_std, false))
        elseif _is_flicker(noise_type)
            model = getfield(_MHDEV_FLICKER, noise_type)
            s_sub = sig_resid[idx]
            valid = s_sub .> 0
            if !any(valid)
                push!(region_log, MHDevFitRegion(noise_type, idx, 0.0, NaN, true))
                continue
            end
            τ_v = τ_vec[idx][valid]
            s_v = s_sub[valid]
            w_v = weights[idx][valid]
            y = log.(s_v) .- model.slope .* log.(τ_v)
            μ, σμ = _weighted_mean(y, w_v)
            sig0_est = exp(μ)
            sig0_std = sig0_est * σμ
            sig0[noise_type] += sig0_est
            comp = sig0_est .* τ_vec .^ model.slope
            sig_resid .= sqrt.(max.(sig_resid .^ 2 .- comp .^ 2, 0.0))
            var_resid .= sig_resid .^ 2
            push!(region_log, MHDevFitRegion(noise_type, idx, sig0_est, sig0_std, false))
        else
            throw(ArgumentError("mhdev_fit: unknown noise_type $noise_type " *
                                "(use :wpm, :wfm, :rwfm, :rrfm, :ffm, :fpm)"))
        end
    end

    return MHDevFitResult(q_total[:wpm], q_total[:wfm],
                          q_total[:rwfm], q_total[:rrfm],
                          sig0[:ffm], sig0[:fpm],
                          region_log, var_resid)
end

# ── Helpers ──────────────────────────────────────────────────────────────────

_to_indices(r::UnitRange{Int}) = collect(r)
_to_indices(r::AbstractRange{<:Integer}) = collect(Int, r)
_to_indices(r::AbstractVector{<:Integer}) = convert(Vector{Int}, r)
_to_indices(i::Integer) = [Int(i)]

# legacy ci2weights.m
function _ci_to_weights(sigma::Vector{Float64},
                         lo::Vector{Float64},
                         hi::Vector{Float64},
                         method::Symbol)
    w = similar(sigma)
    @inbounds for i in eachindex(sigma)
        σ, l, u = sigma[i], lo[i], hi[i]
        wi = if method === :symmetric
            half = 0.5 * (u - l)
            half > 0 ? 1.0 / half^2 : 0.0
        elseif method === :conservative
            width = max(σ - l, u - σ)
            width > 0 ? 1.0 / width^2 : 0.0
        elseif method === :inverse
            width = u - l
            width > 0 ? 1.0 / width : 0.0
        else
            throw(ArgumentError("_ci_to_weights: unknown method $method"))
        end
        w[i] = isfinite(wi) ? wi : 0.0
    end
    return w
end

# legacy weightedMean.m — returns (μ, std-of-mean) with effective-N correction
function _weighted_mean(x::Vector{Float64}, w::Vector{Float64})
    sw = sum(w)
    sw > 0 || throw(ArgumentError("_weighted_mean: sum of weights is zero"))
    μ = sum(w .* x) / sw
    resid = x .- μ
    data_var = sum(w .* resid .^ 2) / sw
    n_eff    = sw^2 / sum(w .^ 2)
    var_mean = data_var / max(n_eff, 1.0)
    return (μ, sqrt(var_mean))
end
