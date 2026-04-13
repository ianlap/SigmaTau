# engine.jl — Shared deviation computation engine
# All 10 deviation functions are thin wrappers that supply a kernel + DevParams.
# Reference: deviation-engine skill architecture

"""
    engine(x, tau0, m_list, kernel, params) → DeviationResult

Shared deviation computation engine.  Each deviation wrapper passes:
- `kernel(x, m, tau0) → (variance::Float64, neff::Int)` — the core computation.
  Kernels return **variance**, not deviation; the engine takes the sqrt.
- `params::DevParams` — configuration (name, m-list and EDF basics)

The engine handles: input validation, default m_list generation, noise
identification, kernel dispatch, EDF computation, optional bias correction,
and result construction.

# Kernel contract
- Input:  full phase vector `x`, averaging factor `m`, sampling interval `tau0`
- Output: `(variance, neff)` where `neff` is the effective sample count
- Return `(NaN, 0)` when there are insufficient samples for this `m`

# Notes
- CI is not computed here; call `compute_ci(result)` afterwards if needed.
- Total deviation kernels (totdev, …) handle their own extended-data logic
  internally; the engine loop is identical for all deviation types.
"""
function engine(
    x         :: AbstractVector{<:Real},
    tau0      :: Real,
    m_list    :: Union{Nothing, AbstractVector{<:Integer}},
    kernel    :: Function,
    params    :: DevParams;
    data_type :: Symbol = :phase,
)
    # Frequency-to-phase conversion — CLAUDE.md §Architecture
    # cumsum(y)*tau0 produces phase in seconds from fractional-frequency samples.
    if data_type === :freq
        x = cumsum(x) .* tau0
    elseif data_type !== :phase
        throw(ArgumentError("data_type must be :phase or :freq, got :$data_type"))
    end

    x    = validate_phase_data(x)
    tau0 = validate_tau0(tau0)
    N    = length(x)

    ms   = Vector{Int}(something(m_list, _default_mlist(N, params.min_factor)))
    isempty(ms) && return _empty_result(params.name, tau0, N)

    # Noise identification (returns Float64 vector; NaN where it fails)
    alpha_float = noise_id(x, ms, "phase", params.dmin, params.dmax)

    tau  = ms .* tau0
    dev  = Vector{Float64}(undef, length(ms))
    neff = Vector{Int}(undef, length(ms))
    edf  = Vector{Float64}(undef, length(ms))

    T_rec = (N - 1) * tau0   # record duration for total deviation EDF

    for (k, m) in enumerate(ms)
        var_val, n = kernel(x, m, tau0)

        if n <= 0 || isnan(var_val)
            dev[k]  = NaN
            neff[k] = 0
            edf[k]  = NaN
            continue
        end

        dev[k]  = sqrt(max(var_val, 0.0))   # guard against fp rounding below zero
        neff[k] = n

        # EDF computation
        alpha_k = isnan(alpha_float[k]) ? 0 : round(Int, alpha_float[k])
        total_type = _total_type_for_name(params.name)
        if !isnothing(total_type)
            edf[k] = totaldev_edf(total_type, alpha_k, T_rec, tau[k])
        else
            F = params.F_fn(m)
            edf[k] = calculate_edf(alpha_k, params.d, m, F, 1, N)
        end
    end

    # Build result with NaN CI (caller may invoke compute_ci)
    alpha_int = [isnan(a) ? 0 : round(Int, a) for a in alpha_float]
    result = DeviationResult(
        tau, dev, edf, fill(NaN, length(ms), 2),
        alpha_int, neff,
        tau0, N, params.name, 0.683
    )

    # Bias correction (applied in-place on a new allocation since struct is immutable)
    bias_type = _bias_type_for_name(params.name)
    isnothing(bias_type) && return result
    return _apply_bias(result, bias_type, T_rec)
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _total_type_for_name(name::String)
    name == "totdev"  && return "totvar"
    name == "mtotdev" && return "mtot"
    name == "htotdev" && return "htot"
    name == "mhtotdev" && return "mhtot"
    return nothing
end

function _bias_type_for_name(name::String)
    name == "totdev" && return "totvar"
    name == "htotdev" && return "htot"
    return nothing
end

function _empty_result(name::String, tau0::Float64, N::Int)
    DeviationResult(
        Float64[], Float64[], Float64[], Matrix{Float64}(undef, 0, 2),
        Int[], Int[], tau0, N, name, 0.683
    )
end

"""
    _apply_bias(result, bias_type, T_rec) → DeviationResult

Return a new `DeviationResult` with bias-corrected deviation and EDF.
Divides `deviation` (and `ci` if not NaN) by the bias factor B(α).
"""
function _apply_bias(result::DeviationResult, bias_type::String, T_rec::Real)
    B = bias_correction(result.alpha, bias_type, result.tau, T_rec)

    dev_corr = result.deviation ./ B

    ci_corr = if any(!isnan, result.ci)
        result.ci ./ B
    else
        result.ci
    end

    return DeviationResult(
        result.tau, dev_corr, result.edf, ci_corr,
        result.alpha, result.neff,
        result.tau0, result.N, result.method, result.confidence
    )
end
