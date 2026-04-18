# kf_diagnostics.jl — Innovation whiteness + bias diagnostics for KF output.
#
# References:
#   Anderson & Moore, "Optimal Filtering" (1979) §5  — innovations of a
#     well-tuned KF should be a white sequence with cov S = HP⁻H' + R.
#   Ljung & Box (1978) "On a measure of lack of fit in time series models"
#     Biometrika 65(2):297-303 — Q-statistic for autocorrelation testing.

using HypothesisTests
using Statistics

"""
    KFDiagnosticsResult

Three KF residual diagnostics. The Diagnostic 2 fields are `missing` when
`normalized_innovations` was not supplied to `kf_residual_diagnostics`.

Fields:
- `innov_lb_pvalue`, `innov_lb_passed`               — D1 (raw innovation whiteness)
- `norm_innov_lb_pvalue`, `norm_innov_lb_passed`     — D2 (normalized; optional)
- `resid_mean`, `resid_se`, `resid_bias_passed`      — D3 (posterior-residual bias)
- `lag`, `significance`, `n`                         — echo of run parameters
"""
struct KFDiagnosticsResult
    innov_lb_pvalue::Float64
    innov_lb_passed::Bool
    norm_innov_lb_pvalue::Union{Float64, Missing}
    norm_innov_lb_passed::Union{Bool, Missing}
    resid_mean::Float64
    resid_se::Float64
    resid_bias_passed::Bool
    lag::Int
    significance::Float64
    n::Int
end

"""
    kf_residual_diagnostics(innovations, residuals;
                            normalized_innovations = nothing,
                            lag = min(20, length(innovations) ÷ 5),
                            significance = 0.05) -> KFDiagnosticsResult

Three KF residual diagnostics:

1. **Raw innovation whiteness** — Ljung-Box on `innovations`.
   Passes iff `p > significance`.
2. **Normalized innovation whiteness** — Ljung-Box on `normalized_innovations`
   (typically `ν / sqrt(S)` where `S = H·P⁻·H' + R`). Optional; fields
   are `missing` when not supplied.
3. **Posterior-residual bias** — naive iid SE check `|mean| < 3·σ/√N`.

The bias check (D3) is statistically appropriate only when the residuals
are approximately iid. Posterior KF residuals are typically autocorrelated,
so D3 is a coarse health-check, not a rigorous test.
"""
function kf_residual_diagnostics(innovations::AbstractVector{<:Real},
                                  residuals::AbstractVector{<:Real};
                                  normalized_innovations::Union{Nothing, AbstractVector{<:Real}} = nothing,
                                  lag::Int = min(20, length(innovations) ÷ 5),
                                  significance::Float64 = 0.05)
    n = length(innovations)
    n >= 2                       || throw(ArgumentError("innovations must have length >= 2 (got $n)"))
    length(residuals) == n       || throw(ArgumentError("residuals length $(length(residuals)) != innovations length $n"))
    lag >= 1                     || throw(ArgumentError("lag must be >= 1 (got $lag)"))
    lag < n                      || throw(ArgumentError("lag must be < length(innovations)=$n (got $lag)"))

    innov_v = collect(Float64, innovations)
    resid_v = collect(Float64, residuals)

    # D1: Ljung-Box on raw innovations
    p_innov    = pvalue(LjungBoxTest(innov_v, lag))
    pass_innov = p_innov > significance

    # D2: Ljung-Box on normalized innovations (optional)
    if normalized_innovations === nothing
        p_norm    = missing
        pass_norm = missing
    else
        length(normalized_innovations) == n ||
            throw(ArgumentError("normalized_innovations length $(length(normalized_innovations)) != innovations length $n"))
        norm_v    = collect(Float64, normalized_innovations)
        p_norm    = pvalue(LjungBoxTest(norm_v, lag))
        pass_norm = p_norm > significance
    end

    # D3: posterior-residual bias under naive iid SE
    μ          = mean(resid_v)
    σ          = std(resid_v)
    se         = 3.0 * σ / sqrt(n)
    pass_bias  = abs(μ) < se

    return KFDiagnosticsResult(p_innov, pass_innov, p_norm, pass_norm,
                                μ, se, pass_bias,
                                lag, significance, n)
end
