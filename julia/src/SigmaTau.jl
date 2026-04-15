"""
    SigmaTau

Frequency stability analysis and clock steering.

Exports:
- `DeviationResult`: result struct for all deviation computations
- `DevParams`: engine configuration struct (used by deviation wrappers)
- `engine`: shared deviation computation engine
- `compute_ci`: fill EDF and confidence intervals on a result
- `bias_correction`: bias factor lookup (totvar / mtot / htot)
- `noise_id`: power-law noise identification (SP1065 §5.6)
"""
module SigmaTau

using Statistics
using LinearAlgebra
using StatsFuns: norminvcdf, chisqinvcdf
using PrecompileTools: @setup_workload, @compile_workload
using Random

# ── Public API ────────────────────────────────────────────────────────────────

export DeviationResult, DevParams
export engine
export compute_ci, bias_correction, edf_for_result
export noise_id
export mhdev_fit, MHDevFitResult, MHDevFitRegion
export validate_phase_data, validate_tau0
export unpack_result
export adev, mdev, tdev, hdev, mhdev, ldev
export totdev, mtotdev, htotdev, mhtotdev
export KalmanConfig, KalmanResult, kalman_filter
export kf_filter                                   # alias for kalman_filter
export PredictConfig, PredictResult, kf_predict
export OptimizeConfig, OptimizeResult, optimize_kf
export generate_power_law_noise

# ── Source files (order matters: later files call earlier definitions) ─────────

include("types.jl")      # DeviationResult, DevParams, helpers
include("validate.jl")   # validate_phase_data, validate_tau0, detrend_*
include("noise.jl")      # noise_id and supporting functions
include("noise_gen.jl")  # generate_power_law_noise (Kasdin & Walter, 1992)
include("stats.jl")      # EDF, CI, bias correction
include("engine.jl")     # shared engine
include("deviations.jl") # thin wrappers: adev, …
include("noise_fit.jl")  # mhdev_fit (port of legacy kflab/mhdev_fit.m)
include("filter.jl")     # KalmanConfig, KalmanResult, kalman_filter
include("predict.jl")    # PredictConfig, PredictResult, kf_predict
include("optimize.jl")   # OptimizeConfig, OptimizeResult, optimize_kf

# kf_filter is an alias for kalman_filter (matches problem-statement export name)
const kf_filter = kalman_filter

# ── Precompile workload ───────────────────────────────────────────────────────
# Exercise every deviation once at package precompile time. This caches the JIT
# specializations (engine dispatch, kernel, noise_id, edf, bias, CI) to disk,
# so `using SigmaTau` gives warm code immediately — no runtime warmup needed.
#
# N=64 is the minimum that lets all 10 kernels succeed with default m_list
# (mhtotdev's min_factor=4 ⇒ N ≥ 8 for m=2^0,2^1; higher octaves get NaN'd out,
# which still compiles the same code paths).
@setup_workload begin
    _pc_x = cumsum(randn(Xoshiro(0), 64))
    _pc_tau0 = 1.0
    @compile_workload begin
        for fn in (adev, mdev, hdev, mhdev, tdev, ldev,
                   totdev, mtotdev, htotdev, mhtotdev)
            fn(_pc_x, _pc_tau0)
        end
    end
end

end # module
