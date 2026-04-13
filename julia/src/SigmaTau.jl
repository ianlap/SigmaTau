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

# ── Public API ────────────────────────────────────────────────────────────────

export DeviationResult, DevParams
export engine
export compute_ci, bias_correction, edf_for_result
export noise_id
export validate_phase_data, validate_tau0
export unpack_result
export adev, mdev, tdev, hdev, mhdev, ldev
export totdev, mtotdev, htotdev, mhtotdev
export KalmanConfig, KalmanResult, kalman_filter, safe_sqrt
export kf_filter                                   # alias for kalman_filter
export PredictConfig, PredictResult, kf_predict
# optimize exports deferred to PR #13

# ── Source files (order matters: later files call earlier definitions) ─────────

include("types.jl")      # DeviationResult, DevParams, helpers
include("validate.jl")   # validate_phase_data, validate_tau0, detrend_*
include("noise.jl")      # noise_id and supporting functions
include("stats.jl")      # EDF, CI, bias correction
include("engine.jl")     # shared engine
include("deviations.jl") # thin wrappers: adev, …
include("filter.jl")     # KalmanConfig, KalmanResult, kalman_filter
include("predict.jl")    # PredictConfig, PredictResult, kf_predict
# include("optimize.jl") — deferred to PR #13

# kf_filter is an alias for kalman_filter (matches problem-statement export name)
const kf_filter = kalman_filter

end # module
