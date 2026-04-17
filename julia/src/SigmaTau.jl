"""
    SigmaTau

Frequency stability analysis and clock steering. Export groups (see `export`
blocks below for the full list):

- **Deviations (10 functions):** `adev`, `mdev`, `tdev`, `hdev`, `mhdev`,
  `ldev`, `totdev`, `mtotdev`, `htotdev`, `mhtotdev`.
- **Deviation engine / CI:** `engine`, `DevParams`, `DeviationResult`,
  `compute_ci`, `bias_correction`, `edf_for_result`, `unpack_result`.
- **Input validation:** `validate_phase_data`, `validate_tau0`.
- **Noise identification (SP1065 §5.6):** `noise_id`.
- **Noise fitting (MHDEV → q):** `mhdev_fit`, `MHDevFitResult`,
  `MHDevFitRegion`.
- **Noise generation (Kasdin & Walter):** `generate_power_law_noise`,
  `generate_composite_noise`.
- **Clock models:** `ClockNoiseParams`, `ClockModel2`, `ClockModel3`,
  `ClockModelDiurnal`, `build_phi`, `build_Q`, `build_H`, `nstates`,
  `sigma_y_theory`, `steady_state_covariance`, `steady_state_gain`.
- **h↔q conversion (Wu 2023 convention):** `h_to_q`, `q_to_h`.
- **Kalman filter:** `kalman_filter` (alias `kf_filter`), `KalmanResult`,
  `predict_holdover`, `HoldoverResult`.
- **NLL optimization / ALS tuning:** `optimize_nll`, `innovation_nll`,
  `als_fit`, `OptimizeNLLResult`.
- **ML features (for dataset-to-q regression):** `CANONICAL_TAU_GRID`,
  `CANONICAL_M_LIST`, `FEATURE_NAMES`, `compute_feature_vector`.
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
export KalmanResult, kalman_filter
export kf_filter                                   # alias for kalman_filter
export HoldoverResult, predict_holdover
export optimize_nll, innovation_nll, als_fit, OptimizeNLLResult
export ClockNoiseParams, ClockModel2, ClockModel3, ClockModelDiurnal
export build_phi, build_Q, build_H, sigma_y_theory, h_to_q, q_to_h, steady_state_covariance, steady_state_gain, nstates
export generate_power_law_noise
export generate_composite_noise
export CANONICAL_TAU_GRID, CANONICAL_M_LIST, FEATURE_NAMES, compute_feature_vector

# ── Source files (order matters: later files call earlier definitions) ─────────

include("types.jl")      # DeviationResult, DevParams, helpers
include("validate.jl")   # validate_phase_data, validate_tau0, detrend_*
include("noise.jl")      # noise_id and supporting functions
include("noise_gen.jl")  # generate_power_law_noise (Kasdin & Walter, 1992)
include("stats.jl")      # EDF, CI, bias correction
include("engine.jl")     # shared engine
include("deviations.jl") # thin wrappers: adev, …
include("ml_features.jl") # canonical τ grid and 196-feature extraction
include("noise_fit.jl")  # mhdev_fit (port of legacy kflab/mhdev_fit.m)
include("clock_model.jl")
include("filter.jl")
include("predict.jl")
include("optimize.jl")
include("als_fit.jl")

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
