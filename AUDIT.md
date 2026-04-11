# SigmaTau Code Audit

**Scope**: `julia/src/` and `matlab/legacy/` (note: `matlab/+sigmatau/` does not yet exist; all MATLAB code lives under `matlab/legacy/`)

**Date**: 2026-04-11

---

## 1. File Line Counts

### Julia — `julia/src/`

| File | Lines |
|------|------:|
| `SigmaTau.jl` | 39 |
| `validate.jl` | 37 |
| `types.jl` | 141 |
| `noise.jl` | 222 |
| `engine.jl` | 127 |
| `stats.jl` | 307 |
| `deviations.jl` | 686 |
| **Total** | **1 559** |

### MATLAB — `matlab/legacy/stablab/`

| File | Lines |
|------|------:|
| `avar.m` | 23 |
| `tdev.m` | 46 |
| `mdev.m` | 50 |
| `bias_correction.m` | 72 |
| `ldev.m` | 57 |
| `adev.m` | 74 |
| `hdev.m` | 74 |
| `compute_ci.m` | 76 |
| `mhdev.m` | 79 |
| `mhdev_noID.m` | 79 |
| `mhtotdev.m` | 98 |
| `totaldev_edf.m` | 104 |
| `totdev.m` | 115 |
| `mtotdev.m` | 119 |
| `mhtotdev_par.m` | 128 |
| `calculate_edf.m` | 128 |
| `powerlaw_noise.m` | 53 |
| `htotdev.m` | 164 |
| `noise_id.m` | 230 |
| `plotdev.m` | 340 |
| `compute_devs_from_file.m` | 293 |
| `compute_all_devs_from_file.m` | 266 |
| **Total** | **2 630** |

### MATLAB — `matlab/legacy/kflab/`

| File | Lines |
|------|------:|
| `avar.m` (none) | — |
| `text_progress.m` | 39 |
| `compre_dare_and_orig.m` | 24 |
| `save_adev_table.m` | 60 |
| `weightedLinearFit.m` | 56 |
| `ci2weights.m` | 67 |
| `weightedMean.m` | 73 |
| `save_kalman_results.m` | 84 |
| `display_steering_summary.m` | 90 |
| `merge_analyze_config.m` | 122 |
| `create_steering_plots.m` | 153 |
| `prompt_user_analyze_config.m` | 173 |
| `prompt_user_config.m` | 178 |
| `mhtot_fit.m` | 321 |
| `kalman_filter.m` | 344 |
| `optimize_kf_dare.m` | 295 |
| `optimize_kf.m` | 366 |
| `kf_predict.m` | 285 |
| `kf_predict_dare.m` | 306 |
| `test_optimizations.m` | 240 |
| `plot_optimization_results.m` | 309 |
| `mhdev_fit.m` | 433 |
| `analyze_steering.m` | 600 |
| `main_kf_pipeline_unified.m` | 1 239 |
| **Total** | **5 895** |

---

## 2. Functions Longer Than 50 Lines

### Julia (`julia/src/`)

| Function | File | Start line | End line | Length |
|----------|------|----------:|--------:|-------:|
| `engine()` | `engine.jl` | 27 | 94 | **68** |
| `_noise_id_b1rn()` | `noise.jl` | 104 | 165 | **62** |
| `_mtotdev_kernel()` | `deviations.jl` | 420 | 477 | **58** |
| `_htotdev_kernel()` | `deviations.jl` | 522 | 594 | **73** |

### MATLAB `stablab/`

| Function | File | Start line | End line | Length |
|----------|------|----------:|--------:|-------:|
| `htotdev()` | `htotdev.m` | 1 | 164 | **164** |
| `compute_all_devs_from_file()` | `compute_all_devs_from_file.m` | 1 | 252 | **252** |
| `compute_devs_from_file()` | `compute_devs_from_file.m` | 1 | 242 | **242** |
| `plotdev()` | `plotdev.m` | 1 | 237 | **237** |
| `add_data_table()` | `plotdev.m` | 238 | 327 | **90** |
| `mtotdev()` | `mtotdev.m` | 1 | 119 | **119** |
| `totdev()` | `totdev.m` | 1 | 115 | **115** |
| `mhtotdev()` | `mhtotdev.m` | 1 | 98 | **98** |
| `mhdev()` | `mhdev.m` | 1 | 79 | **79** |
| `mhdev_noID()` | `mhdev_noID.m` | 1 | 79 | **79** |
| `noiseID_B1Rn()` | `noise_id.m` | 109 | 185 | **77** |
| `adev()` | `adev.m` | 1 | 74 | **74** |
| `hdev()` | `hdev.m` | 1 | 74 | **74** |
| `bias_correction()` | `bias_correction.m` | 1 | 72 | **72** |
| `ldev()` | `ldev.m` | 1 | 57 | **57** |
| `totaldev_edf()` | `totaldev_edf.m` | 1 | 56 | **56** |
| `compute_ci()` | `compute_ci.m` | 1 | 59 | **59** |

### MATLAB `kflab/`

| Function | File | Start line | End line | Length |
|----------|------|----------:|--------:|-------:|
| `analyze_steering()` | `analyze_steering.m` | 1 | 506 | **506** |
| `write_analysis_summary()` | `analyze_steering.m` | 507 | 600 | **94** |
| `mhdev_fit()` | `mhdev_fit.m` | 1 | 207 | **207** |
| `updatePlot()` | `mhdev_fit.m` | 308 | 419 | **112** |
| `mhtot_fit_wplot()` | `mhtot_fit.m` | 1 | 160 | **160** |
| `updatePlot()` | `mhtot_fit.m` | 221 | 307 | **87** |
| `kalman_filter()` | `kalman_filter.m` | 1 | 233 | **233** |
| `initialize_kf()` | `kalman_filter.m` | 234 | 304 | **71** |
| `kf_predict()` | `kf_predict.m` | 1 | 150 | **150** |
| `compute_prediction_rms()` | `kf_predict.m` | 206 | 285 | **80** |
| `kf_predict_dare()` | `kf_predict_dare.m` | 1 | 108 | **108** |
| `compute_prediction_rms_dare()` | `kf_predict_dare.m` | 109 | 194 | **86** |
| `optimize_kf()` | `optimize_kf.m` | 1 | 75 | **75** |
| `set_optimization_defaults()` | `optimize_kf.m` | 76 | 131 | **56** |
| `optimize_grid()` | `optimize_kf.m` | 132 | 205 | **74** |
| `optimize_fmincon()` | `optimize_kf.m` | 254 | 343 | **90** |
| Many functions | `main_kf_pipeline_unified.m` | 1 | 1 239 | **1 239** |

`main_kf_pipeline_unified.m` contains 15 named sub-functions, the largest of which are
`create_optimization_plots()` (lines 569–626, **58 lines**), `save_all_results()` (lines 932–951, **20 lines**),
and `write_summary_file()` (lines 952–1041, **90 lines**).  The top-level function body spans lines 1–337 (**337 lines**).

---

## 3. Duplicated Code Blocks

### D1 — `convert_units()` in two stablab scripts

`compute_devs_from_file.m` lines 279–293 and `compute_all_devs_from_file.m` lines 253–266 contain near-identical
`convert_units()` helpers.  The only differences are a whitespace variant and a slightly different warning string.

### D2 — `showDataTable()`, `fitFixedSlope()`, `addIndexLabels()` duplicated in fit files

`mhdev_fit.m` and `mhtot_fit.m` each define their own copies:

| Function | `mhdev_fit.m` | `mhtot_fit.m` |
|----------|--------------|--------------|
| `showDataTable()` | lines 208–221 | lines 161–174 |
| `fitFixedSlope()` | lines 222–251 | lines 175–205 |
| `addIndexLabels()` | lines 420–433 | lines 308–321 |

Bodies are functionally identical; only the outer function context differs.

### D3 — `set_optimization_defaults()` in two optimize scripts

`optimize_kf.m` lines 76–131 and `optimize_kf_dare.m` lines 242–295 define the same default-merging logic.
The DARE version even carries the comment `% SAME AS ORIGINAL` (line 244).

### D4 — Six plotting sub-functions duplicated between `plot_optimization_results.m` and `main_kf_pipeline_unified.m`

`main_kf_pipeline_unified.m` inlines `plot_rms_comparison` (lines 627–669), `plot_improvement` (lines 670–696),
`plot_parameter_comparison` (lines 697–726), `plot_2d_cost_surface` (lines 727–802),
`plot_1d_slice` (lines 803–867), and `add_summary_title` (lines 868–880).
All six also exist as top-level functions in `plot_optimization_results.m` (lines 70–309).
The bodies are character-for-character identical except for indentation.

### D5 — `mhdev.m` and `mhdev_noID.m` are near-identical files

The only differences are:
- Function name on line 1.
- `noise_id` call on line 47 is commented out in `mhdev_noID.m`.
- Lines 78–79 (CI computation) are commented out in `mhdev_noID.m`.

`mhdev_noID.m` is flagged in the architecture notes as dead code to be deleted.

### D6 — Symmetric reflection loop triplicated in Julia total-deviation kernels

The pattern `for j in 1:seg_len; ext[j] = pd[seg_len - j + 1]; ext[seg_len+j] = pd[j]; ext[2*seg_len+j] = pd[seg_len-j+1]; end`
(or equivalent) appears verbatim in:

| Kernel | File | Lines |
|--------|------|-------|
| `_mtotdev_kernel()` | `deviations.jl` | 451–455 |
| `_htotdev_kernel()` | `deviations.jl` | 565–569 |
| `_mhtotdev_kernel()` | `deviations.jl` | 652–656 |

Extracting a `_symmetric_reflect(v)` helper would eliminate the repetition.

### D7 — `_simple_avar()` duplicates the inner loop of `_adev_kernel()`

`noise.jl:199–205` and `deviations.jl:47–54` implement the same overlapping second-difference variance.
The only difference is that `_adev_kernel` divides by `tau0²` (it is called from the engine which already
knows `tau0`) while `_simple_avar` does not.

### D8 — `_simple_mdev()` duplicates the inner loop of `_mdev_kernel()`

`noise.jl:212–222` and `deviations.jl:100–111` implement the same cumsum-prefix-sum MDEV.
`_simple_mdev` returns the deviation directly; `_mdev_kernel` returns the raw variance and lets the engine
take the square root.

### D9 — `compute_covariance_uncertainties()` and `compute_covariance_uncertainties_silent()` in `main_kf_pipeline_unified.m`

Lines 1075–1131 and 1132–1188 are identical except for the function name (line 1 of each) and two comment
strings.  A single function with a `verbose` flag would suffice.

---

## 4. Helper Functions Called Only Once (Inlining Candidates)

### Julia

| Function | Defined | Called | Notes |
|----------|---------|--------|-------|
| `_make_result()` | `types.jl:93` | **Never called** | Dead code — the engine constructs `DeviationResult` directly. The docstring in `SigmaTau.jl` still references it. |
| `_empty_result()` | `engine.jl:98` | `engine.jl:48` (once) | 6-line helper; inlinable. |
| `_apply_bias()` | `engine.jl:111` | `engine.jl:93` (once) | 17-line helper; readable enough to keep, but only one call site. |
| `_z_from_confidence()` | `stats.jl:300` | `stats.jl:287` (once, inside loop) | 8-line helper inside `compute_ci`; inlinable. |

### MATLAB

| Function | Defined in | Called | Notes |
|----------|-----------|--------|-------|
| `safe_sqrt()` | `kalman_filter.m:305` | Lines 211, 213–214, 220–222 (6 call sites) | Called six times — **not** a single-use candidate; the name is descriptive. |
| `make_symmetric()` | `kalman_filter.m:318` | Line 170 (commented out) | The only call site is commented out (`%P = make_symmetric(...)`). Effectively dead code. |

---

## 5. Summary of Priority Issues

| Priority | Issue | Location |
|----------|-------|----------|
| 🔴 High | `main_kf_pipeline_unified.m` is 1 239 lines with 15 sub-functions; exceeds 100-line limit by ~12× | `kflab/main_kf_pipeline_unified.m` |
| 🔴 High | Six plotting functions copied verbatim from `plot_optimization_results.m` into `main_kf_pipeline_unified.m` (D4) | lines 627–880 |
| 🔴 High | `_make_result()` is dead code — defined but never called (Julia) | `types.jl:93` |
| 🟠 Medium | `mhdev_noID.m` is dead code (confirmed in architecture notes) | `stablab/mhdev_noID.m` |
| 🟠 Medium | `make_symmetric()` in `kalman_filter.m` has its only call site commented out | `kalman_filter.m:318` |
| 🟠 Medium | Symmetric reflection loop triplicated across three total-deviation kernels (D6) | `deviations.jl:451, 565, 652` |
| 🟡 Low | `convert_units()` duplicated between two stablab scripts (D1) | `stablab/` |
| 🟡 Low | `showDataTable`, `fitFixedSlope`, `addIndexLabels` duplicated between fit files (D2) | `kflab/` |
| 🟡 Low | `set_optimization_defaults()` duplicated between optimize scripts (D3) | `kflab/` |
| 🟡 Low | `compute_covariance_uncertainties` / `_silent` near-identical twins (D9) | `main_kf_pipeline_unified.m:1075` |
| 🟡 Low | `_simple_avar` / `_adev_kernel` and `_simple_mdev` / `_mdev_kernel` duplicate algorithms (D7, D8) | `noise.jl`, `deviations.jl` |
| 🟡 Low | `_empty_result` and `_z_from_confidence` each have a single call site and could be inlined | `engine.jl:98`, `stats.jl:300` |
