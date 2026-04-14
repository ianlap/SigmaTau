# SigmaTau — Critical Audit & Refactor Plan

## Current Status: Phase 1 COMPLETED
As of 2026-04-14, Phase 1 (Critical Correctness & Math) has been implemented and verified.
- Fixed negative variance masking and filter averaging factors in `engine.m` and `engine.jl`.
- Standardized CI fallback (Gaussian for EDF < 1 or missing Statistics Toolbox) with explicit warnings.
- Updated TOTDEV denominator to `2(N-1)(mτ₀)²` in both languages (SP1065 compliance).
- Clamped bias correction noise slopes to valid ranges with warnings.
- Verified all deviations via Julia test suite (`175/175 Pass`) and MATLAB test suite (including cross-validation with Julia).
- **Note:** Slope tolerance in `test_deviations.m` was relaxed to **10%** (0.10) per user hint.

## Resumption Guide (Start Here for Phase 2)
1. **Context Check:** Review `matlab/+sigmatau/+noise/identify.m`. It currently duplicates deviation kernels (ADEV/MDEV), which violates the "shared engine" mandate in `GEMINI.md`.
2. **First Task:** Execute Phase 2, Step 1 (Refactor Noise Identification) by replacing `simple_avar` and `simple_mdev` with calls to the official `sigmatau.dev.adev` and `mdev` functions.
3. **Validation:** Run `matlab/tests/test_noise.m` and `test_deviations.m` to ensure noise identification still functions correctly after the refactor.

---

## Phased Implementation Plan

### Phase 2: Architectural Fixes & Completion (Medium Risk)
Focus on architectural mandates from `GEMINI.md` and completing missing ports.
1. **Refactor Noise Identification:** In `matlab/+sigmatau/+noise/identify.m`, replace the duplicated `simple_avar` and `simple_mdev` logic with calls to the shared engine kernels (`sigmatau.dev.adev`/`mdev`), adhering to the "shared engine" rule.
2. **Fix EDF Dispatch:** In `matlab/+sigmatau/+dev/engine.m`, remove the implicit `params.name` string-matching for EDF dispatch. Add an explicit `edf_method` field to `DevParams` so wrappers explicitly declare their intent.
3. **Port MATLAB Kalman Filter:** Implement the missing Kalman filter port in `matlab/+sigmatau/+kf/` (including `kalman_filter`, `build_phi`, `build_Q`, `update_pid`). Ensure it strictly follows the "Struct In, Struct Out" pattern (config struct in, results struct out) and matches Julia to < 10^-12 relative error.

### Phase 3: Performance & Tech Debt (Low Risk)
Focus on optimization, cleanup, and adherence to style guidelines.
1. **Optimize Julia Kernels:** In `julia/src/deviations.jl` (`mtotdev`, `htotdev`, `mhtotdev`), move buffer allocations outside the `nsubs` loop and use in-place operations (`copyto!`, `.=`) to reduce GC pressure.
2. **Optimize Julia KF Allocations:** Refactor `P_history` in `kalman_filter` from a `Vector{Matrix{Float64}}` to a 3D array (`Array{Float64, 3}`) to reduce allocations.
3. **Optimize MATLAB Total Kernels:** Refactor total deviation kernels to use pre-allocated buffers and direct indexing instead of repeated array concatenations (`xstar = [...]`).
4. **Enforce File Length Limits:** Refactor files exceeding the 100-line limit (e.g., `calculate_edf.m`, `identify.m`, `optimize.m`) by extracting nested helpers to private directories (`+stats/private/`, etc.).
5. **Legacy Purge:** Remove dead code (e.g., `mhdev_noID.m`, `mhtotdev_par.m` in `matlab/legacy/stablab/`) and move utility scripts to `examples/`.

## Verification & Testing
1. **Cross-Validation:** Ensure MATLAB and Julia implementations match to < 10^-12 relative error across all output fields for all modified deviations.
2. **SP1065 Compliance:** Validate the updated TOTDEV implementation against known reference values (e.g., Stable32 output or verified datasets).
3. **Test Suite Expansion:** Add tests for the `empty_result` path, missing slope cases in `mhtotdev`, and explicit coverage for `noise_fit`.
4. **Kalman Filter Parity:** Run the `cross-validate` command for the newly ported MATLAB Kalman filter to guarantee strict numerical parity with Julia.
