# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `OptimizeNLLResult` return struct from `optimize_nll` exposing `noise`, `nll`, `n_evals`, and `converged` fields (previously only the `ClockNoiseParams` was returned; callers had to re-run `innovation_nll` for the NLL and had no access to convergence state).
- Regression test for non-contiguous index vectors in `mhdev_fit`.
- CLI test for empty-result display paths.

### Changed
- **Engine Signature Refactor**: All deviation kernels in `julia/src/engine.jl` and `matlab/+sigmatau/+dev/engine.m` now follow a 4-argument signature `(x, m, tau0, x_cs)`, where `x_cs = cumsum([0; x])` is the engine-precomputed phase prefix sum, shared across kernels. This enables $O(N)$ computation for `mdev`, `mhdev`, `mtotdev`, and `mhtotdev`. Kernels that do not consume the prefix sum (`adev`, `hdev`, `totdev`, `htotdev`) accept the 4th argument as `~`. Closes GEMINI.md Goal G2.
- **TDEV/LDEV Consolidation**: `tdev` and `ldev` are now architectural wrappers around `mdev` and `mhdev` respectively, scaling the variance results by the appropriate SP1065 constants. This eliminates boilerplate and ensures exact numerical parity across implementations.

### Fixed
- **Index Vector Handling**: Fixed `_to_indices` in `mhdev_fit` to correctly handle non-contiguous index vectors (e.g., `[1, 3, 7]`), preventing errors when fitting specific tau regions.
- Regenerated MATLAB↔Julia cross-val fixture (stale since commit `f046d50` on Apr 14 14:18; Method A → Method B MHDEV kernel form). Cross-val now passes at 2e-10 across all 10 deviations × 3 noise types.
