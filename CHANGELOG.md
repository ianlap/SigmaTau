# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Regression test for non-contiguous index vectors in `mhdev_fit`.
- CLI test for empty-result display paths.

### Changed
- **Engine Signature Refactor**: All deviation kernels in `julia/src/engine.jl` and `matlab/+sigmatau/+dev/engine.m` now follow a 4-argument signature `(x, m, tau0, x_cs)`, where `x_cs` is the pre-computed prefix sum of `x`. This enables $O(N)$ computation for `mdev`, `mhdev`, and their derivatives.
- **TDEV/LDEV Consolidation**: `tdev` and `ldev` are now architectural wrappers around `mdev` and `mhdev` respectively, scaling the variance results by the appropriate SP1065 constants. This eliminates boilerplate and ensures exact numerical parity across implementations.

### Fixed
- **Index Vector Handling**: Fixed `_to_indices` in `mhdev_fit` to correctly handle non-contiguous index vectors (e.g., `[1, 3, 7]`), preventing errors when fitting specific tau regions.
