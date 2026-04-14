# SigmaTau — Foundational Mandates

This document contains absolute mandates for the SigmaTau project. These instructions take precedence over all other workflows, tool defaults, or general guidelines.

## 1. Architectural Invariants

### Deviation Engine (The "Kernel" Pattern)
- **Shared Engine**: All 10 deviation functions MUST use the shared `engine` (`matlab/+sigmatau/+dev/engine.m` and `julia/src/engine.jl`). Do NOT duplicate boilerplate.
- **Kernel Interface**: Each deviation is a thin wrapper that defines a kernel function `@(x, m, tau0)`.
- **Variance return**: Kernels MUST return variance ($ \sigma^2 $), not deviation ($ \sigma $). The engine handles the square root.
- **Data Types**: All deviation wrappers MUST accept both phase and frequency data via a `data_type` keyword/argument. Frequency-to-phase conversion (`cumsum(y)*tau0`) MUST live in the engine.
- **O(N) Complexity**: `mdev` and `mhdev` MUST use `cumsum`-based prefix sums for $ O(N) $ computation. Do NOT use $ O(N \cdot m) $ loops.

### Kalman Filter (The "Struct" Pattern)
- **Input/Output**: All Kalman filter, prediction, and optimization functions MUST use a `config` struct for inputs and a `results` struct for outputs. No positional arguments for configurations.
- **Composition**: The KF pipeline MUST be built from composable functions (`load_data` → `engine` → `noise_fit` → `filter` → `optimize`). No monolithic scripts.

## 2. Numerical & Algorithmic Mandates

### Stability Analysis
- **htotdev m=1**: MUST use the overlapping HDEV algorithm, not the total deviation reflection algorithm.
- **mhtotdev EDF**: Use approximate coefficients from mc fitting (no published model exists).
- **Bias Correction**: Lookup tables for `totvar`, `mtot`, and `htot` MUST match NIST SP1065 exactly.
- **Noise ID**: Implement dual-path identification:
    - If $ N_{eff} \geq 30 $: Use lag-1 ACF.
    - If $ N_{eff} < 30 $: Use $ B_1 $ ratio + $ R(n) $ lookup.
- **Totdev Denominator**: Use $ 2(N-2)(m\tau_0)^2 $ as per SP1065 Eq. 25 (not $ N-1 $).

### Kalman Filter & Steering
- **PID Convention**: The integral term MUST accumulate phase error: `sumx += x(1)`.
- **Q-Matrix**: Elements MUST follow exact continuous-time noise model integration (powers of $ \tau $, $ \tau^3/3 $, $ \tau^5/20 $, etc.). No approximations.
- **Covariance**: Use `P = (I - K*H) * P` (standard form). Use `safe_sqrt` for diagonal elements to handle numerical drift.

### Accuracy Targets
- **Internal Consistency**: MATLAB and Julia implementations MUST match to $< 10^{-12}$ relative error.
- **Cross-Validation**: Pass if relative error $< 10^{-10}$ across all output fields (tau, dev, edf, ci).

## 3. Code Style & Standards

- **Function Length**: No function longer than 100 lines. If it exceeds this, it should be attempted to be split.
- **Documentation**: All non-trivial blocks MUST cite equations: `% SP1065 Eq. 12` or `# Greenhall (2003) Eq. 8`.
- **Magic Numbers**: Zero tolerance for magic numbers. Use named constants (e.g., `CONFIDENCE_DEFAULT = 0.683`).
- **Dependencies**: `Plots.jl` MUST be a package extension in Julia, not a hard dependency.
- **Namespace**: MATLAB code MUST reside in the `+sigmatau` package namespace.
