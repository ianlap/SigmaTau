# SigmaTau — σ(τ) Frequency Stability Analysis & Clock Control

## Project
Combined MATLAB + Julia package for frequency stability analysis (10 NIST SP1065 deviations) and Kalman filter clock steering. Refactored from legacy "AllanLab" + "StabLab.jl" + "KalmanLab.jl".

**Author**: Ian Lapinski (ian.lapinski.99@gmail.com)
**License**: MIT

## Build & Test

```bash
# MATLAB — run from repo root
cd matlab && matlab -batch "addpath(genpath('.')); run('tests/run_all.m')"

# Julia — run from julia/ directory  
cd julia && julia --project=. -e 'using Pkg; Pkg.test()'
```

## Code Style

- MATLAB uses `+sigmatau` package namespace. User calls `sigmatau.dev.adev(x, tau0)`.
- Julia module is `SigmaTau`. User calls `using SigmaTau; adev(x, tau0)`.
- No function longer than 100 lines. If longer, split it.
- Struct in, struct out. No 5+ positional args — use a config struct.
- Comments cite equations: `% SP1065 Eq. 12` or `# Greenhall (2003) Eq. 8`.
- No magic numbers. Name constants: `CONFIDENCE_DEFAULT = 0.683`.
- MATLAB arrays are 1-indexed. Julia arrays are 1-indexed. No off-by-one risk between languages.

## Architecture

IMPORTANT: All 10 deviation functions share a common engine. Do NOT duplicate boilerplate across deviation functions. Each deviation is a thin wrapper that passes a kernel function to the shared engine.

IMPORTANT: All deviation functions accept both phase and frequency data via a `data_type` keyword (:phase default, :freq). Frequency-to-phase conversion (cumsum(y)*tau0) lives in the engine so every wrapper gets it for free. Do NOT implement conversion in individual wrappers.

```
matlab/+sigmatau/
  +dev/engine.m          — shared deviation computation (validate, loop, edf, ci)
  +dev/adev.m ... ldev.m — thin wrappers: define kernel + params, call engine
  +noise/identify.m      — lag1ACF + B1/Rn noise identification
  +noise/generate.m      — power-law noise (Kasdin FFT)
  +stats/edf.m           — equivalent degrees of freedom (Greenhall)
  +stats/ci.m            — confidence intervals (chi-squared + fallback)
  +stats/bias.m          — bias correction (SP1065)
  +kf/filter.m           — Kalman filter (struct config in, struct results out)
  +kf/predict.m          — multi-step prediction / holdover
  +kf/optimize.m         — Q-parameter grid search
  +kf/pipeline.m         — end-to-end: data → deviation → noise fit → KF
  +steering/analyze.m    — PID/PD steering sweeps
  +plot/stability.m      — log-log σ(τ) plots
  +util/                 — validate, mlist_default, progress
```

Julia mirrors this layout in `julia/src/`.

## Critical Implementation Details

- **htotdev m=1**: Uses overlapping HDEV, not the total deviation algorithm. Do not change this.
- **mhtotdev EDF**: No published model. Uses approximate coefficients from FCS 2001.
- **Bias correction**: Different formulas for totvar, mtot, htot. Lookup tables must stay exact.
- **Noise ID fallback**: N_eff ≥ 30 → lag-1 ACF. N_eff < 30 → B1 ratio + R(n). Both paths required.
- **KF PID convention**: Integral accumulates `sumx += x[1]` (phase error). This matches masterclock convention.
- **Plots.jl**: Must be a package extension, not a hard dependency. `using SigmaTau` loads without Plots.

## Verification

Every refactored function must produce identical numerical output to the legacy code. Test with:
1. White PM noise (α=2): slope should be τ^(-1) for ADEV
2. White FM noise (α=0): slope should be τ^(-1/2) for ADEV
3. RWFM noise (α=-2): slope should be τ^(+1/2) for ADEV
4. Cross-validate MATLAB vs Julia to < 1e-12 relative error

## Files to Delete
- `mhdev_noID.m` — dead code
- `mhtotdev_par.m` — merge parallelism into mhtotdev as option
- `compute_devs_from_file.m` → move to examples/
- `compute_all_devs_from_file.m` → move to examples/

## References
- NIST SP1065: Riley & Howe, "Handbook of Frequency Stability Analysis"
- Greenhall & Riley, "Uncertainty of Stability Variances," PTTI 2003
- IEEE Std 1139-2022

## Known Legacy Bugs (must be correct in new code)
- htotdev bias correction: verify direction (multiply vs divide) against SP1065 and Julia output
- htotdev EDF loop: after trimming invalid taus, loop over numel(tau) not numel(valid)
- mhtotdev Neff: verify segment count formula — is it N-4m+1 or N-3m? Check against Riley and Julia
- totdev denominator: SP1065 §5.2.11 uses 2(N-1)(mτ₀)², not 2(N-2)(mτ₀)²
