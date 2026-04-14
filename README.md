# SigmaTau

Frequency stability analysis and Kalman clock steering in MATLAB and Julia.

SigmaTau is a twin-language package for time-and-frequency work: compute the ten NIST SP1065 deviations, identify power-law noise, and run Kalman filters for clock steering and holdover. The MATLAB and Julia implementations mirror each other and cross-validate to machine precision.

## What's in the box

- **Deviations** — `adev`, `mdev`, `hdev`, `mhdev`, `totdev`, `mtotdev`, `htotdev`, `mhtotdev`, `tdev`, `ldev`. All share one engine, accept phase or frequency data, and return point estimates plus EDF-based confidence intervals.
- **Noise identification** — lag-1 ACF for long records, B1/R(n) ratio fallback for short ones. Returns SP1065 α ∈ {-2, -1, 0, 1, 2}.
- **Kalman filter** — three-state clock model, Q-parameter grid-search optimizer, multi-step prediction/holdover, and an end-to-end pipeline (data → deviation → noise fit → KF).
- **Plots** — log–log σ(τ) with CI bands. In Julia, loaded as a package extension so `using SigmaTau` stays light.

## Install

### MATLAB
```matlab
cd matlab
addpath(genpath(pwd));
```
Everything lives under the `+sigmatau` namespace: `sigmatau.dev.adev(x, tau0)`.

### Julia
```julia
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
Then `using SigmaTau`.

## Usage

MATLAB:
```matlab
[tau, dev, ci, alpha] = sigmatau.dev.adev(phase, tau0);
result = sigmatau.kf.pipeline(phase, tau0);
```

Julia:
```julia
using SigmaTau
result = adev(phase, tau0)             # result.tau, result.dev, result.ci, result.alpha
kf     = kf_pipeline(phase, tau0)
```

Both accept `data_type = :freq` to pass frequency data instead of phase — the engine handles the conversion.

## Scripts & Examples

- `scripts/`: Production-ready command-line tools for stability analysis and Kalman filtering (Julia, MATLAB, Python).
- `examples/`: Guided examples and validation datasets (e.g., `noise_id_validation`, `mixed_noise_validation`).

## Tests

```bash
# MATLAB
cd matlab && matlab -batch "addpath(genpath('.')); run('tests/run_all.m')"

# Julia
cd julia && julia --project=. -e 'using Pkg; Pkg.test()'
```

## References

- NIST SP1065 — Riley & Howe, *Handbook of Frequency Stability Analysis*
- Greenhall & Riley, *Uncertainty of Stability Variances*, PTTI 2003
- IEEE Std 1139-2022

## Author

Ian Lapinski — ianlapinski01@gmail.com

## License

MIT
