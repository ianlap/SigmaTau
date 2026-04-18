# SigmaTau

Frequency stability analysis and Kalman clock steering in MATLAB and Julia.

SigmaTau is a twin-language package for time-and-frequency work: compute the ten NIST SP1065 deviations, identify power-law noise, and run Kalman filters for clock steering and holdover. MATLAB and Julia mirror each other for the deviation and noise-ID engines — the ten deviation point estimates cross-validate within `REL_TOL = 2e-10` on canonical noise types. The Kalman filter is Julia-first; the MATLAB `+kf/` package mirrors the legacy pre-refactor interface and is maintained as a numerical reference target.

## What's in the box

- **Deviations** — `adev`, `mdev`, `hdev`, `mhdev`, `totdev`, `mtotdev`, `htotdev`, `mhtotdev`, `tdev`, `ldev`. All share one engine, accept phase or frequency data, and return point estimates plus EDF-based confidence intervals.
- **Noise identification** — lag-1 ACF for long records, B1/R(n) ratio fallback for short ones. Returns SP1065 α ∈ {-2, -1, 0, 1, 2}.
- **Kalman filter** — 2-, 3-, and diurnal 5-state clock models (Julia; MATLAB is 3-state); Nelder-Mead optimizer on Zucca–Tavella diffusion parameters; multi-step holdover prediction; end-to-end CLI pipeline (data → deviation → noise fit → KF).
- **Plots** — log–log σ(τ) with CI bands. `using SigmaTau` currently pulls `Plots.jl` as a direct dependency; migration to a package extension is tracked in `TODO.md`.

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
% End-to-end KF pipeline: see scripts/matlab/kf_pipeline.m
```

Julia:
```julia
using SigmaTau
result = adev(phase, tau0)             # result.tau, result.deviation, result.ci, result.alpha

# Function-level KF: see examples/kf_holdover.jl
# CLI end-to-end pipeline: julia --project=julia scripts/julia/kf_pipeline.jl <dataset>
```

Both accept `data_type = :freq` to pass frequency data instead of phase — the engine handles the conversion.

## Scripts & Examples

- `scripts/`: Production-ready command-line tools for stability analysis and Kalman filtering (Julia, MATLAB, Python).
- `examples/`: Guided examples and validation datasets (e.g., `noise_id_validation`, `mixed_noise_validation`).

## Tool Handbook (Help Section)

For full command syntax, options, and workflow recipes, use the handbook:

- [Handbook index](docs/handbook/index.md)
- [CLI command reference (`sigmatau`)](docs/handbook/cli.md)
- [Julia scripts reference](docs/handbook/julia_scripts.md)
- [Python tools reference](docs/handbook/python_tools.md)
- [MATLAB scripts reference](docs/handbook/matlab_scripts.md)
- [Workflow recipes](docs/handbook/workflows.md)

### Quick reference

| Tool | Purpose | Minimal invocation |
| :--- | :--- | :--- |
| `bin/sigmatau` | Interactive CLI for loading data, computing deviations, plotting, and export | `bin/sigmatau` |
| `scripts/julia/compute_all_devs.jl` | Batch compute all 10 deviations | `julia --project=julia scripts/julia/compute_all_devs.jl <file> <tau0>` |
| `scripts/julia/mhdev_preview.jl` | Fast noise-character preview for KF prep | `julia --project=julia scripts/julia/mhdev_preview.jl <dataset>` |
| `scripts/julia/kf_pipeline.jl` | Full noise-fit + KF optimization + filtering pipeline | `julia --project=julia scripts/julia/kf_pipeline.jl <dataset>` |
| `scripts/python/plot_devs.py` | Multi-deviation log-log plot generation | `python3 scripts/python/plot_devs.py <dataset>` |
| `scripts/python/plot_kf.py` | KF diagnostics plotting | `python3 scripts/python/plot_kf.py <dataset>` |
| `scripts/python/mhdev_fit_interactive.py` | Interactive power-law fit UI | `python3 scripts/python/mhdev_fit_interactive.py <dataset>` |
| `scripts/python/generate_comprehensive_report.py` | Stable32/allantools/SigmaTau cross-validation report | `python3 scripts/python/generate_comprehensive_report.py` |
| `scripts/matlab/compute_all_devs.m` | MATLAB batch all-deviation run | `compute_all_devs('<file>', tau0)` |
| `scripts/matlab/kf_pipeline.m` | MATLAB KF pipeline | `kf_pipeline('<file>', tau0)` |

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
