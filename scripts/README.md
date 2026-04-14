# SigmaTau Scripts — Stability Analysis & Kalman Filtering

This directory contains command-line tools and pipeline scripts for SigmaTau in Julia, MATLAB, and Python.

## Directory Layout

- `julia/`: Core analysis and filtering tools.
  - `basic_usage.jl`: Simple introduction to the SigmaTau Julia API.
  - `compute_all_devs.jl`: Computes all 10 NIST deviations from a phase data file.
  - `kf_pipeline.jl`: Full Kalman Filter characterization and optimization pipeline.
  - `mhdev_preview.jl`: Quick noise identification preview tool.
- `matlab/`: Equivalent MATLAB tools (using the `sigmatau` package).
  - `basic_usage.m`: Simple introduction to the SigmaTau MATLAB API.
  - `compute_all_devs.m`: MATLAB version of the all-deviation tool.
  - `kf_pipeline.m`: MATLAB version of the KF pipeline.
- `python/`: Plotting and cross-validation utilities.
  - `plot_kf.py`: 6-panel Kalman Filter diagnostic plots.
  - `plot_devs.py`: Log-log stability plots for all 10 deviations.
  - `mhdev_fit_interactive.py`: Interactive power-law noise component fitter.
  - `generate_comprehensive_report.py`: Cross-validation report generator (Stable32 vs. allantools vs. SigmaTau).

## Usage Examples

### 1. Basic Stability Analysis (Julia)
```bash
# Compute all 10 deviations for a dataset and save to [file]_devs/
julia --project=julia scripts/julia/compute_all_devs.jl reference/validation/stable32gen.DAT 1.0
```

### 2. Kalman Filter Pipeline (Julia)
```bash
# Step A: Preview noise characteristics
julia --project=julia scripts/julia/mhdev_preview.jl 6krb25apr
python3 scripts/python/plot_mhdev_preview.py 6krb25apr

# Step B: Interactive noise fit (opens matplotlib window)
python3 scripts/python/mhdev_fit_interactive.py 6krb25apr

# Step C: Run the full optimization and filtering pipeline
julia --project=julia scripts/julia/kf_pipeline.jl 6krb25apr

# Step D: Generate final diagnostic plots
python3 scripts/python/plot_kf.py 6krb25apr
```

### 3. MATLAB Equivalent
```matlab
% In MATLAB command window:
results = kf_pipeline('reference/validation/stable32gen.DAT', 1.0);
compute_all_devs('reference/validation/stable32gen.DAT', 1.0);
```

## Data Format
Scripts expect a single-column or two-column (MJD, Phase) text file. If two columns are provided, `tau0` is automatically inferred from the MJD spacing.

Outputs are generally saved to a directory named after the dataset (e.g., `results/<dataset>/`).
