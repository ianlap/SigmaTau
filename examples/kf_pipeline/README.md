# `kf_pipeline/` — dataset-agnostic clock analysis

Non-interactive Julia port of the MATLAB `main_kf_pipeline_unified.m` workflow,
plus matplotlib equivalents of the legacy plots. All scripts take a dataset
basename (no extension) as their first argument; the data file is expected
to live at `reference/<dataset>.txt` (two columns: MJD, phase-in-seconds).
Outputs for a run land under `results/<dataset>/{data,devs,kf}/`.

Reusable pieces now live in the SigmaTau package:

- `SigmaTau.mhdev_fit(tau, sigma, regions; ci, weight_method)` — port of
  `matlab/legacy/kflab/mhdev_fit.m`. Fits q_wpm/q_wfm/q_rwfm/q_rrfm (and
  flicker intercepts) by successive residual subtraction over user-declared
  τ-index ranges.
- `kalman_filter`, `kf_predict`, `optimize_kf` already in the package.

## End-to-end workflow (new dataset)

From the repo root, with `<dataset>` the basename (e.g. `6k27febunsteered`):

```bash
# 1. Compute MHDEV and write a preview CSV + plot so you can eyeball regions
julia --threads=auto --project=julia examples/kf_pipeline/mhdev_preview.jl <dataset>
python3 examples/kf_pipeline/plot_mhdev_preview.py <dataset>

# 2. Interactive noise-component fit — live-updating matplotlib, writes
#    results/<dataset>/kf/mhdev_fit.csv when you exit with 0
python3 examples/kf_pipeline/mhdev_fit_interactive.py <dataset>

# 3. Run the KF, refine q via innovation NLL, compare
julia --threads=auto --project=julia examples/kf_pipeline/kf_pipeline.jl <dataset>

# 4. Generate 6-panel NLL-opt summary + MHDEV-comparison plots
python3 examples/kf_pipeline/plot_kf.py <dataset>
```

If you skip step 2, `kf_pipeline.jl` falls back to `SigmaTau.mhdev_fit` on
the `FIT_REGIONS` constant declared at the top of the script (useful for
automation, but the interactive flow usually gives better initial q values).

## All-deviation comparison (orthogonal tool)

`compute_devs.jl` / `plot_devs.py` are an independent utility that runs all
10 NIST deviations on the first 50_000 points of a dataset and writes the
trimmed subset next to the results — useful for cross-checking against
Stable32. Same CLI convention:

```bash
julia --threads=auto --project=julia examples/kf_pipeline/compute_devs.jl <dataset>
python3 examples/kf_pipeline/plot_devs.py <dataset>
```

## Output layout

```
examples/kf_pipeline/
├── compute_devs.jl              # run all 10 deviations on a subset
├── plot_devs.py
├── mhdev_preview.jl             # MHDEV on full dataset → preview CSV
├── plot_mhdev_preview.py
├── mhdev_fit_interactive.py     # live fit, writes mhdev_fit.csv
├── kf_pipeline.jl               # stages 1–7 of the KF pipeline
├── plot_kf.py                   # 6-panel NLL-opt + MHDEV-comparison figures
└── results/
    └── <dataset>/
        ├── data/                # subset dump for Stable32 (compute_devs.jl)
        ├── devs/                # 10 deviation CSVs + PNGs (compute_devs.jl)
        └── kf/
            ├── mhdev_preview.csv / .png
            ├── mhdev_fit.csv    (from interactive fit)
            ├── mhdev_fit.png
            ├── kf_pipeline_summary.csv
            ├── kf_pipeline_prediction.csv
            ├── kf_nll_surface.csv
            ├── kf_optimization.png
            └── kf_mhdev_comparison.png
```
