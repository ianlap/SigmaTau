# `plots/` — ADEV cross-check against allantools

One-shot demo that runs `sigmatau.dev.adev` on a white-FM phase series and
overlays `allantools.oadev` on the same data. Produces a labelled log-log PNG
with slope readings from both implementations, a theoretical τ^(-1/2) reference
line, and the max relative disagreement between the two.

## Files

| File | Purpose |
|---|---|
| `generate.m` | MATLAB: synthesize N=8192 white-FM phase, compute sigmatau ADEV + CI, dump `wfm_phase.csv` and `sigmatau_adev.csv` |
| `plot_adev.py` | Python: read the two CSVs, call allantools.oadev, plot overlay, save `adev_wfm.png` |
| `sigmatau_adev.csv`, `wfm_phase.csv` | Generated inputs (kept for reference) |
| `adev_wfm.png` | Final figure (kept as a visual regression target) |

## Reproducing

```sh
# From the repo root:
matlab -batch "addpath(genpath('matlab')); run('examples/plots/generate.m')"
python3 examples/plots/plot_adev.py
```

Expected: slopes ≈ -0.5 from both implementations, max relative disagreement
< 1e-3 across all τ.

## Intent

This example is the simplest possible MATLAB + Python cross-check, useful when
introducing sigmatau to someone who already knows allantools.
