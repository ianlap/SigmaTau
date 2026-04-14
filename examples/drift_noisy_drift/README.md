# `drift_noisy_drift/` — drift vs noisy-drift behavior across Allan/Hadamard families

This example synthesizes three phase time series and compares how long-term deterministic drift and noisy drift reshape the long-τ tails of several deviations:

1. **Comparison plot 1:** ADEV and HDEV
2. **Comparison plot 2:** MDEV and MHDEV
3. **Comparison plot 3:** TDEV and LDEV

The key intent is to push τ far enough that long-term processes dominate:

- RWFM-like behavior in the base record,
- deterministic quadratic phase drift,
- and RRFM-like noisy drift (a third-integral random process) that is especially visible in the Hadamard-family trends.

## Files

| File | Purpose |
|---|---|
| `generate.jl` | Build synthetic phase records; compute ADEV/HDEV/MDEV/MHDEV/TDEV/LDEV; save CSV tables. |
| `plot_drift_comparison.py` | Generate the three requested comparison plots from the CSV output. |
| `out/*.csv` | Generated tables (written by `generate.jl`). |
| `out/comparison_plot_*.png` | Generated figures (written by `plot_drift_comparison.py`). |

## Reproduce

From the repo root:

```bash
julia --project=julia examples/drift_noisy_drift/generate.jl
python3 examples/drift_noisy_drift/plot_drift_comparison.py
```

The generator prints a compact slope summary over the longest available τ decade so you can quickly verify that long-τ behavior has moved toward drift / RRFM-like regimes.
