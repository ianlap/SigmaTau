# Mixed-Noise Validation Dataset

A 2^15-point phase-data file that exercises SigmaTau's noise-ID dispatch and
bias-correction branches in a single recording, built for cross-validation
against Stable32 and allantools.

## Composition

Three independent pure-noise realisations summed with calibrated amplitudes
to put the ADEV crossovers near specific τ values:

| Component | α  | Slope of ADEV | Seed (Xoshiro) |
|-----------|----|--------------|----------------|
| WPM       | +2 | τ^(-1)       | 1001           |
| WFM       |  0 | τ^(-1/2)     | 1002           |
| RWFM      | -2 | τ^(+1/2)     | 1003           |

`x_mixed = c_WPM · x_WPM + x_WFM + c_RWFM · x_RWFM`

`c_WFM = 1`; `c_WPM` and `c_RWFM` are solved from the closed-form slope
equations so that the component ADEVs cross at

- τ ≈ 32   (WPM → WFM)
- τ ≈ 512  (WFM → RWFM)

See `generate.jl` for the full derivation and the achieved coefficients.

## Expected behaviour

- **ADEV slope:** −1 for short τ, −1/2 for mid τ, +1/2 for long τ.
- **Noise ID:** +2 in the short-τ band, ~0 in the mid-τ band, ~−2 in the
  long-τ band; the transition bands (around each crossover) are where the
  dispatch logic gets exercised.
- **Bias correction:** `totdev`, `mtotdev`, `htotdev`, `mhtotdev` each apply
  α-dependent bias corrections — the mixed record hits several branches in
  one file.

## Dataset specification

| Field         | Value                        |
|---------------|------------------------------|
| N             | 32_768 (2^15)                |
| tau0          | 1.0                          |
| Data type     | Phase                        |
| File format   | Plain single-column text     |
| Float format  | `%.17g` (IEEE 754 round-trip)|
| Line count    | 32_768                       |

## Files

```
mixed_noise.txt          — 2^15 single-column phase data
generate.jl              — reproducible Julia generator (seeded)
reference/
  adev.csv mdev.csv hdev.csv mhdev.csv
  totdev.csv mtotdev.csv htotdev.csv mhtotdev.csv
  tdev.csv ldev.csv
    columns: m, tau, sigma, edf, ci_lo, ci_hi, alpha_id
  noise_id.csv
    columns: m, tau, alpha_id
```

The reference CSVs are SigmaTau's own output at the octave m-list
`m = 1, 2, 4, …, ⌊N/4⌋`, included so external tools can diff directly without
re-running the generator.

## Loading into external tools

### Stable32

1. File → Open → select `mixed_noise.txt`.
2. Data Type: **Phase**.
3. τ₀: **1.0**.
4. Run ADEV / MDEV / HDEV / TOTDEV etc. and compare against `reference/*.csv`.

### allantools (Python)

```python
import numpy as np
import allantools as at

x = np.loadtxt("mixed_noise.txt")
taus, adev, adev_err, _ = at.oadev(x, rate=1.0, data_type="phase", taus="octave")
```

Use `at.mdev`, `at.hdev`, `at.totdev`, … for the other deviations; pass the
same octave τ list to match SigmaTau's reference rows.

### SigmaTau (Julia) — round-trip check

```julia
using SigmaTau
x  = parse.(Float64, readlines("mixed_noise.txt"))
r  = adev(x, 1.0)        # returns a DeviationResult
# r.tau, r.deviation, r.edf, r.ci, r.alpha
```

## Regenerating

From the `julia/` directory:

```bash
julia --project=. ../examples/mixed_noise_validation/generate.jl
```

Run time on a developer laptop is ~35 s. Output is byte-identical across
runs — the three component realisations use explicit `Xoshiro(seed)` RNGs
so reproducibility does not depend on Julia's default RNG.

## Design notes

Full rationale — motivation, crossover math, algorithm, risks, success
criteria — lives in
`docs/archive/2026-04-13-mixed-noise-validation-dataset-design.md`.
