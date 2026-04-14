# `noise_id_validation/` — pure-α phase series for noise-ID cross-checks

Five synthetic phase records, one per α ∈ {+2, +1, 0, −1, −2}, generated via
Kasdin–Walter FFT synthesis. Lets you spot-check `SigmaTau.noise_id` (and the
matching MATLAB `sigmatau.noise.identify`) returns the expected α at each τ.

The raw `.csv` files are **gitignored** — regenerate locally:

```sh
julia --project=examples/noise_id_validation \
      examples/noise_id_validation/generate.jl
```

First run will `Pkg.instantiate()` the example's local env (FFTW + SigmaTau
path-dep) — a few seconds. Later runs reuse the cache.

Each file is space-separated `MJD phase(sec)` with N = 65536 samples at τ0 = 1 s.

## Files

| File | α | Noise type |
|---|---|---|
| `alpha+2_whpm.csv` | +2 | White phase modulation |
| `alpha+1_flpm.csv` | +1 | Flicker phase modulation |
| `alpha+0_whfm.csv` | 0 | White frequency modulation |
| `alpha-1_flfm.csv` | −1 | Flicker frequency modulation |
| `alpha-2_rwfm.csv` | −2 | Random-walk frequency modulation |

## Expected output

For a τ in the interior (m=64 in the generator's default sweep) `noise_id`
should return α̂ within ≈ 0.3 of the true α for clean power-law cases (WPM,
WFM, RWFM) and within ≈ 0.6 for the flicker cases where B1/R(n) is less sharp.

## Reference

N. J. Kasdin & T. Walter, "Discrete simulation of power law noise",
*Proc. IEEE Frequency Control Symposium*, pp. 274–283, 1992.
