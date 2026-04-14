# `6krb25apr/` — Stable32 cross-validation scratch directory

Drop a Stable32 phase log (e.g. a real rubidium oscillator capture named
`6krb25apr.txt`, two columns: MJD and phase-in-seconds) into this directory
and use `examples/kf_pipeline/` scripts to compute the 10 NIST deviations on
it for side-by-side comparison against Stable32's own output.

The raw phase log is **not checked in** (it's proprietary or large). Nothing
in the repo depends on it — this directory exists so the comparison has a
stable home when you do have the data.

## Layout (when populated)

```
examples/6krb25apr/
  6krb25apr.txt          # your raw phase log (gitignored via *.txt rule)
  stable32_out/          # Stable32 exports — all *.txt files gitignored
    adev.txt hdev.txt mdev.txt oadev.txt ohdev.txt total.txt time.txt
  README.md              # this file
```

## Cross-check workflow

```sh
# 1. Compute sigmatau deviations on the dataset
julia --threads=auto --project=julia \
    examples/kf_pipeline/compute_devs.jl 6krb25apr

# 2. Visual overlay vs Stable32
python3 examples/kf_pipeline/plot_devs.py 6krb25apr
```

Results land under `examples/kf_pipeline/results/6krb25apr/` (gitignored).
