# SigmaTau — Developer Workflows

## Build & Test

```bash
# MATLAB — run from repo root
cd matlab && matlab -batch "addpath(genpath('.')); run('tests/run_all.m')"

# Julia — run from julia/ directory  
cd julia && julia --project=. -e 'using Pkg; Pkg.test()'
```

## Verification

Every refactored function must produce identical numerical output to the legacy code. Test with:
1. White PM noise (α=2): slope should be τ^(-1) for ADEV
2. White FM noise (α=0): slope should be τ^(-1/2) for ADEV
3. RWFM noise (α=-2): slope should be τ^(+1/2) for ADEV
4. Cross-validate MATLAB vs Julia (relative error < 1e-12)

## Resources

- **References**:
  - NIST SP1065: Riley & Howe, "Handbook of Frequency Stability Analysis"
  - Greenhall & Riley, "Uncertainty of Stability Variances," PTTI 2003
  - IEEE Std 1139-2022
- **Legacy Components**: Read from `matlab/legacy/` and `julia/legacy_stablab/`.
- **Known Bugs**: Verify htotdev bias direction, htotdev EDF loop indexing, and mhtotdev Neff formula against Riley (2001). totdev denominator verified 2026-04-14: SP1065 Eq. 25 uses 2τ²(N-2) for phase form (equivalently 2(M-1) for frequency form, M=N-1) — do not change.

## Maintenance Tasks
- [ ] Delete `mhdev_noID.m` — dead code.
- [ ] Merge parallelism in `mhtotdev_par.m` into `mhtotdev`.
- [ ] Move `compute_devs_from_file.m` and `compute_all_devs_from_file.m` to `examples/`.
