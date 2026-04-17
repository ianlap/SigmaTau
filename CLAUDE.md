# SigmaTau — Developer Workflows

## Build & Test

```bash
# MATLAB — run from repo root
cd matlab && matlab -batch "addpath(genpath('.')); run('tests/run_all.m')"

# Julia — run from julia/ directory
cd julia && julia --project=. -e 'using Pkg; Pkg.test()'
```

## Verification

Refactored code must preserve numerical behavior. Two independent checks cover this:

1. **Noise-slope sanity.** The usual SP1065 asymptotics must hold on synthetic power-law noise:
   - White PM noise (α=2): ADEV slope ≈ τ⁻¹
   - White FM noise (α=0): ADEV slope ≈ τ⁻½
   - RWFM noise (α=−2): ADEV slope ≈ τ^(+½)

   Covered by `matlab/tests/test_noise_slopes.m` and `julia/test/test_allan_family.jl`.

2. **MATLAB ↔ Julia cross-validation (stability deviations only).** Deviation point estimates agree within `REL_TOL = 2e-10`. Scope is the 10 deviations, 3 noise types — **not** EDF, CI, or KF outputs. The test silently skips (warning-only) when the Julia reference file `crossval_results.txt` is missing; regenerate with `julia --project=julia julia/scripts/gen_crossval_data.jl`. There is no equivalent KF cross-validation across languages yet.

See `matlab/tests/test_crossval_julia.m:47` for the tolerance and `:10-14` for the skip-on-missing behavior.

## Resources

- **References**
  - NIST SP1065 (Riley & Howe, *Handbook of Frequency Stability Analysis*) — `docs/papers/reference/sp1065.pdf`
  - Greenhall & Riley, "Uncertainty of Stability Variances," PTTI 2003
  - IEEE Std 1139-2022
  - Matsakis & Banerjee, *Timekeeping* (2023) — `docs/papers/reference/2023_banerjee_matsakis_timekeeping_book.pdf` (conceptual / derivation tiebreaker)
  - Wu, "KF Performance for an LTI Atomic Clock" IEEE TAES (2023) — `docs/papers/state_estimation/2023_wu_kf_performance_lti_atomic_clock_ieee_taes.pdf` (canonical h↔q convention)

## Institutional Memory

### Totdev denominator (preserve verbatim — verified against SP1065)

totdev denominator verified 2026-04-14: SP1065 Eq. 25 uses 2τ²(N-2) for phase form (equivalently 2(M-1) for frequency form, M=N-1) — do not change.
_[Verified 2026-04-14 — check: SP1065 Eq 25 reviewed against `julia/src/deviations/total.jl` and `matlab/+sigmatau/+dev/totdev.m` on 2026-04-14; denominator matches `2τ²(N-2)` in phase form. This note is institutional memory — body preserved verbatim from prior CLAUDE.md.]_

### Open verification items

Verify against Riley (2001) when time allows: htotdev bias direction, htotdev EDF loop indexing, and mhtotdev N_eff formula. Current implementations are consistent with in-tree tests but have not been cross-checked against the original source.

### Int64 overflow in MDEV/MHDEV denominators (fixed — don't regress)

The denominators `N_e · 2 · m⁴` and `N_e · 6 · m⁴` silently wrapped Int64 at `m ≳ 10⁴` — at `m=11461, N=131072` the true value is `3.35×10²¹` but wrapped to `~10¹⁸`, producing MDEV ~200× too large and MHDEV negative (clamped to 0). The fix (commit `ead11ea`) promotes `2 → 2.0` and `m → Float64(m)` in `julia/src/{deviations/allan.jl, deviations/hadamard.jl, noise.jl, deviations/total.jl}`. Do not reintroduce integer arithmetic in those denominators. See `ml/STATE.md:170-181` for the full narrative.

## Maintenance Queue

The maintenance queue is no longer tracked in this file. Engineering work lives in `TODO.md` (prioritized by impact / difficulty), and in-flight-fix debt lives in `FIX_PARKING_LOT.md`. The three items previously listed here (`mhdev_noID.m`, `mhtotdev_par.m`, `compute_devs_from_file.m` / `compute_all_devs_from_file.m`) have all been removed from the repo — verified absent via glob 2026-04-16.
