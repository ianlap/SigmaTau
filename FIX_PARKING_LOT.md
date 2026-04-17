# Fix Parking Lot

Observations surfaced during fix passes that are out-of-scope for the current fix but worth tracking.

- [scripts/julia/kf_pipeline.jl migration] Rolling-origin RMS scales O((N−maturity)·max_hor). At N=50k, max_hor=40k this is ~95s per KF run; the full 406k-sample dataset would be ~40× longer. An N_STARTS cap (sample K equally-spaced origins instead of all) would trade statistical precision for wall-time.
- [scripts/python/plot_kf.py coupling] `plot_kf.py` resolves its CSV directory relative to `scripts/python/` (`HERE / "results" / DATASET / "kf"`), but `kf_pipeline.jl` writes to `scripts/julia/results/...`. The two don't meet. Pre-existing; also noted in AUDIT_02.
- [h→q formula sweep] Per user direction, `h_to_q`/`q_to_h` in `julia/src/clock_model.jl` (Wu 2023: 4π² for WPM, 1/2 for WFM, 2π² for RWFM) is canonical. `test_filter.jl:186` still uses `(2π²/3)·h[-2.0]` for expected q_rwfm — factor-of-3 disagreement that happens to fit inside the test's 1-decade tolerance. `scripts/python/plot_kf.py:54-59` MHDEV component coefficients also encode the pre-Wu legacy formula. Both should be audited against `q_to_h` in a future pass.
- [ml/dataset/real_data_fit_file2.jl] Hardcoded file1 q values at lines 219-220 were captured from a pre-Wu-2023 run of `real_data_fit.jl`. Under the current convention these will differ from a fresh run. Combined plot's file1 theoretical ADEV and the file1 row in `proposed_h_ranges.md` are stale until someone re-runs `real_data_fit.jl` and updates the hardcoded values (or wires up runtime parsing of `ml/data/real_data_fit.txt`).
