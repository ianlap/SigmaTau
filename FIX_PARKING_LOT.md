# Fix Parking Lot

Observations surfaced during fix passes that are out-of-scope for the current fix but worth tracking.

- [scripts/julia/kf_pipeline.jl migration] Rolling-origin RMS scales O((N−maturity)·max_hor). At N=50k, max_hor=40k this is ~95s per KF run; the full 406k-sample dataset would be ~40× longer. An N_STARTS cap (sample K equally-spaced origins instead of all) would trade statistical precision for wall-time.
- [scripts/python/plot_kf.py coupling] `plot_kf.py` resolves its CSV directory relative to `scripts/python/` (`HERE / "results" / DATASET / "kf"`), but `kf_pipeline.jl` writes to `scripts/julia/results/...`. The two don't meet. Pre-existing; also noted in AUDIT_02.
- [ml/notebook.py:463 legacy q↔h] The `analytical_adev` helper's inline h-derivation uses pre-Wu `h_+2 = q_wpm·2π²/f_h` (off by factor 2) and `h_-2 = 3·q_rwfm/(2π²)` (off by factor 3). Exploratory notebook only; defer. When notebook is next touched, replace with `q_to_h(ClockNoiseParams(...), tau0)` to share the canonical source.
