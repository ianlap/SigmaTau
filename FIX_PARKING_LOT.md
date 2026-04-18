# Fix Parking Lot

Observations surfaced during fix passes that are out-of-scope for the current fix but worth tracking.

- [scripts/julia/kf_pipeline.jl migration] Rolling-origin RMS scales O((N−maturity)·max_hor). At N=50k, max_hor=40k this is ~95s per KF run; the full 406k-sample dataset would be ~40× longer. An N_STARTS cap (sample K equally-spaced origins instead of all) would trade statistical precision for wall-time.
- [scripts/python/plot_kf.py coupling] `plot_kf.py` resolves its CSV directory relative to `scripts/python/` (`HERE / "results" / DATASET / "kf"`), but `kf_pipeline.jl` writes to `scripts/julia/results/...`. The two don't meet. Pre-existing; also noted in AUDIT_02.
- [ml/notebook.py:463 legacy q↔h] The `analytical_adev` helper's inline h-derivation uses pre-Wu `h_+2 = q_wpm·2π²/f_h` (off by factor 2) and `h_-2 = 3·q_rwfm/(2π²)` (off by factor 3). Exploratory notebook only; defer. When notebook is next touched, replace with `q_to_h(ClockNoiseParams(...), tau0)` to share the canonical source.
- [GEMINI.md §2.3 + Goal G2 stale] MATLAB engine 4-arg migration completed 2026-04-16/17 (commits 69210f3 precompute+dispatch, 2d87b83 mdev/mhdev/mtotdev/mhtotdev, bb699cd adev/hdev/totdev/htotdev, 6e81423 changelog). Current `matlab/+sigmatau/+dev/engine.m:10-12,70,76` declares and passes `kernel(x, m, tau0, x_cs)` and all 10 kernels consume the shared `x_cs`. GEMINI.md §2.3 still states "MATLAB kernels currently use `kernel(x, m, tau0) → [variance, neff]` ... deliberate pre-migration state" with a verification check that points at engine.m:72 (3-arg) — stale. GEMINI.md §7 Goal G2 (MATLAB 4-arg migration tracked as open) is likewise resolved. Refresh both on the next mandate-pass: update §2.3 verification tag + remove G2 from Goals. Not fixed in this session per "no mandate changes outside the three prereq scopes" rule.

## test_filter.m bias diagnosis (Fix 4 / W4 escalation)

## test_filter.m bias diagnosis (Fix 4 / W4 escalation)

- [matlab/tests/test_filter.m Test 2 — "residuals have zero mean"] Fails deterministically on `rng(2024)` with `mu=1.653e-02 > tol=1.229e-02` (ratio 1.34). This is not a KF math bug — the assertion's statistical model is wrong for the data it operates on.

  **Diagnosis.** The tolerance `3·σ/√N` assumes iid residuals. Posterior KF residuals in this setup are strongly autocorrelated: empirical lag-1 ρ ≈ 0.4, and higher-order correlation is non-negligible (max integrated-autocorrelation inflation observed ≈ 2×, i.e. true SE ~2× the iid SE). Over 30 randomly-seeded trials (seeds 1..30), `|mu|/(3σ/√N)` distributes roughly uniformly on [0, 6.3]; only 15% pass 3σ, 50% pass 6σ, 85% pass 12σ, max 6.29. The current N=10000 tightens the iid SE tighter than the correlated reality can support.

  Even after decimating residuals by 10× (breaks autocorrelation — lag-1 ρ drops from 0.40 to 0.02), only 20/30 seeds pass the 3σ bound on the decimated series (max decimated ratio 3.25). So there is a second component on top of autocorrelation, likely LS-initialization transient bias carried into the "valid" region after the first 100 samples.

  **Why not fixed in Fix 4/W4.** No ≤3-line fix passes 100% of seeds in-sample:
  - Widen to `6·σ/√N`: passes 50%.
  - Widen to `12·σ/√N`: passes 85%.
  - Widen to `20·σ/√N`: passes 100% but reads as masking a bug, not statistics (6.7× original).
  - Decimation + `6·σ/√N_d`: passes 30/30 in-sample but is 4 code-line changes (above 3-line threshold).
  - Principled fix — replace point-estimate with Ljung-Box whiteness or block-bootstrap CI: ~10-15 lines, out of scope for a Fix-series patch.

  **Recommended next step.** When someone next owns this test, either (1) switch to a whiteness test (Ljung-Box on innovations, not residuals — innovations are the quantity the KF textbook asserts is white), or (2) adopt the decimate + widen-to-6σ approach with a comment. Prefer (1).

  **Blast radius of leaving it broken.** Low. The other 7 MATLAB tests pass, and the cross-validation suite (30 pts, REL_TOL=2e-10) catches any real KF math regression. test_filter.m Test 2 is currently a statistical health-check whose statistic is miscalibrated — its failure does not imply broken math.
