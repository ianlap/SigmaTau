# ML Pipeline Design — PH 551 Final Project

**Date:** 2026-04-14
**Author:** Ian Lapinski (with Claude)
**Supersedes / refines:** `ml/ml_pipeline_spec_v2.md`
**Deadline:** 2026-04-24

This document resolves the open architectural questions in `ml_pipeline_spec_v2.md` and specifies the implementation to follow.

---

## 1. Scientific Question (refined)

**Original (spec v2):** "Does MHDEV provide better KF-parameter predictions than MDEV?"

**Refined:** "Which stability statistics — ADEV, MDEV, HDEV, or MHDEV — carry the most Kalman-filter-relevant information about the underlying clock noise, and at which averaging times τ?"

The refinement follows from putting all four statistics into a single feature matrix and letting tree feature-importance answer the question directly. The MDEV-vs-MHDEV comparison is preserved as a secondary analysis — a slice of the importance bar chart, not a separate model.

---

## 2. Architectural Decisions

### 2.1 Three-dimensional NLL labels (q_wpm optimized)

The existing `optimize_kf` in [julia/src/optimize.jl](../../../julia/src/optimize.jl) fixes `q_wpm = R`. The spec requires ML to predict `q_wpm`, so labels must come from a 3-D NLL optimization that walks log10(R) alongside log10(q_wfm), log10(q_rwfm).

**Implementation:**
- Add `optimize_qwpm::Bool = false` to `OptimizeConfig`. Default preserves today's behavior.
- When true, Nelder-Mead optimizes `[log10(q_wpm), log10(q_wfm), log10(q_rwfm)]`.
- Bounds unchanged: `log10 q ∈ [-40, -10]` applied uniformly.
- New convenience wrapper `optimize_kf_nll(phase, tau0; h_init=nothing)` hides the config plumbing and supplies analytical warm start when `h_init` is given.

### 2.2 Physics-preserving noise generator (option B)

Port MATLAB Kasdin generator to Julia with **absolute h_α amplitude** — skip the unit-std normalization. One bin's complex amplitude in the one-sided spectrum is

`A_k = sqrt(h_α / 2 · Δf · f_k^α) · exp(i·φ_k)`

where `Δf = 1 / (N·τ₀)` and `φ_k ~ U[0, 2π)`. IFFT of the mirrored symmetric spectrum gives frequency data `y(t)` with the correct PSD in expectation; `x(t) = cumsum(y) · τ₀` gives phase in seconds.

Supports fractional α (needed for FPM at α=+1 and FFM at α=−1) — no separate kernel.

Composite noise generation: call the kernel once per (α, h_α) pair in the sample's h-coefficient dict, sum the phase series. Single seed per sample drives `Random.Xoshiro(42 + sample_idx)`; each α draws from the same RNG sequentially for reproducibility.

**Location:** `julia/src/noise_gen.jl` (new file), exports `generate_composite_noise(h_coeffs::Dict, N, tau0; seed)`.

### 2.3 Unified feature matrix (196 features)

80 raw + 76 slopes + 40 ratios = **196 features** per sample.

| Block | Count | Description |
|---|---|---|
| Raw log₁₀ σ | 4 × 20 = 80 | ADEV, MDEV, HDEV, MHDEV at each τ |
| Adjacent-τ log-log slopes | 4 × 19 = 76 | `(log σ[k+1] − log σ[k]) / (log τ[k+1] − log τ[k])` per stat |
| Variance ratios | 2 × 20 = 40 | MVAR/AVAR and MHVAR/HVAR at each τ |

NaN handling: column median imputation (applied in Python at load time). Tree models are robust to this; the alternative of per-feature missingness indicators is unnecessary given the shared τ grid guarantees ≥ N/10 effective samples at every τ.

### 2.4 Shared τ grid

**Twenty log-spaced τ in [1s, 13107s]**, with τ_max = N·τ₀ / 10 (safety factor 10 against MHDEV's min_factor = 4).

`m_list = sort(unique(round.(Int, exp10.(range(0, log10(13107), length=20)))))`

With N = 131,072 the 20 values round to 20 unique integers — no collision fill-in needed. Stored as a module constant `CANONICAL_TAU_GRID` shared between generator, feature extractor, and ML loader.

### 2.5 FPM/FFM handling: absorbed into optimizer

Per spec. FPM always generated (when present, ~30% probability) as α=+1. FFM always generated as α=−1. The KF has no state for either — the 3-D NLL optimizer absorbs them as *effective* q values:

- FPM inflates `q_wpm` label.
- FFM inflates `q_wfm` label.

This is the correct operational target — an operator's best q_wpm already absorbs flicker they cannot separately model. Dataset saves both `h_coeffs` (5 values) and `q_labels` (3 values) so the bias relationship can be analyzed.

### 2.6 Dataset storage

`ml/data/dataset_v1.npz` (new directory), compressed NumPy archive.

| Key | Shape | dtype | Description |
|---|---|---|---|
| `X` | (10000, 196) | float32 | Feature matrix |
| `y` | (10000, 3) | float64 | log₁₀(q_wpm, q_wfm, q_rwfm) labels |
| `h_coeffs` | (10000, 5) | float64 | log₁₀(h₊₂, h₊₁, h₀, h₋₁, h₋₂) |
| `fpm_present` | (10000,) | bool | FPM inclusion mask |
| `nll_values` | (10000,) | float64 | Final NLL at optimum |
| `converged` | (10000,) | bool | Nelder-Mead convergence flag |
| `feature_names` | (196,) | str | Feature column labels |
| `taus` | (20,) | float64 | Canonical τ grid |
| `metadata` | dict | — | N, τ₀, seed, sigmatau git sha, timestamp |

**Not saved:** raw phase time series (~10.5 GB). Phase is exactly reproducible from `(h_coeffs, seed=42+i)` via the generator.

### 2.7 Parallelization and checkpointing

- **Threads** (12) via `Threads.@threads` over sample index.
- **Checkpoint every 500 samples** to `ml/data/dataset_v1.checkpoint.npz`; on startup, if checkpoint exists, resume from the highest completed index.
- Individual-sample failures (NLL non-convergence, numerical issues) are caught; the row is marked `converged=false` and training filters those out.

### 2.8 KF NLL performance optimization

Scoped to a new kernel `_kf_nll_static` in [julia/src/optimize.jl](../../../julia/src/optimize.jl), dispatched when `nstates == 3`. Existing `_kf_nll` and `kalman_filter` stay untouched.

Changes:
1. `StaticArrays.SMatrix{3,3}` / `SVector{3}` for Φ, Q, P, x — stack-allocated, zero GC in hot loop.
2. Scalar-H shortcut: `S = P[1,1] + R`, `ν = z - x[1]`, `K = P[:,1] / S` — no matrix multiplies.
3. LS warm start hoisted out of `_kf_nll` into `optimize_kf` — one solve per sample, not per NLL eval.
4. Analytical warm start from h_α when `h_init` is provided.

Expected per-sample time: ~50 ms for NLL optimization + ~100 ms noise gen + ~50 ms deviations → ~200 ms/sample. 10,000 samples / 12 threads → **< 5 minutes wall clock**. Big margin vs the 14-hour budget.

### 2.9 Real-data validation (Section 7)

**Data:** Both records are **unsteered** GMR6000 Rb phase traces, τ₀ = 1 s (verified from MJD column spacing).

Both records use a **5 MHz reference** (cycle period 200 ns).

- `reference/raw/6k27febunsteered.txt` — 3M samples ≈ 34.7 days. Phase column believed to be in **clock cycles** → seconds via `×2e−7`.
- `reference/raw/6krb25apr.txt` — 407k samples ≈ 4.7 days. Phase column believed to be in **nanoseconds**, quantized to 20 ns steps (counter resolution; 1/10 of the 200 ns cycle, interpolated). Leading zero-run is the noise-floor dead-band, not steering.

**Empirical unit detection at load time.** Before committing to a conversion, the loader computes raw ADEV at τ=1s and matches against the GMR6000-expected unsteered Rb stability (σ_y(1s) ≈ 1e−10 to 1e−11, based on user's prior Stable32 analysis of these records).

| Raw ADEV(1s) | Inferred unit | Conversion |
|---|---|---|
| ≈ 1e−10 | already in seconds | none |
| ≈ 1e−1  | nanoseconds | ×1e−9 |
| ≈ 5e−4  | cycles @ 5 MHz | ×2e−7 |

The loader logs the inferred unit and the conversion chosen; any mismatch vs the expected ranges is surfaced as a loud warning before proceeding.

Both conversions happen in the Python loader (`ml/src/real_data.py`); raw files are not touched on disk.

**Protocol:**
1. Feb record: extract ~22 non-overlapping 131,072-sample windows.
2. Apr record: extract ~3 non-overlapping windows (quantization may damage short-τ ADEV; short-τ features will be dominated by the 20 ns step floor — expect high q_wpm prediction driven by quantization-as-WPM).
3. Per window: compute 196-feature vector → trained model prediction → compare to `optimize_kf_nll` run on the same window.
4. Headline plot: overlay predicted analytical σ(τ) from predicted q values on the measured ADEV. One panel per window; 4–6 representative windows in the notebook.
5. Quantization commentary: show side-by-side Feb (unquantized) and Apr (20 ns quantized) predictions — the model's handling of quantization noise as white PM is itself a result.

---

## 3. Code Layout

### Julia (sigmatau additions)
```
julia/src/
├── noise_gen.jl      [new]   generate_composite_noise
├── optimize.jl       [edit]  _kf_nll_static, optimize_qwpm flag, optimize_kf_nll wrapper
└── ml_features.jl    [new]   CANONICAL_TAU_GRID, compute_feature_vector
```

### Julia (dataset driver)
```
ml/dataset/
└── generate_dataset.jl   [new]  main driver, Threads, checkpointing, npz write
```

### Python (ML notebook + utilities)
```
ml/
├── requirements.txt
├── data/                       (ignored by git; dataset_v1.npz lives here)
├── notebook.ipynb               deliverable: EDA + training + evaluation + plots
├── src/
│   ├── loader.py               load dataset, handle NaN, train/test split
│   ├── models.py               RF and XGBoost wrappers
│   ├── evaluation.py           RMSE, R², MAE, UQ, Wilcoxon
│   └── real_data.py            GMR6000 loader, unit detection, window extraction
└── figures/                     notebook exports here
```

### Output structure in dataset_v1.npz
See §2.6.

---

## 4. ML Pipeline (Python)

- **Models:** Random Forest (`sklearn.ensemble.RandomForestRegressor`, multi-output native) and XGBoost (`xgboost.XGBRegressor` wrapped in `MultiOutputRegressor`). Spec v2's GBR is replaced by XGBoost — stronger baseline, feature importance API identical, actively maintained.
- **Hyperparameter tuning:** `GridSearchCV` 5-fold, scoring `neg_mean_squared_error`. Grids from spec v2 §5.5 adapted:
  - RF: n_estimators ∈ {200, 500, 1000}, max_depth ∈ {None, 20, 30}, min_samples_leaf ∈ {3, 5, 10}, max_features ∈ {sqrt, 0.5}.
  - XGB: n_estimators ∈ {200, 500}, learning_rate ∈ {0.01, 0.05, 0.1}, max_depth ∈ {4, 6, 8}, subsample ∈ {0.8, 1.0}.
- **Train/test split:** 80/20, stratified by `fpm_present`, seed 42.
- **Metrics:** per-target RMSE/R²/MAE in log₁₀ space; overall multi-output RMSE.
- **Uncertainty:** RF via `forestci` infinitesimal jackknife; XGB via quantile regression at α=0.05, α=0.95. Empirical coverage reported.
- **Secondary analysis:** MDEV-vs-MHDEV slice = sum of feature importance over MDEV columns vs MHDEV columns; paired Wilcoxon on per-sample absolute errors (across 5 CV folds).

---

## 5. Plots (eight required by rubric §5.7)

Unchanged from spec v2 except:
- Plot #4 "aggregated importance by statistic" now has **four bars** (ADEV, MDEV, HDEV, MHDEV) from a single model fit, not a paired comparison.
- Plot #5 "aggregated importance by τ region" splits τ ∈ {<10s, 10–1000s, >1000s}.
- Add derived-feature importance breakdown: raw vs slopes vs ratios (simple stacked bar).

---

## 6. Rubric Alignment (100 pts + 5 bonus)

| Category (pts) | Plan |
|---|---|
| Data Exploration & Visualization (15) | Feature/target histograms, example σ(τ) curves with all 4 stats, correlation heatmap, NaN analysis, FPM slice |
| Problem Definition & Stats (15) | Refined research question, naive baseline, feature-target correlations, statistical baselines |
| Methodology & Model Choice (15) | RF + XGBoost justified; tree invariance to log features; multi-output strategy; derived features motivated |
| Hyperparameter Tuning (15) | 5-fold GridSearchCV with explicit grids; per-parameter justification |
| Evaluation & Metrics, UQ (15) | RMSE/R²/MAE per target, forestci CIs, quantile intervals, empirical coverage, Wilcoxon comparison |
| Result Visualization (15) | 8 plots (§5 above); real-data overlay as Section 7 |
| Submission (10) | Apr 24 |
| Bonus presentation (5) | 15 min talk Apr 24 |

---

## 7. Timeline (refined)

| Date | Milestone |
|---|---|
| **2026-04-14 (today)** | Finalize design (this doc), begin Julia implementation |
| Apr 15 | `noise_gen.jl` + `_kf_nll_static` + `ml_features.jl` + unit tests |
| Apr 15 (PM) | Kick off 10k-sample dataset generation (overnight, < 1 hour expected) |
| Apr 16 | Python loader, RF + XGB training pipeline, GridSearchCV |
| Apr 17 | EDA notebook, all 8 plots, uncertainty quantification |
| Apr 18 | Real-data validation on GMR6000 Feb record |
| Apr 19–22 | Notebook polish, writeup, reruns with any tuning lessons |
| Apr 23 | Slides (15 min) |
| **Apr 24** | **Submission + presentation** |

---

## 8. Risks and Open Items

| Risk | Mitigation |
|---|---|
| GMR6000 unit ambiguity not resolvable empirically | Ask user; worst case, document both candidate unit choices and run validation both ways |
| 3-D NLL local minima | Analytical h_α warm start + shared-memory restart from best of 3 perturbed inits for the ~1% worst samples |
| XGBoost install/version drift | Pin in `requirements.txt`; fall back to sklearn GBR if CI ever breaks |
| Runtime budget blown by unexpectedly slow FFT on N=2¹⁷ | Julia FFTW is threaded; worst case drop to N=2¹⁶ (still >17h record, still observable drift) |

---

## 9. Non-goals

- Deep learning models (trees are appropriate; DL has no rubric benefit on 10k tabular samples).
- Online learning / sequential updates.
- Alternative KF state structures (IRWFM etc.) — future work.
- Steering-aware noise models (the Apr record is illustrative only).

---

## 10. Appendix — Analytical h_α → q warm start

Per spec v2 §3.3, with f_h = 1 / (2τ₀):

```
q_wpm  ≈ h₊₂ · f_h / (2π²) = h₊₂ / (4π² τ₀)
q_wfm  ≈ h₀ / 2
q_rwfm ≈ (2π² / 3) · h₋₂
```

These come from equating the continuous-time KF process-noise PSD to the two-sided Sy(f) integrated over the KF observation bandwidth. FFM and FPM have no analytical warm start — their contribution falls into q_wfm and q_wpm respectively during Nelder-Mead refinement.
