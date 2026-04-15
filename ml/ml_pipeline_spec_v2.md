# ML Pipeline Specification for sigmatau

**Author:** Ian Lapinski
**Date:** April 2026
**Context:** PH 551 Machine Learning final project, University of Alabama
**Deadline:** April 24, 2026

---

## 1. Problem Statement

Given a measured frequency stability curve σ(τ), predict the Kalman filter noise parameters (q_wpm, q_wfm, q_rwfm) that minimize the negative log-likelihood when steering a clock described by that stability curve. This replaces manual log-log slope fitting with a data-driven approach and directly outputs the values the KF needs — no intermediate h_α conversion step.

### Why this matters

Clock steering via Kalman filter requires three noise parameters:

- **q_wpm** — measurement noise (enters as scalar R)
- **q_wfm** — white frequency modulation (drives Q[1,1])
- **q_rwfm** — random walk frequency modulation (drives Q[2,2], Q[1,2])

Currently these are set by manual inspection of stability plots or offline NLL optimization. An ML model that maps σ(τ) → (q_wpm, q_wfm, q_rwfm) enables real-time autonomous noise characterization.

---

## 2. Physical Scenario

The dataset simulates **10,000 independent measurement records of the same clock type** — a GPS-disciplined rubidium oscillator comparable to the Masterclock GMR6000 with the Rb option (GMR-HSO-3). Each sample represents a different realization of the same underlying noise process: same clock physics, different random draws.

This is physically realistic. A lab characterizing a Rb standard would collect many phase records under varying conditions (temperature cycles, aging, different measurement runs). The noise *types* present are always the same; the *levels* vary modestly from run to run due to environmental and aging effects.

### Noise types present (always all four)

| Noise Type | IEEE α | ADEV Slope | Role in KF | Physical origin |
|---|---|---|---|---|
| White PM (WPM) | +2 | −1 | Measurement noise R = q_wpm | Counter noise, cabling |
| White FM (WFM) | 0 | −1/2 | Process noise q_wfm | Shot noise, thermal |
| Flicker FM (FFM) | −1 | 0 (flicker floor) | *Not modeled* — absorbed by optimizer | 1/f physics of resonator |
| Random Walk FM (RWFM) | −2 | +1/2 | Process noise q_rwfm | Environmental, aging |

### Sometimes-present: Flicker PM (FPM)

Flicker PM (α = +1) is included with **probability ~30%** at a level 0–10 dB below WPM. When present, it manifests as a slight flattening of the short-τ ADEV before the WPM slope takes over. This is realistic — FPM arises from electronics noise in the measurement chain and is sometimes visible, sometimes buried under WPM.

When FPM is present, the NLL optimizer absorbs it into a slightly elevated q_wpm (since the KF has no separate FPM state). This is the correct physical behavior.

### h_α Magnitude Ranges

Centered on realistic Rb/OCXO values with ±1.5 decades of variation to capture run-to-run and unit-to-unit spread:

| Noise Type | α | log₁₀(h_α) center | log₁₀(h_α) range | Notes |
|---|---|---|---|---|
| WPM | +2 | −25 | [−26.5, −23.5] | DMTD/counter limited |
| FPM (30% prob) | +1 | −24 | [−25.5, −22.5] | Electronics, when present |
| WFM | 0 | −24 | [−25.5, −22.5] | Fundamental oscillator noise |
| FFM | −1 | −24 | [−25.5, −22.5] | Flicker floor, always present |
| RWFM | −2 | −26 | [−27.5, −24.5] | Drift / environment |

---

## 3. What sigmatau Needs to Provide

### 3.1 Noise Generation

**Function:** Generate composite phase time series from specified h_α coefficients.

**Input:**
- h_coefficients: Dict mapping α → h_α value (4 or 5 entries)
- N: number of phase samples, **131072 (2¹⁷)**
- τ₀: sampling interval (1.0 s)
- seed: integer for reproducibility

**Output:**
- Phase time series x(t) in seconds, length N

**Method:** Timmer-Koenig synthesis or Kasdin & Walter FFT method. Each noise type generated independently and summed in phase domain.

**Why 131,072 points:** ~36 hours at τ₀=1s. Required for the KF drift state (driven by q_rwfm) to be observable. The RWFM rise on the ADEV plot appears at τ beyond the WFM→RWFM crossover, typically 100–1000s for Rb clocks. With N=2¹⁷, max useful τ ≈ 13,000s — well beyond the crossover.

### 3.2 Stability Curve Computation

**Function:** Compute overlapping ADEV, MDEV, and MHDEV at specified τ values.

**Input:**
- Phase time series x(t)
- τ₀: sampling interval
- taus: array of 20 log-spaced τ values from τ₀ to N·τ₀/10

**Output:**
- σ_ADEV(τ), σ_MDEV(τ), σ_MHDEV(τ) at each τ point (or NaN where insufficient data)

**Requirements:**
- Overlapping estimators (NIST SP1065)
- NaN for τ values with fewer than ~3 overlapping samples

**Why all three statistics:** The scientific question is whether MHDEV features improve KF parameter estimation over MDEV. Two models are trained — ADEV+MDEV vs ADEV+MHDEV — and compared.

### 3.3 KF NLL Optimizer

**Function:** Given phase data, find (q_wpm, q_wfm, q_rwfm) minimizing KF negative log-likelihood.

**Input:**
- Phase time series x(t), length 131072
- τ₀: sampling interval
- nstates: 3
- Initial guess: analytical warm start from h_α (see below)

**Output:**
- Optimized (q_wpm, q_wfm, q_rwfm)
- Final NLL value
- Convergence flag

**Requirements:**
- Optimize in log space: parameters are (log₁₀ q_wpm, log₁₀ q_wfm, log₁₀ q_rwfm)
- Bounded: each log₁₀ q ∈ [−40, −10]
- NLL from innovation sequence: NLL = Σ [log(S_k) + z_k²/S_k]
- Robust to local minima — Nelder-Mead or L-BFGS-B with analytical warm start

**Analytical warm start:**
```
q_wpm_init  = h₊₂ · f_h / (2π²)        where f_h = 1/(2τ₀)
q_wfm_init  = h₀ / 2
q_rwfm_init = (2π²/3) · h₋₂
```

**Performance:** Target < 60s per sample. 10,000 samples × 12 cores ≈ 14 hours.

### 3.4 Dataset I/O

**Format:** NumPy .npz (compressed) or HDF5.

**Contents:**
- `X_mdev`: ADEV+MDEV features, shape (n_samples, 40), log₁₀(σ)
- `X_mhdev`: ADEV+MHDEV features, shape (n_samples, 40), log₁₀(σ)
- `y`: targets, shape (n_samples, 3), log₁₀(q)
- `taus`: τ grid, shape (20,)
- `h_coeffs`: generating h_α values, shape (n_samples, 5), for provenance
- `nll_values`: final NLL per sample, shape (n_samples,), for quality filtering
- `fpm_present`: boolean mask, shape (n_samples,), which samples include FPM
- Metadata: N_points, τ₀, seed, sigmatau version

---

## 4. Dataset Generation Parameters

| Parameter | Value | Rationale |
|---|---|---|
| N_samples | 10,000 | Sufficient for tree models |
| N_points | 131,072 (2¹⁷) | ~36 hrs at 1s; drift state observable |
| τ₀ | 1.0 s | Standard |
| N_tau | 20 | Log-spaced from 1s to ~13,000s |
| Noise types | WPM + WFM + FFM + RWFM (always); FPM (30% prob) | Realistic Rb clock |
| h_α sampling | Log-uniform within ranges in Section 2 | |
| Parallelism | 12 cores | Julia Threads or Distributed |
| Random seed | 42 + sample_index | Reproducible |

---

## 5. ML Pipeline (Python / scikit-learn)

Everything below runs in Python using the saved dataset from Section 3.4.

### 5.1 Data Exploration & Visualization (Rubric: 15 pts)

Required notebook sections:

- **Feature distributions:** Histograms of all 40 features (log₁₀ σ at each τ) for both MDEV and MHDEV feature sets. Identify any long-τ features with heavy NaN rates.
- **Target distributions:** Histograms of log₁₀(q_wpm), log₁₀(q_wfm), log₁₀(q_rwfm). Check for outliers (NLL convergence failures → filter these out).
- **Example stability curves:** Plot 5–10 representative samples as log-log σ vs τ, overlaying ADEV/MDEV/MHDEV. Show visual diversity across the dataset.
- **NaN analysis:** Count and visualize NaN locations. Document the median-imputation strategy.
- **FPM presence:** Compare ADEV shapes for FPM-present vs FPM-absent samples at short τ.
- **Correlation heatmap:** Feature-feature and feature-target Pearson correlations. Expect strong correlations between adjacent τ points (they share phase data) and between ADEV and MDEV at same τ.

### 5.2 Problem Definition & Statistics (Rubric: 15 pts)

**Research question:** Can a tree-based regression model predict Kalman filter noise parameters (q_wpm, q_wfm, q_rwfm) from frequency stability curves, and does MHDEV provide better predictions than MDEV?

**Statistical baseline:**
- Mean, variance, min, max of each target
- Feature-target correlations: expect short-τ features to correlate with q_wpm, mid-τ with q_wfm, long-τ with q_rwfm
- Naive baseline: predict column mean for each target → compute RMSE. Models must beat this.
- Check: is q_wfm/q_wpm ratio correlated with any single feature? (It should be — the ADEV slope at mid-τ determines this ratio.)

### 5.3 Features

Two feature sets, trained and evaluated independently:

- **Model A (ADEV+MDEV):** 40 features — log₁₀(σ) at 20 τ points × 2 statistics
- **Model B (ADEV+MHDEV):** 40 features — log₁₀(σ) at 20 τ points × 2 statistics

Both use identical targets, train/test split (80/20, stratified by FPM presence), and hyperparameters.

- No normalization or standardization (tree-based models)
- NaN handling: replace with column median

### 5.4 Methodology & Model Choice (Rubric: 15 pts)

**Random Forest (sklearn.ensemble.RandomForestRegressor):**
- Multi-output natively (predicts all 3 targets simultaneously)
- Ensemble of bagged decision trees — robust to correlated features (adjacent τ points are highly correlated)
- No feature scaling required
- `n_estimators=500`, `max_features='sqrt'`, `min_samples_leaf=5`

**Gradient Boosted Trees (sklearn.ensemble.GradientBoostingRegressor):**
- Wrapped in `sklearn.multioutput.MultiOutputRegressor` (GBR doesn't natively support multi-output)
- Sequential boosting corrects residuals — may capture nonlinear interactions between τ regions better than RF
- `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`

**Justification:** Tree-based models are chosen because (1) features are log-transformed but span different ranges — trees are invariant to monotonic transforms, (2) no feature scaling or normalization needed, (3) strong performance on tabular regression tasks, (4) built-in feature importance for the MDEV vs MHDEV comparison.

### 5.5 Hyperparameter Tuning (Rubric: 15 pts)

**Method:** `sklearn.model_selection.GridSearchCV`, 5-fold CV

**RF grid:**
```python
{
    'n_estimators': [200, 500, 1000],
    'max_depth': [None, 20, 30],
    'min_samples_leaf': [3, 5, 10],
    'max_features': ['sqrt', 0.5],
}
```

**GBR grid (per target, via MultiOutputRegressor):**
```python
{
    'estimator__n_estimators': [200, 500],
    'estimator__learning_rate': [0.01, 0.05, 0.1],
    'estimator__max_depth': [4, 6, 8],
}
```

**Scoring:** Negative MSE on log₁₀(q). Errors are in decades — an RMSE of 0.3 means the model predicts q within a factor of 2.

**Justification for tuned parameters:**
- `n_estimators`: more trees reduce variance; diminishing returns beyond ~500
- `max_depth`: controls overfitting to specific noise realizations
- `min_samples_leaf`: prevents leaf nodes from memorizing individual samples
- `learning_rate × n_estimators`: trade-off between fit quality and overfitting for GBR

### 5.6 Evaluation & Metrics, Uncertainty (Rubric: 15 pts)

**Per-model metrics (A vs B, RF vs GBR):**
- Per-target RMSE in log₁₀ space (error in decades)
- Per-target R² score
- Per-target MAE in log₁₀ space
- Overall multi-output RMSE (average across targets)

**Uncertainty quantification:**
- **RF:** Wager-Hastie-Efron infinitesimal jackknife via `forestci` → per-prediction variance
- **GBR:** Quantile regression at α=0.05 and α=0.95 → 90% prediction intervals
- Report empirical coverage: what fraction of test samples fall within the 90% interval?

**Cross-model comparison (the scientific result):**
- Side-by-side RMSE table: Model A (MDEV) vs Model B (MHDEV) for each target and each algorithm
- Paired Wilcoxon signed-rank test on per-sample absolute errors across 5 CV folds
- Breakdown by FPM presence: does MHDEV help more when FPM is present? (Expected: yes, because MHDEV resolves FPM from WPM differently than MDEV)

**Naive baseline comparison:** All models must significantly outperform the mean-prediction baseline.

### 5.7 Result Visualization (Rubric: 15 pts)

Required plots:

1. **Predicted vs actual scatter** — 3 panels (one per q target), colored by FPM presence. 45° line for reference. One plot per best model.
2. **Residual histograms** — 3 panels, with fitted Gaussian overlay. Check for bias and symmetry.
3. **Feature importance bar chart** — top 20 features for RF, colored by statistic type (ADEV vs MDEV/MHDEV) and annotated with τ value. One chart per model (A vs B).
4. **Aggregated importance by statistic** — stacked bar: total importance from ADEV features vs MDEV features (Model A) and ADEV vs MHDEV (Model B). This is the headline comparison figure.
5. **Aggregated importance by τ region** — short (τ < 10s) / mid (10–1000s) / long (>1000s). Shows which τ region matters for each target.
6. **Uncertainty intervals** — 50 test samples sorted by predicted q_wfm, with 90% CI bands. Colored by actual q_wfm to show calibration.
7. **Model comparison bar chart** — RMSE for all (model × algorithm × target) combinations. The key result plot.
8. **Example prediction on a real sample** — if any of your GMR6000 phase records are available, compute ADEV/MDEV/MHDEV, run through the trained model, and show predicted q values alongside the stability curve.

---

## 6. Deliverables from sigmatau

1. **`generate_noise(h_coeffs, N, τ₀; seed)`** → phase time series
2. **`compute_adev(phase, τ₀, taus)`** → σ_ADEV at each τ
3. **`compute_mdev(phase, τ₀, taus)`** → σ_MDEV at each τ
4. **`compute_mhdev(phase, τ₀, taus)`** → σ_MHDEV at each τ
5. **`optimize_kf_nll(phase, τ₀; nstates=3, q_init)`** → optimized (q_wpm, q_wfm, q_rwfm), NLL, converged
6. **`generate_dataset(n_samples, N, τ₀; seed, n_cores)`** → .npz file

Items 1–4 likely already exist. Item 5 may need a wrapper. Item 6 is a parallelized driver script.

---

## 7. Validation on Real Data

After training on synthetic data:

1. Load a GMR6000 Rb phase record (from lab measurements)
2. Compute ADEV, MDEV, MHDEV at the same 20 τ points used for training
3. Format as 40-feature input, run through trained model
4. Report predicted (q_wpm, q_wfm, q_rwfm) with uncertainty intervals
5. If possible: run sigmatau's NLL optimizer on the same phase data and compare ML prediction to optimizer output
6. Overlay the predicted noise model (analytical ADEV from the predicted q values) on the measured ADEV curve

---

## 8. Timeline

| Date | Milestone |
|---|---|
| Apr 15–16 | Finalize sigmatau functions, begin dataset generation (overnight) |
| Apr 17–18 | EDA notebook, train models, GridSearchCV |
| Apr 19–20 | Evaluation, UQ, all result plots |
| Apr 21–22 | Real-data validation, notebook polish |
| Apr 23 | Presentation slides (15 min talk) |
| **Apr 24** | **Submission deadline (11:59 PM)** |
| Apr 24 | Presentation (5 bonus points if on time) |

---

## 9. Rubric Alignment Checklist

| Category (pts) | How we hit 100% |
|---|---|
| Data Exploration & Visualization (15) | Feature/target histograms, correlation heatmap, example curves, NaN analysis, FPM comparison |
| Problem Definition & Stats (15) | Clear research question (MDEV vs MHDEV for KF), statistical baselines, feature-target correlations |
| Methodology & Model Choice (15) | RF + GBR justified for tabular regression; tree invariance to log features; multi-output strategy explained |
| Hyperparameter Tuning (15) | GridSearchCV with 5-fold CV; explicit grid; justification for each hyperparameter |
| Evaluation & Metrics, Uncertainty (15) | RMSE, R², MAE per target; UQ via forestci + quantile regression; coverage; Wilcoxon test for model comparison |
| Result Visualization (15) | 8 distinct plot types; feature importance comparison is the headline; real-data demo |
| Submission (10) | On time April 24 |
| Bonus (5) | 15-min presentation on time |

---

## 10. Summary of Key Design Decisions

1. **Physically grounded dataset.** Simulates 10,000 measurement runs of a GMR6000-class Rb oscillator, not arbitrary random noise blends.
2. **4+1 noise types in, 3 targets out.** WPM+WFM+FFM+RWFM always present; FPM ~30% of the time. KF absorbs FFM and FPM via NLL optimization.
3. **Targets are log₁₀(q).** Log space equalizes scale; direct KF input.
4. **MDEV vs MHDEV head-to-head.** Same targets, same algorithms, different features. The comparison is the scientific contribution.
5. **Long records (2¹⁷ = 131,072 points).** ~36 hours at 1s. Drift state observable, RWFM clearly resolved.
6. **NLL-optimized labels.** Captures FFM/FPM model mismatch honestly.
7. **scikit-learn only.** RF + GBR with GridSearchCV.
8. **12-core parallel generation.** Overnight run.
9. **Real-data validation.** GMR6000 phase records through the trained model.
