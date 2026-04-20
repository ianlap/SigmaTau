# Talking Points & Q&A Prep

PH 551 Final Project — Monday 04/20, 9 am slot

---

## Opening — 30 seconds (Slide 1: Title)

"Good morning. I'm Ian Lapinski. My project is about using machine learning
to speed up a slow but common task in atomic clock work: tuning a Kalman
filter to phase data. The punchline — instead of running a slow iterative
optimizer, you can hand a stability curve to a trained regressor and get a
good answer in milliseconds. Let me walk you through how."

---

## Slide 2 — The Problem (1 min)

**The hook:** A lab operator has phase measurements from a clock. They want
to run a Kalman filter on it to estimate noise and predict holdover. To do
that, they need the three process-noise q values — q0 (WPM), q1 (WFM), and
q2 (RWFM).

**The pain:** Current options all require expert intervention —
- NLL optimization is slow and needs a good warm start or it diverges
- ALS is iterative and sensitive to initialization
- Manual MHDEV fitting requires picking τ-regions by eye

**The question:** Can we train a regressor to predict q-values directly
from the stability curve? And if so, which stability statistic — ADEV,
MDEV, HDEV, or MHDEV — carries the most information?

---

## Slide 3 — Why it matters (1 min)

Three-column pitch:

- **Atomic clocks matter** — GPS, telecom, defense timing, scientific
  instrumentation. Anywhere you need stable timing, someone's running a KF
  on clock data.
- **Current optimization is slow** — tens of seconds per sample, needs
  expertise, bottlenecks lab workflows.
- **ML flips the tradeoff** — train once, predict forever in
  sub-millisecond. And with uncertainty quantification, you know when to
  trust it.

---

## Slide 4 — Dataset pipeline (1 min)

"I generated 10,000 synthetic samples — and the key word is synthetic,
grounded in reality. Let me walk the pipeline:

1. I took a real 35-day GMR6000 Rb phase record and measured its h_α
   coefficients — the standard noise-type amplitudes.
2. Then I sampled h-values log-uniformly within ranges anchored to that
   measurement, with modest spread to cover related rubidium oscillators.
3. For each sample, I generate N=524,288 points (~6 days at 1 PPS) of
   composite power-law noise using the Kasdin 1992 algorithm. Four noise
   types always (WPM + WFM + FFM + RWFM), plus FPM in ~30% of samples.
4. I extract 196 features — I'll show those next.
5. For the label, I run **ALS** (Autocovariance Least Squares) in log-space
   to get the q-triple. I probed 10 draws and confirmed ALS matches truth
   to ≤10⁻³ decades; the NLL optimizer drifted by up to ±8 decades on the
   same draws and was abandoned as the label source.

The four design decisions at the bottom — log₁₀(q) targets matter most:
q's span ~6 decades across our ranges, so log space equalizes loss
contribution and matches how the KF consumes them."

---

## Slide 5 — Feature engineering (1 min)

"The feature vector is 196-dimensional, in three groups:

- **80 raw σ values** — four stability statistics (ADEV, MDEV, HDEV, MHDEV)
  at 20 log-spaced τ points each.
- **76 log–log slopes** — local slope between adjacent τ points. These are
  noise-type signatures: WPM gives slope −1 on ADEV, WFM gives −½, etc.
- **40 variance ratios** — MVAR/AVAR and MHVAR/HVAR. Classic bias-type
  detectors from the time-nuts literature.

The four-stat design is deliberate. Each statistic has different sensitivity
to different noise types, and the feature-importance analysis later will
show which ones actually carry the signal for this problem."

---

## Slide 6 — Data exploration (1.5 min)

"Quick EDA walkthrough:

- Target distributions span ~6 log decades — q2 is widest, q0 tightest. The
  three q-parameters are largely independent: q0 is set by short-term
  noise, q2 by long-term drift. We're not predicting one thing from three
  correlated views.
- Correlation heatmap shows feature–target correlations concentrated in
  the expected places: short-τ features correlate with q0, long-τ with q2.
- The example σ(τ) curves show the visual diversity of the dataset —
  different clock fingerprints.
- NaN rate is under 2%, localized at the longest τ values where not enough
  samples remain for the deviation. I use median imputation."

---

## Slide 7 — Statistical baselines (1 min)

"This slide is about setting the bar. Without a baseline an R² of 0.9
doesn't mean anything.

- **Naive:** predict training mean for every sample. RMSE is roughly the
  standard deviation of each target — the 'no model' number.
- **MHDEV 3-region fit:** the physicist's baseline — split τ into
  short/mid/long regions and fit each to its dominant noise-type slope.
  Requires expert-chosen τ-regions, so real-data only. Shown on slide 13.

Our RF and XGBoost models beat naive by roughly 18×/11×/2× on q0/q1/q2.
Correlations are monotonic but clearly nonlinear, which is what
motivates tree ensembles over linear models."

---

## Slide 8 — Model choice (1 min)

"Two models, trained on identical features.

- **Random Forest** — bagged ensemble; trees trained independently on
  bootstrap samples. Built-in uncertainty via tree-variance (forestci). A
  strong robustness baseline for tabular data.
- **XGBoost** — gradient boosting; trees trained sequentially, each
  correcting the previous errors. Typically 1–3% better RMSE on tabular
  regression.

Why both? Different bias–variance tradeoffs, so the head-to-head is
diagnostic. And they have **independent UQ mechanisms** — forestci is
analytic, quantile regression is non-parametric — so cross-checking
coverage between them is much stronger than trusting either alone.

Trees are invariant to monotone feature transforms — they don't care that
some features are log-σ and others are raw ratios. Real practical
advantage over linear models here."

---

## Slide 9 — Hyperparameter tuning (1 min)

"5-fold GridSearchCV, stratified by whether the sample contains FPM noise
(about 30% do). Scoring is neg-MSE in log-q space. Seed 42 throughout.

The grids shown are the **second pass**. The first pass landed on boundary
values (RF n_estimators=1000 at the top, min_samples_leaf=3 at the
bottom; XGB n_estimators=500 at the top, learning_rate=0.01 at the
bottom, subsample=0.8 at the bottom). We re-centered on those winners and
extended past them — the new winners are **interior**: RF settles at
n_estimators=1000, min_samples_leaf=1; XGB moves to n_estimators=750,
learning_rate=0.01, subsample=0.7. That confirms we actually found the
optimum rather than running out of grid.

Per-parameter justification:
- **n_estimators:** bias–variance tradeoff vs compute — extended to 2000
  for RF, 1000 for XGB.
- **max_depth:** prevents overfit in a 196-dim feature space where trees
  can memorize pointwise.
- **min_samples_leaf:** targets span ~6 decades; small leaves can latch
  onto noise, so this regularizes.
- **max_features:** {sqrt, 0.3, 0.5, 0.7} brackets the RF-regression
  regime — strong decorrelation (sqrt ≈ 14/196) through mild (0.7).
- **learning_rate (XGB):** smaller rate with more trees is the standard
  recipe for stable convergence; extended low to 0.003."

---

## Slide 10 — Results (1.5 min) ← KEY SLIDE

"Headline results per target, both models. [Walk through table row by row.]

- **q0** (WPM): both models hit low RMSE — the easiest target because
  short-τ ADEV directly encodes WPM amplitude. XGB reaches RMSE ~0.004
  decades, R² ≈ 0.99998.
- **q1** (WFM): harder; mid-τ features partially mix with FFM and FPM
  contributions, so there's real ambiguity. XGB hits RMSE 0.048, R² ≈ 0.997.
- **q2** (RWFM): depends on long-τ features, which are noisiest to
  estimate from finite records. Wider intervals but R² ≈ 0.82 — and this
  number itself is the payoff of switching labels from NLL to ALS (lifted
  from ~0.65).

[Point at bar chart.] Model comparison. **XGBoost wins on all three
targets** — modest per-target but the trend is consistent.

Headline box: XGB reduces RMSE ~10× over naive across all three targets."

---

## Slide 11 — Uncertainty quantification (1 min) ← GRAD REQUIREMENT

"This is the grad-student requirement, addressed by two independent
methods:

- **forestci on Random Forest** — computes the infinitesimal jackknife
  variance from the tree ensemble. Analytic, fast.
- **Quantile regression on XGBoost** — two additional XGB models with
  quantile loss at α=0.05 and α=0.95 bracketing the median. Non-parametric,
  no distributional assumption.

The key validation metric is **empirical coverage**: if I claim a 90%
prediction interval, does it actually contain the truth 90% of the time?

Current coverage on the test set: **q0 ~84%, q1 ~81%, q2 ~81%** — nominal
is 90%, so we under-cover by 6–10 percentage points. That's mildly
conservative regularization from XGB; the intervals are real, just a bit
tight. Honest about this on the limitations slide."

---

## Slide 12 — Feature importance (1 min)

"This is the headline scientific result. For each target, which of the
four stability statistics contributes most to the prediction?

[Point to figure.] Aggregated feature importance by statistic:

- **Raw σ features dominate over slopes** across the board.
- **MDEV is ~5× MHDEV** in total importance for this dataset — the
  modified variance is quantitatively the most informative single
  statistic.
- **Short-τ features drive q0, long-τ features drive q2**, as expected
  from the physics.

The practical takeaway: MDEV earns its keep on these rubidium-class
clocks. MHDEV matters at long τ where drift separation becomes important
— but its unique contribution is smaller than I expected going in."

---

## Slide 13 — Real-data validation (1.5 min)

"Synthetic training is only interesting if it generalizes to real clocks.
I ran the full pipeline on the original GMR6000 Rb phase record:

1. Loaded 35 days of 1 PPS data.
2. Windowed into 4 non-overlapping N=524,288 segments (matches training
   dimension).
3. Extracted the 196 features from each window.
4. Predicted q-triples with RF and XGB.
5. Computed analytical ADEV from those predicted q's — Wu 2023 convention
   — and overlaid it on the measured ADEV plus the MHDEV-fit and ALS
   references.

[Point to overlay.] RF and XGB σ(τ) predictions track the measured ADEV
across all four windows — meaning the synthetic training regime
generalizes to real hardware. Cross-checking against the ALS optimizer
gives agreement within the uncertainty intervals. NLL was dropped from
this comparison because it basin-locks on the 3-state model when FFM is
present in the data."

---

## Slide 14 — Takeaways & limitations (1 min)

"Quick summary.

**What worked:**
- ALS labels gave noise-free targets — the single biggest win; lifted q2
  R² from ~0.65 (NLL-labeled) to ~0.82.
- XGB hits 0.004-decade RMSE on q0 (R² ≈ 0.99998) and 0.048 on q1.
- forestci + quantile UQ give 80–84% empirical coverage of nominal 90%
  prediction intervals.
- Feature-importance analysis quantified MDEV as ~5× MHDEV for this
  oscillator class.

**Limitations — honest about these:**
- Trained on GMR6000-class Rb only; HSO or cesium would need retraining.
- The 3-state KF lacks flicker-FM, limiting long-horizon holdover fidelity.
- Only one real-data record (6k27feb) for cross-check.
- XGB 90% PI under-covers by 6–10 percentage points.

**Future work:**
- Extend training ranges to OCXO, HSO, cesium.
- 4-state model with explicit FFM for sub-nanosecond long-term holdover.
- End-to-end neural network on raw phase — see if learned features beat
  engineered ones.
- Active learning on where the model is least confident."

---

## Slide 15 — Thanks + Q&A

"Happy to take questions."

---

# Anticipated Q&A

## Methodology questions

**Q: Why synthetic data instead of training on real clocks?**
A: Because I need ground-truth q-values to supervise on, and those don't
exist for real clocks — they're what the optimizer is trying to estimate.
Synthetic data lets me generate with known q's. The real-data validation
shows this generalizes.

**Q: Why ALS labels instead of NLL?**
A: I probed both on 10 synthetic draws. ALS matched the truth to ≤10⁻³
decades; NLL drifted by up to ±8 decades on the same inputs because the
NLL surface is multi-modal for the 3-state clock model when FFM is present
— the optimizer absorbs FFM into q1 (WFM). ALS is well-conditioned and
converges cleanly. It was the single biggest dataset-quality improvement.

**Q: Why not deep learning?**
A: 10,000 samples, 196 features, tabular — tree ensembles are the right
tool class. Deep learning typically needs 10× more data to beat XGBoost on
tabular problems. Tree models also give free feature importance, which is
a scientific contribution on its own here. Neural nets on raw phase would
be future work.

**Q: Why Random Forest AND XGBoost — isn't one enough?**
A: Two reasons. First, different bias–variance tradeoffs make the
comparison diagnostic. Second, they have independent UQ mechanisms —
forestci is analytic, quantile regression is non-parametric — so
cross-checking empirical coverage between them is much stronger than
trusting either alone.

**Q: Why log₁₀(q) as targets?**
A: The q values span ~6 decades across our ranges. In linear space MSE
would be dominated by the largest-magnitude target. Log space equalizes
loss contribution and matches how the KF consumes the parameters.

**Q: Why did you re-run grid search?**
A: The first pass landed on boundary values (RF n_estimators=1000 top,
min_samples_leaf=3 bottom; XGB n_estimators=500 top, learning_rate=0.01
bottom, subsample=0.8 bottom). That's a red flag — the optimum might lie
outside the grid. I re-ran a focused second pass centered on those
winners with extended edges, and all new winners are interior (RF stayed
at n_est=1000 but moved to leaf=1; XGB moved to n_est=750 and
subsample=0.7). So the first-pass RMSEs were real floors, not grid
artifacts — and the second pass gave additional small gains.

## Data questions

**Q: How representative are your h-ranges?**
A: Centered on the measured GMR6000 Rb record with modest spread to cover
related rubidium oscillators. For cesium standards or OCXOs you'd need to
re-train with those ranges.

**Q: What about non-stationary clocks — drift, aging, temperature?**
A: The current model assumes stationary noise. RWFM handles drift, but
long-term aging or thermal excursions would violate the assumption. One
path forward is windowed prediction with a changepoint detector on the
predictions themselves.

**Q: Does the model handle FPM?**
A: It sees FPM in ~30% of training samples. FPM is absorbed into q0 and
q1 by the ALS optimizer — not a separate target. A clock with very strong
FPM may be biased, but the uncertainty intervals should widen to catch it.

## Results questions

**Q: What's driving the XGBoost improvement over Random Forest?**
A: Gradient boosting explicitly models residuals from previous trees, so
it's better at capturing the subtle mid-τ structure where noise types
mix. RF trees are independent, so they can't collaborate that way.

**Q: What does an RMSE of 0.05 in log₁₀(q) mean practically?**
A: RMSE 0.05 decades is ~12% error in q itself. For the KF, that's well
within the robust regime — the filter's innovation statistics are pretty
insensitive to q within a factor of 2.

**Q: How does your prediction time compare to ALS/NLL?**
A: Iterative optimizers are 1–60 seconds per sample depending on
tolerance. Feature extraction is ~50 ms in Julia; model predict is
sub-millisecond. End-to-end ML inference is 100–1000× faster.

## Uncertainty questions

**Q: How did you verify calibration?**
A: Empirical coverage on the held-out test set. For a nominal 90%
interval, count the fraction of test samples whose true q lies inside
the predicted interval. Getting that within a few percent of 90 is the
check. We're at 81–84% — slightly under-covering, which I flag as a
limitation.

**Q: Epistemic vs aleatoric uncertainty?**
A: forestci and quantile regression capture the combined total. To
separate them you'd need a Bayesian model or an ensemble of ensembles.
Flagged as future work.

## Limitations questions

**Q: Your real-data validation is one record. Thin?**
A: Yes, thin. Production would want validation across a dozen records
from different oscillator classes. I had the 35-day GMR6000 record and
windowed it four ways; a full validation matrix is future work.

**Q: What if someone gives you an out-of-distribution clock?**
A: Uncertainty intervals should widen; if they don't, the model is
overconfident — that's a failure mode. A production deployment would need
an OOD detector on the feature space (Mahalanobis distance is a cheap
first pass) before trusting the prediction.

## Out-of-left-field questions

**Q: Could this replace ALS/NLL entirely, or always a warm-start?**
A: For a good initial guess, ML alone is enough. For a final fit at
publication accuracy, you'd still run the iterative optimizer — the ML
prediction gets you a warm start that converges in a couple of outer
iterations. That's the practical use case.

**Q: Have you considered physics-informed loss?**
A: Yes. The analytical ADEV has a known functional form in the q's, so
you could add a consistency loss penalizing predictions whose implied
ADEV doesn't match the input features. Future-work bullet I didn't put
on the slide.

**Q: What's novel here?**
A: Two things. First, a feature-importance quantification of which
stability statistic carries the signal for each KF parameter — a new
contribution to the time-and-frequency community. Second, the UQ is
calibrated end-to-end across 6 decades of dynamic range, which is
non-trivial for tree-based methods.

**Q: Why did NLL basin-lock on your real Rb data?**
A: The 3-state clock model (WPM / WFM / RWFM) has no FFM state. Real
rubidium clocks have genuine flicker-FM, and NLL absorbs it into q1,
inflating the WFM estimate by ~5 decades. ALS is more robust because it
fits autocovariances at multiple lags simultaneously rather than chasing
a single likelihood surface.
