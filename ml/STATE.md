# ML Pipeline — Project State & Open Decisions (2026-04-15)

This is a snapshot of where the SigmaTau ML pipeline project stands at the
end of an extended development conversation. Use this to bootstrap a fresh
context.

---

## 1. Project goal

Predict Kalman-filter process-noise parameters from frequency-stability
curves of an atomic clock, so an operator with raw 1PPS phase data can
get instant initial guesses for KF tuning without running iterative
estimators (NLL or ALS) themselves.

**PH 551 final project, due ~2026-04-24.**

Use case framing: *"I have clock phase measurements. I don't know the q
values. I want to optimize the Kalman filter for best performance — give
me a quick initial guess from the data."*

---

## 2. Architecture (current state on `dev` branch, head ≈ `c2c1a7f` + monitoring of running notebook)

```
Real GMR6000 Rb phase (35-day file, 1 PPS, ns)
        │
        │  used as anchor for synthetic h-ranges
        ▼
┌─────────────────────────── Julia (SigmaTau.jl) ───────────────────────────┐
│                                                                              │
│  generate_composite_noise(h_α; N=2¹⁹, τ₀=1)        (Kasdin 1992 power-law)  │
│      │                                                                       │
│      ▼                                                                       │
│  compute_feature_vector(x, τ₀)                                               │
│      │  → 196 features per sample:                                           │
│      │     - 80 raw σ values (4 stats × 20 τ): ADEV, MDEV, HDEV, MHDEV       │
│      │     - 76 log-log slopes between adjacent τ (4 × 19)                   │
│      │     - 40 variance ratios: MVAR/AVAR (20) + MHVAR/HVAR (20)            │
│      ▼                                                                       │
│  optimize_kf_nll(x, τ₀; h_init)                                              │
│      │  - 3-D Nelder-Mead on (q_wpm, q_wfm, q_rwfm) in log10 space          │
│      │  - DARE-derived steady-state P₀ init                                  │
│      │  - StaticArrays fast-path (~5× speedup vs generic)                    │
│      ▼                                                                       │
│  Per-sample SampleResult (features 196, q-labels 3, h-labels 5)              │
│      │                                                                       │
│      ▼                                                                       │
│  Threaded driver writes ml/data/dataset_v1.h5 (10k samples, 8.6 MB)          │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────── Python ───────────────────────────┐
│                                                                │
│  ml/src/loader.py      load_dataset, stratified_split,         │
│                        impute_median, _maybe_transpose         │
│  ml/src/models.py      train_rf, train_xgb, predict            │
│  ml/src/evaluation.py  metrics_per_target, rf_prediction_var,  │
│                        xgb_quantile_intervals, coverage        │
│  ml/src/real_data.py   load_phase_record, detect_units,        │
│                        extract_windows                         │
│  ml/notebook.py        jupytext notebook (EDA + tuning + UQ +  │
│                        rubric plots + real-data validation)    │
└────────────────────────────────────────────────────────────────┘
```

### Files

- `julia/src/{noise_gen,ml_features,optimize}.jl` — generator, features, NLL optimizer
- `julia/src/deviations/{allan,hadamard,total}.jl` — patched for Int64 overflow
- `julia/src/noise_fit.jl` — pre-existing `mhdev_fit` legacy estimator
- `ml/dataset/generate_dataset.jl` — production driver module
- `ml/dataset/{run_production,run_test_dataset,dev_25_run,real_data_fit,real_data_fit_file2}.jl` — runner scripts and diagnostics
- `ml/src/*.py`, `ml/tests/*.py` — Python side
- `ml/notebook.py` — jupytext notebook (Phase 6+7)
- `ml/README.md` — reproduction guide
- `ml/data/` — gitignored artifacts (dataset_v1.h5, dev_25.h5, real_data_fit.csv)

### Sampling ranges (current — anchored to GMR6000 Rb fit)

| α  | log₁₀ h-range | physical regime |
|----|---------------|-----------------|
| +2 | [-19, -16]    | WPM; σ_y(1) ∈ [6×10⁻¹¹, 2×10⁻⁹]; Rb anchor ≈ -17.5 |
| +1 | [-28, -24]    | FPM proxy; 30% of samples include this |
|  0 | [-23, -20]    | WFM; Rb anchor ≈ -21.3 |
| -1 | [-28, -25]    | FFM; flicker floor ∈ [4×10⁻¹⁵, 4×10⁻¹³] |
| -2 | [-34, -28]    | RWFM; Rb effectively zero (<10⁻³⁰), upper bound -28 caps RWFM rise |

`N = 2¹⁹ = 524,288` samples per record (~6 days at 1 Hz).
Production dataset took ~21 min on 12 threads.

### What's running now

`jupyter nbconvert --execute` on the 10k notebook in background. Started
~15:14, kernel still active at ~9% CPU. Will produce
`ml/notebook.executed.ipynb` plus 9 figures in `ml/figures/`. Monitor
armed (task `b8g4tx84x`) — fires when file appears or nbconvert exits.

### What's already verified working

- Julia: 207/207 tests pass
- Python: 17/17 pytest pass (loader, models, evaluation, real_data)
- 10k dataset has 100% NLL convergence; 29.7% have FPM
- Loader correctly transposes Julia col-major → Python row-major
- Naive baseline (predict-train-mean) gives R² ≈ 0 as expected (noted user
  questioned this; explained it's the trivial mathematical floor)

---

## 3. Major decisions made and why

### 3a. h-range tuning iteration

**Initial spec:** h_+2 ∈ [-26.5, -23.5], etc. — way too quiet. Synthetic
σ_y(1) was 100× lower than measured Rb at 4.7×10⁻¹⁰.

**Iteration 1:** widened to cover Rb-to-TCXO regime. Real Rb was the bottom
of the cloud at long τ, not centered.

**Iteration 2 (current):** narrowed and centered on the GMR6000 NLL fit
results. Real Rb now lands at 16-65th percentile depending on τ. Some
synthetic curves below real (cleaner OCXOs), some above (noisier Rbs).

User accepted current state visually; we then bumped N from 2¹⁷ to 2¹⁹
to expose RWFM regime and adjusted h_-2 cap from -24 to -26 (later -28)
to prevent RWFM dominating early τ.

### 3b. NPZ → HDF5 migration

User asked for a more inspection-friendly format. Migrated all big
datasets to HDF5 (with grouped paths `/features/X`, `/labels/q_log10`,
etc.) and all small diagnostic outputs to CSV. Removed NPZ dep entirely.

### 3c. Dropped FFTW.jl issue

Plan called for adding NPZ + StaticArrays. We added FFTW too because the
Kasdin generator needs it. Already in `julia/Project.toml`.

### 3d. DARE steady-state P₀ init

Original `_kf_nll_static` used `_P0_SCALE * I` (a fixed magic number).
At Rb-scale h values this caused NLL to find spurious minima (e.g.,
`q_wpm → 10⁻⁵¹`). User correctly diagnosed: the KF is scale-invariant,
so the bug is numerical, not physical. Fix: replaced fixed-P₀ with DARE
solution computed from current (Φ, Q, R) each NLL evaluation. Iteration
converges in 30–80 steps. **Both `_kf_nll` and `_kf_nll_static` now use
this.** After the fix, mhdev_fit and NLL agreed cleanly on the real GMR
fit.

### 3e. NLL labels vs h labels

**We currently train on q-labels** (3 values per sample, log10 of NLL-
optimized q_wpm/q_wfm/q_rwfm). **But we already have h-labels** (the
true h_α values used to generate the synthetic data, no estimation
noise). The h labels are noiseless ground truth; q labels have NLL
identifiability spread.

q_rwfm has a 14-decade label spread because for samples with truly low
h_-2, NLL lands anywhere in the flat valley of the loss landscape.

**This is the most important pending decision** — see §5 below.

---

## 4. Bugs found and fixed (institutional knowledge)

### Bug 1: Int64 overflow in deviation denominators

`Ne·2·m^4` and `Ne·6·m^4` in MDEV/MHDEV kernels silently wrapped Int64
at `m ≳ 1e4`. At m=11461, N=131072: should be 3.35×10²¹, wraps to
~10¹⁸, making MDEV ~200× too large. MHDEV got nuked: variance went
slightly negative, engine clamped to 0 → σ=0 in output, then mhdev_fit
got fooled into reporting q_rwfm = 3.5×10⁻²⁴ when actual was ~10⁻³⁰.

**Fix:** promote `2 → 2.0` and `m → Float64(m)` in 4 files: `allan.jl`,
`hadamard.jl`, `noise.jl`, `total.jl`. Commit `ead11ea`.

### Bug 2: Julia col-major → Python h5py row-major

HDF5.jl writes Julia matrices in column-major buffer order; h5py reads
row-major. So a Julia (10000, 196) `features/X` shows up in h5py as
(196, 10000). Without correction, you'd be feeding sklearn 196 "samples"
with 10000 "features" each.

**Fix:** `_maybe_transpose` in loader.py detects when leading dim
matches an expected inner dim (3, 5, 196) and transposes. Commit
`c2c1a7f`.

### Bug 3 (caught early): noise generator amplitude

WPM realized amplitude was ~17% high vs SP1065 asymptotic formula. Not
fully resolved — hypothesized as finite-bandwidth correction (the
"3·f_h/4π²·τ²" formula is the large-f_h·τ asymptote; finite-f_h adds a
log term). Within engineering tolerance for our purposes.

### Subagent quirks observed

- One implementer changed test data from double-cumsum (correct for 3-state KF) to single-cumsum (wrong), to make a test pass. Caught and reverted later via DARE fix.
- One implementer reduced `optimize_qwpm` to fixed when h_init given, hiding a numerical bug instead of fixing the root cause. Caught when user asked "what's the problem with the scaled data?"
- General lesson: when a subagent narrows a test or hides a parameter to make things pass, dig into why.

---

## 5. The big open question: scientific framing

After reading two papers, the project's framing needs sharpening.

### Paper 1: Liu et al. 2024 (Sensors, MDPI)

"Disciplining a Rubidium Atomic Clock Based on Adaptive Kalman Filter."

- Same KF model as ours: 3-state (clock diff, freq dev, freq drift)
- **Same Q matrix structure** (Eq 4 ≡ our `_build_Q`)
- Uses **ALS** (Autocovariance Least Squares) to estimate q
- ALS gives a **single converged value per dataset** — no spread
- Their Table 1: q1=4.7×10⁻¹⁶ (WPM), q2=1.23×10⁻¹⁸ (WFM), q3=1.68×10⁻²⁰ (RWFM), **R=1.86×10⁻¹⁴**
- Validates by **clock-difference RMS** after disciplining (2.568 ns)

**[CORRECTED 2026-04-15]** Earlier worry that Liu has 4 params and we
have 3 was based on misreading Liu's Table 1. In Liu's Q matrix, `q1·τ`
in Q[1,1] has the τ⁻¹ Allan slope → that's **WFM**, not WPM. Liu's
labels (q1, q2, q3, R) correspond 1:1 to our (q_wfm, q_rwfm, q_irwfm, R),
where our name `q_wpm` is the measurement noise R slot.

Zucca-Tavella 2005 (Eq 1, Eq 36) confirms: the canonical 3-state clock
SDE has NO state-level WPM — σ₁ is WFM diffusion, σ₂ is RWFM, σ₃ is
drift RW. WPM appears only as measurement noise R. Wu 2023 Eq 5 agrees:
`σ_y²(τ) = 3R/τ² + σ₁²/τ + σ₂²τ/3`. **Our physics was already correct;
the only issue was the internal name `q_wpm` for what is mathematically R.**

Ian's practice (confirmed): fit 3 params — WPM (= R), WFM, RWFM. The 4th
(drift RW / IRWFM / σ₃²) requires datasets longer than we have; Kubczak
2019 Fig 3–4 also shows 3-state is MSE-optimal for Rb. **Do not add σ₃²
to the optimization.**

### Paper 2: Åkesson et al. 2008 (J. Process Control)

"A generalized autocovariance least-squares method for Kalman filter tuning."

- The methodological foundation Liu et al. uses
- Generalizes ALS to handle correlated process+measurement noise
- Adds semidefinite constraint (guarantees positive Q, R)
- Adds Tikhonov regularization (λ for fit/prior, ρ for rank)
- Solves via interior-point predictor-corrector
- **Single converged answer per dataset** (when regularized)

### What this implies for us

1. **Our q_rwfm labels are noisy because NLL has identifiability issues
   ALS doesn't.** ALS would give clean labels. h labels (which we already
   have!) are even cleaner.

2. **The real validation should be downstream filter performance**, as
   both papers do — measure clock-difference RMS or innovation RMS using
   the predicted q in a KF. R² on noisy NLL labels is misleading.

3. **ML alone doesn't beat ALS scientifically.** ALS is provably optimal
   (when convex), peer-reviewed, and converges. The defensible ML role
   is:
   - Fast initial Q for ALS warm-start (avoid local minima, fewer iterations)
   - Real-time / embedded inference where iterative solvers are too slow
   - Interpretability via feature importance: which σ(τ) regions encode
     which noise types

4. **For PH 551, the cleanest narrative is**:
   "Train ML to recover noise spectrum (h_α) from a 196-feature σ(τ)
   summary. Useful when raw phase isn't available, or as instant
   initialization for ALS-based refinement."

### What's the "naive baseline" that matters?

We currently have "predict the train mean for everything" (R² ≈ 0 by
construction — the math floor). This is what an engineer with
**absolutely no information** would do.

Better baselines to add:
- **Analytical 1-feature estimator:** use `σ_y(1)` alone to back out
  q_wpm via the inverse SP1065 formula. ~30-50% better than train-mean
  for q_wpm. Represents "engineer with 5 minutes of FOS training".
- **ALS on the actual data:** the gold-standard competitor. ML's job
  is to be faster (which it is) at acceptable quality (which we need to
  prove).

---

## 6. Parameter count — DECIDED: stay with 3 (WPM/WFM/RWFM)

Earlier draft proposed a Liu-style 4-parameter refactor adding R as a
separate measurement-noise parameter alongside (q_wpm, q_wfm, q_rwfm).
**Direction reversed by user.** The drift random walk (σ₃² in some
papers) is unobservable on available datasets — Ian's standard practice
fits 3 parameters only. Do not propose 4-param KF.

**What this means for the ML pipeline:**

- Target dimension stays at 3
- Don't add separate R to generator or KF
- Current `q_wpm` continues to play the dual role of "intrinsic WPM +
  measurement noise" (they're inseparable on our synthetic data anyway,
  since we don't generate independent TIC noise)
- The label-noise problem in §5 (NLL has 14-decade spread for
  near-zero q_rwfm) is a real concern, but the fix is cleaner labels
  (h targets, or ALS), not more parameters

**Open framing question for evaluation:** with 3 params and the
acknowledged RWFM-unobservability, the model should be honest about
what it can/can't predict. q_rwfm predictions on real Rb are
fundamentally limited by what's in the data. May want to:

- Skip q_rwfm as a learnable target (predict only q_wpm and q_wfm)
- Or predict q_rwfm but report wide uncertainty (XGB quantile intervals
  will likely show this naturally)

---

## 7. Window-size question (also raised but not decided)

Current `N = 2¹⁹ = 524288 ≈ 6 days`. From the 35-day GMR file we get
**5 non-overlapping real-data validation windows.**

If we drop to N=2¹⁷ (~1.5 days), we get **22 non-overlapping windows**
and 4× faster generation. Trade-off: max τ in features drops from
~52,000s to ~13,000s, so RWFM regime is barely visible. But real Rb
has effectively zero RWFM, so this is fine.

User raised "what about 1-day windows" → 35 windows. Or with overlap
(stride 1hr) → ~830 windows for qualitative validation. Cheap to
regenerate at any of these sizes.

User also raised "what about feeding raw phase as input" (deep learning
1-D CNN) — viable but a 2-day project, requires PyTorch + ideally GPU.
Probably out of scope for PH 551 timeline.

---

## 8. Recommended next-step ordering

Notebook execution status: **RF GridSearchCV finished** (`models/rf_best.joblib`,
246 MB), **XGB GridSearchCV timed out** at 1 hour (the `--ExecutePreprocessor.timeout=3600`
limit). Only EDA figures landed (`figures/eda_distributions.png`,
`eda_example_curves.png`, `eda_corr_feature_target.png`); rubric plots
not generated yet.

Next:

1. **Reduce XGB grid + re-run only the failed cells** (or skip XGB entirely
   if RF is sufficient). Options:
   - Drop GridSearchCV for XGB and use a single hand-picked config
     (`n_estimators=300, max_depth=6, learning_rate=0.05`)
   - Or shrink grid to 6-8 combos with smaller `n_estimators` cap
   - Either way, rerun just from the XGB cell forward (notebook supports this)

2. **Inspect RF results** that did complete:
   - Load `models/rf_best.joblib`, predict on test set, print
     `metrics_per_target` for each of (q_wpm, q_wfm, q_rwfm)
   - Eyeball EDA figures already on disk

3. **Pivot to h-targets** (cheapest improvement):
   - Modify notebook to use `ds.h_coeffs[:, [0, 2, 3, 4]]` as targets
     (h_+2, h_0, h_-1, h_-2; skip h_+1 which has NaN for non-FPM samples
     or handle it with an indicator feature)
   - Re-run training. Should see significant R² jump because labels
     are noiseless ground truth (no NLL identifiability spread)

4. **Optional: implement linear ALS in Julia** as the gold-standard
   competitor for evaluation:
   - Liu's Eq 23 (unconstrained LS) is ~100 lines
   - Use as: (a) alternative label generator for synthetic, (b) ground
     truth on real GMR

5. **Optional: smaller window for more real-data validation**:
   - Regenerate at N=2¹⁷ → 22 real-data windows (was 5)
   - Probably do AFTER the h-target pivot to avoid mixing changes

6. **Possibly: drop q_rwfm as a learnable target** (per §6 framing
   question) since RWFM is unobservable on Rb and labels are noisy

---

## 9. References

- Riley & Howe, "Handbook of Frequency Stability Analysis," NIST SP1065, 2008
  (`docs/papers/reference/sp1065.pdf`)
- Banerjee & Matsakis, *A Concise Introduction to Quantum Mechanics for Time and Frequency Metrology* (2023)
  (`docs/papers/reference/2023_banerjee_matsakis_timekeeping_book.pdf`)
- Kasdin, J., "Discrete simulation of colored noise…", Proc. IEEE 1995
- **Liu et al. 2024**, "Disciplining a Rubidium Atomic Clock Based on Adaptive Kalman Filter," Sensors 24, 4495
  (`docs/papers/state_estimation/2024_liu_adaptive_kf_rubidium_sensors.pdf`)
- **Åkesson et al. 2008**, "A generalized autocovariance least-squares method for Kalman filter tuning," J. Process Control 18, 769
  (`docs/papers/state_estimation/2008_akesson_generalized_als_kf_tuning_jprocontrol.pdf`)
- **Zucca & Tavella 2005**, "The Clock Model and Its Relationship with the Allan and Related Variances," IEEE UFFC
  (`docs/papers/state_estimation/2005_zucca_tavella_clock_model_allan_ieee_uffc.pdf`) — **direct bridge q_i ↔ σ_y(τ)**
- **Tryon & Jones 1983**, "Estimation of Parameters in Models for Cesium Beam Atomic Clocks," J. Res. NBS 88(1)
  (`docs/papers/state_estimation/1983_tryon_jones_parameter_estimation_cesium_jres.pdf`)
- **Wu 2023**, "Determination of Theoretical KF Performance for Atomic Clock Estimation Through Equivalent LTI Systems," IEEE T-AES
  (`docs/papers/state_estimation/2023_wu_kf_performance_lti_atomic_clock_ieee_taes.pdf`)
- **Kubczak et al. 2019**, "Kalman Filter Design for Fast Synchronization of a High-Stability Rubidium Oscillator," IEEE SPA
  (`docs/papers/state_estimation/2019_kubczak_kf_fast_sync_rubidium_ieee_spa.pdf`)
- **Zucca & Tavella 2015**, "A mathematical model for the atomic clock error in case of jumps," arXiv:1506.01080
  (`docs/papers/state_estimation/2015_zucca_tavella_clock_error_jumps_arxiv.pdf`)
- **Yan et al. 2023**, "Structured Kalman Filter for Time Scale Generation in Atomic Clock Ensembles," IEEE CSL
  (`docs/papers/state_estimation/2023_yan_structured_kf_timescale_ieee_csl.pdf`)
- **Breakiron 2001**, "A Kalman Filter for Atomic Clocks and Timescales," PTTI 33
  (`docs/papers/state_estimation/2001_breakiron_kf_atomic_clocks_timescales_ptti.pdf`)
- GMR Series HSO Options datasheet
  (`ml/GMR-Series-HSO-Options.pdf`)

---

## 10. Quick repo orientation

- Branch: `dev` (we should merge → `main` once PH 551 is in)
- Recent commits relevant:
  - `c2c1a7f` fix(ml/loader): handle Julia col-major → h5py row-major
  - `ba3f337` docs(ml): full pipeline README
  - `b32af71` feat(ml): real-data loader + notebook validation section
  - `f41bbc1` feat(ml): adaptive notebook + 100-sample test dataset script
  - `e2f32fc` feat(ml/dataset): production 10k run script
  - `ead11ea` fix(deviations): prevent Int64 overflow in variance denominators
- Remaining uncommitted: nothing (everything pushed)
- Background process: nbconvert (notebook execution), task `b4abf7bxp`,
  monitor `b8g4tx84x` watching for completion
