---
name: ml-pipeline
description: >
  Use when building, modifying, or debugging the SigmaTau ML pipeline —
  dataset generation, 196-feature extraction, RF/XGBoost training, HDF5
  Julia↔Python handoff, or real-data validation. Trigger when working in
  ml/src/*, ml/dataset/*, ml/notebook.py, ml/tests/*, or calling
  SigmaTau.compute_feature_vector / CANONICAL_TAU_GRID / FEATURE_NAMES.
---

# ML Pipeline

PH 551 final project (CLAUDE.md / GEMINI.md §1.1): predict Kalman-filter
process-noise parameters `(q_wpm, q_wfm, q_rwfm)` from a σ(τ) curve so an
operator with raw phase data gets an instant KF tuning guess without running
iterative NLL/ALS.

## Architecture

```text
            Julia                        HDF5              Python
──────────────────────────────    ─────────────      ───────────────────
generate_composite_noise(h)            │              loader.load_dataset
    ↓                                  │              → Dataset (X, y, h, …)
compute_feature_vector(x, τ0)          │                      ↓
    ↓                                  │              stratified_split
optimize_nll(x, τ0; h_init, ...)       │              impute_median
    ↓                                  │                      ↓
SampleResult(features, q-labels, ...)  │              train_rf / train_xgb
    ↓                                  │                      ↓
ml/dataset/generate_dataset.jl    →  .h5  →         evaluation + UQ
(threaded; checkpoints every 500)                     (real-data overlay)
```

Key files: `ml/dataset/generate_dataset.jl`, `ml/src/{loader,models,evaluation,real_data}.py`,
`ml/notebook.py`. `ml/dataset/Project.toml` carries its own SigmaTau dep
(separate from `julia/Project.toml`) — `ml/` behaves as a sibling project.

## 196-feature layout

From `julia/src/ml_features.jl`. Canonical m-grid is 20 log-spaced integers
from 1 to 11461 (for N = 131072):

```text
CANONICAL_M_LIST = [1, 2, 3, 4, 7, 12, 19, 31, 51, 83,
                    136, 222, 364, 596, 976, 1597, 2614, 4279, 7003, 11461]
CANONICAL_TAU_GRID = Float64.(CANONICAL_M_LIST)   # at τ₀ = 1 s
```

Feature vector (ordered):

- **80 raw σ values** — 4 stats (adev, mdev, hdev, mhdev) × 20 τ, reported as `log10(σ)`
- **76 adjacent-τ slopes** — 4 stats × 19 log-log slopes between consecutive τ
- **20 MVAR/AVAR ratios** — per τ
- **20 MHVAR/HVAR ratios** — per τ

`FEATURE_NAMES` (exported from SigmaTau) is the ordered string array; treat
`CANONICAL_TAU_GRID`, `CANONICAL_M_LIST`, and `FEATURE_NAMES` as a stable
public-API contract — the ML subsystem pins to these names.

## HDF5 schema

Produced by `ml/dataset/generate_dataset.jl:185-201`:

```text
/features/X              float32  (n_samples, 196)
/labels/q_log10          float64  (n_samples, 3)    — log10(q_wpm, q_wfm, q_rwfm)
/labels/h_log10          float64  (n_samples, 5)    — log10(h₊₂, h₊₁, h₀, h₋₁, h₋₂); NaN where absent
/labels/fpm_present      uint8    (n_samples,)
/diagnostics/nll_values  float64  (n_samples,)
/diagnostics/converged   uint8    (n_samples,)
/meta/taus               float64  (20,)
/meta/feature_names      strings  (196,)
/meta/n_done             int
/meta/n_samples_total    int
```

## Julia↔Python HDF5 axis-swap (Bug 2 — don't regress)

HDF5.jl writes Julia matrices in column-major buffer order. h5py reads
row-major. A Julia `(n_samples, 196)` matrix on disk shows up in h5py as
`(196, n_samples)` without correction. Without the fix, you feed sklearn 196
"samples" of 10000 "features" each, and every downstream step is silently
nonsense.

Fix: `_maybe_transpose` in `ml/src/loader.py:35-42` (commit `c2c1a7f`, ml/STATE.md:183-192).
Detects when the leading dim matches an expected inner dim (3, 5, or 196) and
transposes. Call sites at `loader.py:67-69` for `X`, `y`, `h`.

Do not remove `_maybe_transpose` or skip the call even for "small" arrays.
The signature `expected_inner` makes it safe: it only transposes when shapes
match the expected contract.

## Subprocess shell-out pattern

`ml/notebook.py:396-438` computes features on real data by shelling out to
Julia:

1. Write phase windows to a tempfile HDF5
2. Write a ~25-line Julia script to a tempfile that `using SigmaTau`,
   reads the HDF5, calls `compute_feature_vector(view(windows, :, i), 1.0)`
   per window, writes a CSV
3. Spawn `julia --project={ml/dataset} --threads=auto <script> <in.h5> <out.csv>`
4. `np.loadtxt` the CSV back into Python

This is the coupling shape of an external consumer, not an integrated
package. Do not assume the Julia and Python processes share memory. Assume
the subprocess cost is ~seconds per invocation (package loading dominates).

The inline `h5py wrote (n_windows, window_size) C-order; Julia reads
column-major → (window_size, n_windows)` comment at line 413 is the same axis
swap as Bug 2 in reverse; the Julia script uses `size(windows)` to get
`(window_size, n_windows)` and loops over columns.

## Wu 2023 q↔h canonical convention

**Must match `kalman-filter-shared` exactly.** Canonical source:
`julia/src/clock_model.jl:138-157` (`h_to_q`, `q_to_h`). Mappings are
reproduced in `kalman-filter-shared`; do not re-derive or paraphrase them
here — follow the shared skill's table.

**Pre-Wu legacy at `ml/notebook.py:463` (inline `analytical_adev` helper):**
uses `h₊₂ = q_wpm · 2π² / f_h` (factor of 2 off) and
`h₋₂ = 3·q_rwfm / (2π²)` (factor of 3 off). Parked in FIX_PARKING_LOT.md:7.
When the notebook is next touched, replace with a call through `q_to_h` to
share the canonical source. Don't propagate the pre-Wu formulas into any new
code.

## Three-parameter decision (don't propose a 4-parameter fit)

ClockNoiseParams carries 4 coefficients (`q_wpm`, `q_wfm`, `q_rwfm`,
`q_irwfm`). Drift-random-walk (`q_irwfm`, σ₃²) is **unobservable on
currently-available datasets** — the NLL valley is flat. The pipeline fits 3
parameters (WPM / WFM / RWFM).

CLAUDE.md §1.3, STATE.md §6. Do not propose fitting `q_irwfm` without
explicit user sign-off — it adds weeks of label spread and burns training
compute without improving generalization.

## Convergence logging (Fix 2)

`ml/dataset/generate_dataset.jl:109` reads `opt_res.converged` from the
`OptimizeNLLResult` returned by `optimize_nll`. Commit `e7728f6` replaced a
hardwired `true` with this honest signal. `loader.py` filters unconverged
samples via `filter_unconverged=True` (default) at load time.

Don't revert to hardwired convergence. The optimizer's `converged` flag
(max_iter vs std-of-fvals-below-tol) is the ground truth.

## Int64 overflow on large-m MDEV/MHDEV

At the canonical m=11461, the MDEV/MHDEV Int64 denominators overflow without
the `Float64(m)` promotion. See `deviation-engine` for the full story (commit
`ead11ea`); the ML pipeline is the workload that surfaces this because
`CANONICAL_M_LIST` goes up to 11461.

## Sampling ranges

Anchored to GMR6000 Rb phase records (`reference/raw/6k*.txt`). Per α:

| α | log₁₀ h range | Regime |
|---|---------------|--------|
| +2 | [-19, -16]    | WPM; σ_y(1) ∈ [6×10⁻¹¹, 2×10⁻⁹]; Rb anchor ≈ -17.5 |
| +1 | [-28, -24]    | FPM proxy; 30% of samples include |
|  0 | [-23, -20]    | WFM; Rb anchor ≈ -21.3 |
| -1 | [-28, -25]    | FFM; flicker floor ∈ [4×10⁻¹⁵, 4×10⁻¹³] |
| -2 | [-34, -28]    | RWFM; Rb effectively zero, upper -28 caps rise |

Do not extend to TCXO ranges without explicit product sign-off — TCXO WPM
levels ≥ 1e-8 at τ=1 would dominate the label distribution.
