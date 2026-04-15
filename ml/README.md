# ML Pipeline — Kalman Filter Parameter Prediction

Predicts Kalman-filter process-noise parameters `(q_wpm, q_wfm, q_rwfm)` from
frequency-stability curves. Trained on 10,000 synthetic samples drawn from
distributions anchored to measured GMR6000 Rb oscillator data; validated on
real Rb phase records.

**PH 551 Final Project — Ian Lapinski**

## Pipeline at a glance

1. **Julia: composite power-law noise generator** (Kasdin 1992) with absolute
   h_α amplitudes, validated against SP1065 Table 5 formulas.
2. **Julia: 196-feature extractor** — 80 raw σ values (4 deviations × 20 τ)
   + 76 log-log slopes + 40 variance ratios (MVAR/AVAR and MHVAR/HVAR).
3. **Julia: 3-D NLL labeller** — Kalman filter with DARE steady-state P₀
   init, jointly optimizes `(q_wpm, q_wfm, q_rwfm)` via Nelder-Mead in log₁₀
   space with analytical h-warm start.
4. **Python: RF + XGBoost regressors** trained with GridSearchCV; UQ via
   RF tree-variance and XGBoost quantile regression.
5. **Real-data validation** on GMR6000 Rb phase records with ADEV overlay.

## h_α sampling ranges

Anchored to the measured 6k27febunsteered Rb file (`h_+2 ≈ 10⁻¹⁷·⁶`,
`h_0 ≈ 10⁻²¹·³`, flicker floor ≈ 3×10⁻¹³). Ranges centered on Rb with
modest spread to cover HSO/OCXO-class oscillators:

| α  | Log₁₀ range | Regime |
|----|-------------|--------|
| +2 | [-19, -16]  | WPM; σ_y(1s) ∈ [6×10⁻¹¹, 2×10⁻⁹] |
| +1 | [-28, -24]  | FPM proxy |
|  0 | [-23, -20]  | WFM; σ_y(1s) ∈ [7×10⁻¹³, 2×10⁻¹¹] |
| -1 | [-28, -25]  | FFM; flicker floor ∈ [4×10⁻¹⁵, 4×10⁻¹³] |
| -2 | [-34, -28]  | RWFM; Rb anchor near -30 (effectively zero) |

## Reproducing the full pipeline

### Prerequisites

- Julia 1.8+
- Python 3.10+
- 12-core machine recommended for the 10k run (~2 hr)
- Reference files `reference/raw/6k27febunsteered.txt` (nanoseconds)

### Step 1 — Julia side

```bash
cd julia && julia --project=. -e 'using Pkg; Pkg.test()'
```

All 207 tests should pass.

### Step 2 — Generate the 10k dataset

```bash
cd ml/dataset
julia --project=. -e 'using Pkg; Pkg.develop(path="../../julia"); Pkg.instantiate()'
julia --project=. --threads=12 run_production.jl 2>&1 | tee ../data/dataset_v1.log
```

Runtime: ~2 hr on 12 threads. Checkpoints every 500 samples; safe to
Ctrl-C and restart with `resume=true`. Output: `ml/data/dataset_v1.h5`.

**Optional:** generate a 100-sample test dataset (~6 s) for quick
notebook iteration:

```bash
julia --project=. --threads=12 run_test_dataset.jl
```

### Step 3 — Python side

```bash
cd ml && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run tests (uses `ml/data/dev_25.h5` as fixture):

```bash
cd ..  # back to repo root
PYTHONPATH=. pytest ml/tests/ -v
```

### Step 4 — Notebook

Convert the jupytext `.py` to a notebook and execute:

```bash
cd ml
jupytext --to notebook notebook.py
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.executed.ipynb
```

Or run in place:

```bash
jupytext --execute --to notebook ml/notebook.py
```

The notebook auto-detects which dataset is available (prefers
`dataset_v1.h5` > `dev_100.h5` > `dev_25.h5`) and adapts grid sizes /
CV folds accordingly, so it runs end-to-end on any of them.

## Deliverables

- `ml/data/dataset_v1.h5` — synthetic dataset (10k samples, 196 features, 3 labels)
- `ml/models/rf_best.joblib`, `ml/models/xgb_best.joblib` — tuned models
- `ml/notebook.ipynb` — complete analysis
- `ml/figures/` — EDA, residual, importance, UQ, real-data overlay plots

## File layout

```
ml/
├── README.md                  # this file
├── requirements.txt           # Python deps
├── notebook.py                # jupytext notebook (convert with jupytext)
├── src/
│   ├── loader.py              # HDF5 dataset loader
│   ├── models.py              # RF/XGB wrappers
│   ├── evaluation.py          # metrics + UQ
│   └── real_data.py           # GMR6000 loader + unit detection
├── tests/
│   ├── test_loader.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_real_data.py
├── dataset/                   # Julia side
│   ├── Project.toml
│   ├── generate_dataset.jl    # main dataset module
│   ├── run_production.jl      # 10k driver
│   ├── run_test_dataset.jl    # 100-sample driver
│   ├── test_driver_mini.jl
│   ├── real_data_fit.jl       # diagnostic: fit real Rb file
│   ├── real_data_fit_file2.jl
│   └── dev_25_run.jl
├── data/                      # gitignored — HDF5 datasets and CSVs
└── figures/                   # gitignored — generated plots
```

## Dataset HDF5 schema

```
/features/X                   float32  (n, 196)
/labels/q_log10               float64  (n, 3)   — log10(q_wpm, q_wfm, q_rwfm)
/labels/h_log10               float64  (n, 5)   — log10(h_+2, h_+1, h_0, h_-1, h_-2)
/labels/fpm_present           uint8    (n,)
/diagnostics/nll_values       float64  (n,)
/diagnostics/converged        uint8    (n,)
/meta/taus                    float64  (20,)
/meta/feature_names           string   (196,)
/meta/n_done                  int      scalar
/meta/n_samples_total         int      scalar
```
