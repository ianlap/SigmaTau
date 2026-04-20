# ---
# Convert to notebook:  jupytext --to notebook ml/notebook.py
# Execute in place:     jupytext --execute --to notebook ml/notebook.py
# Or two-step:
#   jupytext --to notebook ml/notebook.py
#   cd ml && jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.executed.ipynb
# ---

# %% [markdown]
# # KF Parameter Prediction from Stability Curves
#
# PH 551 Final Project — Ian Lapinski
#
# Predicts Kalman-filter process-noise parameters `(q_wpm, q_wfm, q_rwfm)` from 196-feature
# stability-curve vectors. Trained on 10k synthetic samples drawn from distributions anchored
# to measured GMR6000 Rb oscillator data; validated on real Rb phase records.

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ml.src.loader import load_dataset, stratified_split, impute_median
from ml.src.models import predict
from ml.src.evaluation import metrics_per_target, rf_prediction_variance, xgb_quantile_intervals, coverage

sns.set_theme(context="talk", style="whitegrid")
np.random.seed(42)

# Switch between fixtures and full production dataset (prefers bigger)
for candidate in ("data/dataset_v1.h5", "data/dev_100.h5", "data/dev_25.h5"):
    if Path(candidate).exists():
        DATA_PATH = candidate
        break
else:
    raise FileNotFoundError("No dataset HDF5 found; run ml/dataset/run_test_dataset.jl first")

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
print("Using dataset:", DATA_PATH)

# %% [markdown]
# ## 1. Dataset & EDA

# %%
ds = load_dataset(DATA_PATH)
Xtr, Xte, ytr, yte, mtr, mte = stratified_split(ds, test_size=0.2, seed=42)
print(f"Train: {Xtr.shape}   Test: {Xte.shape}")
print(f"FPM ratio: train={mtr.mean():.3f}  test={mte.mean():.3f}  full={ds.fpm_present.mean():.3f}")
print(f"Converged: {ds.converged.sum()}/{len(ds.converged)}")

# %%
# Six-panel figure: 3 feature histograms (short/mid/long τ ADEV) + 3 target histograms
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
# Feature indices for raw_adev at short/mid/long τ (feature_names have format "raw_adev_m{m}")
feature_probes = [
    ("short τ=1 s",    "raw_adev_m1"),
    ("mid τ=31 s",     "raw_adev_m31"),
    ("long τ=4279 s",  "raw_adev_m4279"),
]
names_list = ds.feature_names.tolist()
for ax, (label, fname) in zip(axes.ravel()[:3], feature_probes):
    if fname in names_list:
        j = names_list.index(fname)
        vals = ds.X[:, j]
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=50, color="C0", edgecolor="k", alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("log10 σ_ADEV")
    else:
        ax.set_visible(False)

target_labels = ("log10 q_wpm", "log10 q_wfm", "log10 q_rwfm")
for ax, j, name in zip(axes.ravel()[3:], range(3), target_labels):
    ax.hist(ds.y[:, j], bins=50, color="C1", edgecolor="k", alpha=0.8)
    ax.set_title(name)
    ax.set_xlabel("value")
fig.tight_layout()
fig.savefig(FIG_DIR / "eda_distributions.png", dpi=150)
plt.show()

# %%
# 5 random samples, overlay ADEV/MDEV/HDEV/MHDEV for each
rng = np.random.default_rng(0)
picks = rng.choice(len(ds.X), size=min(5, len(ds.X)), replace=False)
taus = ds.taus
# Feature layout: 80 raw in order ADEV(20), MDEV(20), HDEV(20), MHDEV(20)
fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4), sharey=True)
if len(picks) == 1:
    axes = [axes]
stat_labels = ("adev", "mdev", "hdev", "mhdev")
for ax, idx in zip(axes, picks):
    for s_i, stat in enumerate(stat_labels):
        start = s_i * 20
        sigma = 10 ** ds.X[idx, start:start + 20]
        ax.loglog(taus, sigma, label=stat, alpha=0.8)
    ax.set_xlabel("τ (s)")
    ax.set_title(f"sample {idx}   FPM={ds.fpm_present[idx]}")
axes[0].set_ylabel("σ(τ)")
axes[0].legend(loc="lower left")
fig.tight_layout()
fig.savefig(FIG_DIR / "eda_example_curves.png", dpi=150)
plt.show()

# %%
# Pearson correlation of each feature (196) with each target (3)
corr = np.zeros((196, 3))
for j in range(196):
    col = ds.X[:, j]
    finite = np.isfinite(col)
    if finite.sum() < 3:
        continue
    for t in range(3):
        tgt = ds.y[finite, t]
        feat = col[finite]
        if np.std(feat) < 1e-12 or np.std(tgt) < 1e-12:
            continue
        corr[j, t] = np.corrcoef(feat, tgt)[0, 1]

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(corr.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_yticks(range(3))
ax.set_yticklabels(["q_wpm", "q_wfm", "q_rwfm"])
ax.set_xticks([])
ax.set_xlabel("feature index (0..195)")
fig.colorbar(im, ax=ax, label="Pearson r")
fig.tight_layout()
fig.savefig(FIG_DIR / "eda_corr_feature_target.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. Naive baseline

# %%
# Predict the train-set mean for each target; serves as a trivial floor
y_naive = np.tile(ytr.mean(axis=0), (yte.shape[0], 1))
print("Naive (predict train mean) metrics:")
print(metrics_per_target(yte, y_naive))

# %% [markdown]
# ## 3. Random Forest + XGBoost — tuning

# %%
# Adaptive GridSearchCV: on tiny fixtures use a reduced grid + fewer folds
# so the same code runs end-to-end on 25 or 10000 samples.
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib

n_train = Xtr.shape[0]
cv_folds = max(2, min(5, n_train // 10))  # at least 2-fold, at most 5-fold
is_big = n_train >= 1000

if is_big:
    rf_grid = {
        "n_estimators":     [500, 1000, 1500, 2000],
        "max_depth":        [15, 20, 25],
        "min_samples_leaf": [1, 2, 3],
        "max_features":     ["sqrt", 0.3, 0.5, 0.7],
    }
else:
    # Reduced grid for small fixtures — still exercises the methodology
    rf_grid = {
        "n_estimators":     [50, 100],
        "max_depth":        [None, 10],
        "min_samples_leaf": [1, 2],
        "max_features":     ["sqrt"],
    }

print(f"RF GridSearchCV: n_train={n_train}, cv={cv_folds}, grid_size={np.prod([len(v) for v in rf_grid.values()])}")
rf_gs = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=1),
    rf_grid, cv=cv_folds, scoring="neg_mean_squared_error",
    n_jobs=-1, verbose=1,
)
rf_gs.fit(Xtr, ytr)
print("best RF params:", rf_gs.best_params_)
joblib.dump(rf_gs.best_estimator_, MODEL_DIR / "rf_best.joblib")
rf = rf_gs.best_estimator_

# %%
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

if is_big:
    xgb_grid = {
        "estimator__n_estimators":  [300, 500, 750, 1000],
        "estimator__learning_rate": [0.003, 0.01, 0.03],
        "estimator__max_depth":     [5, 6, 7],
        "estimator__subsample":     [0.7, 0.8, 0.9],
    }
else:
    xgb_grid = {
        "estimator__n_estimators":  [50, 100],
        "estimator__learning_rate": [0.05, 0.1],
        "estimator__max_depth":     [3, 5],
        "estimator__subsample":     [1.0],
    }

print(f"XGB GridSearchCV: n_train={n_train}, cv={cv_folds}, grid_size={np.prod([len(v) for v in xgb_grid.values()])}")
xgb_base = MultiOutputRegressor(
    xgb.XGBRegressor(random_state=42, tree_method="hist", n_jobs=-1),
    n_jobs=1,
)
xgb_gs = GridSearchCV(
    xgb_base, xgb_grid, cv=cv_folds, scoring="neg_mean_squared_error",
    n_jobs=2, verbose=1,
)
xgb_gs.fit(Xtr, ytr)
print("best XGB params:", xgb_gs.best_params_)
joblib.dump(xgb_gs.best_estimator_, MODEL_DIR / "xgb_best.joblib")
xgb_m = xgb_gs.best_estimator_

# %% [markdown]
# ## 4. Evaluation

# %%
y_rf  = rf.predict(Xte)
y_xgb = predict(xgb_m, Xte)

print("RF metrics:");  print(metrics_per_target(yte, y_rf))
print("\nXGB metrics:"); print(metrics_per_target(yte, y_xgb))
print("\nNaive metrics:"); print(metrics_per_target(yte, y_naive))

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
target_names = ("q_wpm", "q_wfm", "q_rwfm")
for ax, j, name in zip(axes, range(3), target_names):
    ax.scatter(yte[:, j], y_rf[:, j], s=8,
               c=mte.astype(int), cmap="coolwarm", alpha=0.6)
    lims = [min(yte[:, j].min(), y_rf[:, j].min()),
            max(yte[:, j].max(), y_rf[:, j].max())]
    ax.plot(lims, lims, "k--")
    ax.set_xlabel(f"log10 {name} true")
    ax.set_ylabel("predicted")
    ax.set_title(name)
fig.suptitle("Random Forest — predicted vs actual   (blue=no FPM, red=FPM)")
fig.tight_layout()
fig.savefig(FIG_DIR / "pred_vs_actual_rf.png", dpi=150)
plt.show()

# %%
residuals = y_rf - yte
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, j, name in zip(axes, range(3), target_names):
    ax.hist(residuals[:, j], bins=60, color="C2", edgecolor="k")
    μ, σ = residuals[:, j].mean(), residuals[:, j].std()
    ax.set_title(f"{name}   μ={μ:.3f}  σ={σ:.3f}")
    ax.set_xlabel("residual (log10 decades)")
fig.tight_layout()
fig.savefig(FIG_DIR / "residuals_rf.png", dpi=150)
plt.show()

# %%
names = ds.feature_names
imp = rf.feature_importances_
top20 = np.argsort(imp)[-20:][::-1]
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(20)[::-1], imp[top20])
ax.set_yticks(range(20)[::-1])
ax.set_yticklabels(names[top20])
ax.set_xlabel("feature importance")
ax.set_title("Top-20 RF feature importances")
fig.tight_layout()
fig.savefig(FIG_DIR / "rf_top20_importance.png", dpi=150)
plt.show()

# %%
labels = ["ADEV", "MDEV", "HDEV", "MHDEV"]
raw_by_stat    = np.array([imp[i*20:(i+1)*20].sum() for i in range(4)])
slope_by_stat  = np.array([imp[80 + 19*k:80 + 19*(k+1)].sum() for k in range(4)])
# ratios (40 features total) are split 20:20 mvar/avar, mhvar/hvar at indices [156:176] and [176:196]
ratio_mvar_avar   = imp[156:176].sum()
ratio_mhvar_hvar  = imp[176:196].sum()

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(4)
ax.bar(x - 0.2, raw_by_stat,   width=0.4, label="raw")
ax.bar(x + 0.2, slope_by_stat, width=0.4, label="slopes")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("total importance")
ax.set_title("Importance aggregated by stability statistic")
ax.legend()
ax.text(0.02, 0.95, f"ratio(MVAR/AVAR)={ratio_mvar_avar:.3f}\nratio(MHVAR/HVAR)={ratio_mhvar_hvar:.3f}",
        transform=ax.transAxes, va="top", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "importance_by_statistic.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Uncertainty quantification

# %%
var_rf = rf_prediction_variance(rf, Xte)
std_rf = np.sqrt(np.maximum(var_rf, 0))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, j, name in zip(axes, range(3), target_names):
    order = np.argsort(y_rf[:, j])[:50]
    ax.errorbar(range(len(order)), y_rf[order, j],
                yerr=1.96 * std_rf[order, j],
                fmt=".", alpha=0.5, label="±95% CI")
    ax.scatter(range(len(order)), yte[order, j], c="red", s=8, label="true")
    ax.set_title(name)
    ax.legend()
fig.suptitle("RF prediction intervals — 50 lowest-q samples (tree-variance estimator)")
fig.tight_layout()
fig.savefig(FIG_DIR / "rf_uq.png", dpi=150)
plt.show()

# %%
# Fit 6 quantile XGB models (2 quantiles × 3 targets). On small fixtures use
# reduced n_estimators so the cell stays fast.
qxgb_n = 300 if is_big else 100
qxgb_depth = 6 if is_big else 4
lo, hi = xgb_quantile_intervals(
    Xtr, ytr, Xte,
    n_estimators=qxgb_n, max_depth=qxgb_depth,
)
cov = coverage(yte, lo, hi)
print("XGB 90% prediction-interval empirical coverage per target:")
for name, c in zip(target_names, cov):
    print(f"  {name}: {c:.3f}")

# %% [markdown]
# ## 6. Model comparison

# %%
df_rf  = metrics_per_target(yte, y_rf)
df_xgb = metrics_per_target(yte, y_xgb)
df_nv  = metrics_per_target(yte, y_naive)

comp = pd.concat({"RF": df_rf["rmse"], "XGB": df_xgb["rmse"], "Naive": df_nv["rmse"]}, axis=1)
ax = comp.plot(kind="bar", figsize=(9, 5))
ax.set_ylabel("RMSE (log10 decades)")
ax.set_title("RMSE by model and target")
plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison_rmse.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. MDEV vs MHDEV importance slice
#
# Secondary analysis: compare total feature importance contributed by MDEV vs MHDEV
# (both modified deviations with different drift rejection).

# %%
mdev_idx  = [i for i, n in enumerate(names) if "mdev" in n and "mhdev" not in n]
mhdev_idx = [i for i, n in enumerate(names) if "mhdev" in n]
imp_mdev  = imp[mdev_idx].sum()
imp_mhdev = imp[mhdev_idx].sum()
print(f"Σ importance   MDEV = {imp_mdev:.4f}   MHDEV = {imp_mhdev:.4f}")
print(f"Ratio MDEV/MHDEV = {imp_mdev / max(imp_mhdev, 1e-12):.3f}")

# %% [markdown]
# ## 8. Conclusions
#
# _(To be filled in by the user once the 10k run completes.)_
#
# Summary of results from final run:
# - Naive baseline RMSE: _fill in_
# - RF RMSE:             _fill in_
# - XGB RMSE:            _fill in_
# - XGB coverage @ 90%:  _fill in_

# %% [markdown]
# ## 9. Real-data validation — GMR6000 Rb
#
# Use the trained RF/XGB to predict process-noise parameters on windows of the
# real Rb phase record. We generate features via a Julia subprocess (the feature
# extractor lives in SigmaTau.jl, not Python).

# %%
import subprocess
import tempfile

from ml.src.real_data import load_phase_record, extract_windows

REF_FILE = Path("../reference/raw/6k27febunsteered.txt")
if REF_FILE.exists():
    ph_feb = load_phase_record(REF_FILE, tau0=1.0, override_unit="nanoseconds")
    windows = extract_windows(ph_feb, window_size=524_288, n_windows=4)
    print(f"Extracted {len(windows)} windows of {windows.shape[1]} samples each")
else:
    windows = None
    print(f"Reference file not found at {REF_FILE}; skipping real-data validation.")

# %%
# Compute features via Julia subprocess (SigmaTau.compute_feature_vector).
# We pass windows as an HDF5 file and get back a CSV of features.
# NOTE: h5py writes (n_windows, window_size) in C/row-major order;
#       Julia HDF5.jl reads column-major, so size() returns (window_size, n_windows).
F_real = None
if windows is not None:
    jl_script = '''
using Pkg
Pkg.activate(ARGS[1])
using SigmaTau
using HDF5
# Args: [project_dir, input_h5, output_csv]
in_path  = ARGS[2]
out_path = ARGS[3]
windows = HDF5.h5open(in_path, "r") do f
    f["windows"][]
end
# h5py wrote (n_windows, window_size) C-order; Julia reads column-major → (window_size, n_windows)
window_size, n_windows = size(windows)
features = Matrix{Float32}(undef, n_windows, 196)
Threads.@threads for i in 1:n_windows
    features[i, :] = Float32.(compute_feature_vector(view(windows, :, i), 1.0))
end
open(out_path, "w") do io
    for i in 1:n_windows
        for j in 1:196
            print(io, features[i, j])
            if j < 196; print(io, ","); end
        end
        println(io)
    end
end
'''
    import h5py
    jl_project = str((Path("..") / "ml" / "dataset").resolve())
    with tempfile.TemporaryDirectory() as td:
        in_h5 = Path(td) / "windows.h5"
        out_csv = Path(td) / "features.csv"
        with h5py.File(in_h5, "w") as f:
            f["windows"] = windows
        jl_file = Path(td) / "compute_features.jl"
        jl_file.write_text(jl_script)
        cmd = ["julia", f"--project={jl_project}", "--threads=auto",
               str(jl_file), jl_project, str(in_h5), str(out_csv)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)
        F_real = np.loadtxt(out_csv, delimiter=",").astype(np.float32)
    print(f"Computed features: {F_real.shape}")

# %%
if F_real is not None:
    F_imp = impute_median(F_real)
    pred_rf  = rf.predict(F_imp)
    pred_xgb = predict(xgb_m, F_imp)
    print("RF predictions (log10 q_wpm, q_wfm, q_rwfm) per window:")
    for i, p in enumerate(pred_rf):
        print(f"  window {i}: {p}")
    print("\nXGB predictions:")
    for i, p in enumerate(pred_xgb):
        print(f"  window {i}: {p}")

# %%
# Analytical ADEV from KF q parameters — Wu 2023 / real_data_fit.jl convention.
# For τ₀=1 s (f_h = 1/(2τ₀) = 0.5 Hz):
#   σ²_y,WPM(τ)  = 3·q_wpm / τ²
#   σ²_y,WFM(τ)  = q_wfm / τ
#   σ²_y,RWFM(τ) = q_rwfm · τ / 3
# This matches the formulas used by mhdev_fit, optimize_nll, and als_fit so all
# ADEV overlays in this notebook are numerically self-consistent.
def analytical_adev(q_wpm, q_wfm, q_rwfm, tau):
    tau = np.asarray(tau, dtype=float)
    sigma2 = (3.0 * q_wpm / tau**2
              + q_wfm / tau
              + q_rwfm * tau / 3.0)
    return np.sqrt(np.maximum(sigma2, 0.0))

if F_real is not None:
    fig, axes = plt.subplots(1, min(4, len(windows)), figsize=(5 * min(4, len(windows)), 5), sharey=True)
    if len(windows) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        sigma_measured = 10.0 ** F_real[i, 0:20]   # first 20 raw features are log10 ADEV
        ax.loglog(ds.taus, sigma_measured, "o-", label="measured", ms=4)
        q_pred = 10.0 ** pred_rf[i]
        sigma_theo = analytical_adev(q_pred[0], q_pred[1], q_pred[2], ds.taus)
        ax.loglog(ds.taus, sigma_theo, "--", label="RF prediction", lw=2)
        ax.set_title(f"window {i}")
        ax.set_xlabel("τ (s)")
    axes[0].set_ylabel("σ_y(τ)")
    axes[0].legend()
    fig.suptitle("Real GMR6000 — measured vs ML-predicted ADEV")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "real_data_overlay.png", dpi=150)
    plt.show()

# %% [markdown]
# ### 9b. Full method comparison on real data
#
# Per-window overlay of measured ADEV vs theoretical ADEV from six q-estimators:
#   - **Naive**        — train-set mean of log10(q) (baseline)
#   - **RF / XGB**     — ML predictions from 196-feature vector
#   - **MHDEV fit**    — three-region slope fit on MHDEV (WPM 1–10 s, WFM 30–1000 s, RWFM 1e4–1e5 s)
#   - **NLL**          — Kalman innovation-NLL optimum (warm-started from MHDEV fit)
#   - **ALS**          — Autocovariance Least-Squares fit (warm-started from MHDEV fit)
#
# MHDEV/NLL/ALS fits are produced by `ml/dataset/real_data_all_fits.jl` (run separately).

# %%
FITS_CSV  = Path("data/real_per_window_fits.csv")
ADEV_CSV  = Path("data/real_per_window_adev.csv")
if F_real is not None and FITS_CSV.exists() and ADEV_CSV.exists():
    fits_df = pd.read_csv(FITS_CSV)
    adev_df = pd.read_csv(ADEV_CSV)

    # Naive: train-set mean of log10(q) per target
    q_naive = 10.0 ** ytr.mean(axis=0)   # length-3 vector

    n_win = int(fits_df["window"].max()) + 1
    n_show = min(n_win, len(windows))
    fig, axes = plt.subplots(1, n_show, figsize=(5.2 * n_show, 5.2), sharey=True)
    if n_show == 1:
        axes = [axes]

    # dense τ grid for smooth theoretical curves
    tau_fine = np.logspace(0, 5, 60)

    methods_color = {
        "naive":    ("C7", ":",  1.3),
        "RF":       ("C1", "--", 1.8),
        "XGB":      ("C2", "--", 1.8),
        "MHDEV fit": ("C3", "-.", 1.8),
        "NLL":      ("C4", "-",  1.8),
        "ALS":      ("C5", "-",  1.8),
    }

    for i, ax in enumerate(axes):
        # Measured (long τ grid from Julia)
        m = adev_df[adev_df["window"] == i]
        ax.loglog(m["tau"], m["adev"], "o-", color="k", ms=4, lw=1.2,
                  label="measured ADEV (Julia)")

        row = fits_df[fits_df["window"] == i].iloc[0]

        # Build per-method q triples
        q_rf  = 10.0 ** pred_rf[i]
        q_xgb = 10.0 ** pred_xgb[i]
        q_map = {
            "naive":     q_naive,
            "RF":        q_rf,
            "XGB":       q_xgb,
            "MHDEV fit": np.array([row.mhf_qwpm, row.mhf_qwfm, row.mhf_qrwfm]),
            "NLL":       np.array([row.nll_qwpm, row.nll_qwfm, row.nll_qrwfm]),
            "ALS":       np.array([row.als_qwpm, row.als_qwfm, row.als_qrwfm]),
        }
        for label, q in q_map.items():
            if not np.all(np.isfinite(q)):
                continue
            sig = analytical_adev(q[0], q[1], q[2], tau_fine)
            c, ls, lw = methods_color[label]
            ax.loglog(tau_fine, sig, ls=ls, lw=lw, color=c, label=label)

        ax.set_title(f"window {i}")
        ax.set_xlabel("τ (s)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0].set_ylabel("σ_y(τ)")
    axes[0].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle("GMR6000 Rb — measured ADEV vs theoretical ADEV from 6 q-estimators")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "real_data_method_comparison.png", dpi=150)
    plt.show()

    # Print per-window q-values for the record
    print("\nPer-window q estimates (log10):")
    cols = ["method", "q_wpm", "q_wfm", "q_rwfm"]
    for i in range(n_show):
        print(f"\n  ── window {i} ──")
        row = fits_df[fits_df["window"] == i].iloc[0]
        q_map = {
            "naive":     ytr.mean(axis=0),
            "RF":        pred_rf[i],
            "XGB":       pred_xgb[i],
            "MHDEV fit": np.log10(np.maximum([row.mhf_qwpm, row.mhf_qwfm, row.mhf_qrwfm], 1e-99)),
            "NLL":       np.log10(np.maximum([row.nll_qwpm, row.nll_qwfm, row.nll_qrwfm], 1e-99)),
            "ALS":       np.log10(np.maximum([row.als_qwpm, row.als_qwfm, row.als_qrwfm], 1e-99)),
        }
        for name, vec in q_map.items():
            print(f"    {name:10s}  log10(q) = [{vec[0]:7.2f}, {vec[1]:7.2f}, {vec[2]:7.2f}]")
else:
    print("Skipping method comparison — missing real_per_window_fits.csv; "
          "run `julia --project=ml/dataset ml/dataset/real_data_all_fits.jl` first.")
