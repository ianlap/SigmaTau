"""retune_eval.py — Re-tune RF + XGB on expanded grids, regenerate figures/metrics.

Driven by the user's observation that the prior grid winners landed on multiple
boundary values. Expanded grids are centered on the prior winners with pushed-
out extremes so we can confirm the optimum is interior.

Runs:
  1. Load dataset_v1.h5, reproduce the same stratified split (seed=42).
  2. GridSearchCV(RF) on expanded grid.
  3. GridSearchCV(XGB) on expanded grid.
  4. Overwrite models/rf_best.joblib and models/xgb_best.joblib.
  5. Regenerate figures: pred_vs_actual_rf, residuals_rf, rf_top20_importance,
     importance_by_statistic, rf_uq, model_comparison_rmse.
  6. Write metrics.json with per-target rmse/r2/mae for naive/RF/XGB and the
     chosen hyperparameters.

Usage:
  cd ml && .venv/bin/python retune_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.src.loader import load_dataset, stratified_split
from ml.src.models import predict
from ml.src.evaluation import metrics_per_target, rf_prediction_variance

ML_DIR   = Path(__file__).parent
DATA_H5  = ML_DIR / "data" / "dataset_v1.h5"
MODELS   = ML_DIR / "models"
FIG_DIR  = ML_DIR / "figures"
MODELS.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

TARGETS = ("q0", "q1", "q2")

# Focused boundary-probe grid: holds non-boundary params at prior winners
# (RF max_depth=20, max_features=0.5; XGB max_depth=6), extends only the
# hyperparameters that had previously landed on grid edges.
RF_GRID = {
    "n_estimators":     [1000, 1500, 2000],
    "max_depth":        [20],
    "min_samples_leaf": [1, 2, 3],
    "max_features":     [0.5],
}
XGB_GRID = {
    "estimator__n_estimators":  [500, 750, 1000],
    "estimator__learning_rate": [0.003, 0.01],
    "estimator__max_depth":     [6],
    "estimator__subsample":     [0.7, 0.8],
}


def main():
    print(f"Loading {DATA_H5} ...")
    ds = load_dataset(str(DATA_H5))
    Xtr, Xte, ytr, yte, mtr, mte = stratified_split(ds, test_size=0.2, seed=42)
    print(f"Train: {Xtr.shape}   Test: {Xte.shape}")

    n_rf = int(np.prod([len(v) for v in RF_GRID.values()]))
    n_xgb = int(np.prod([len(v) for v in XGB_GRID.values()]))
    print(f"RF grid size: {n_rf} combos × 5-fold = {n_rf*5} fits")
    print(f"XGB grid size: {n_xgb} combos × 5-fold = {n_xgb*5} fits")

    # --- RF ---
    print("\n=== RF GridSearchCV ===", flush=True)
    rf_gs = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),
        RF_GRID, cv=5, scoring="neg_mean_squared_error",
        n_jobs=-1, verbose=2,
    )
    rf_gs.fit(Xtr, ytr)
    print("best RF params:", rf_gs.best_params_, flush=True)
    print(f"best RF CV score (neg-MSE): {rf_gs.best_score_:.6f}")
    rf = rf_gs.best_estimator_
    joblib.dump(rf, MODELS / "rf_best.joblib")

    # --- XGB ---
    print("\n=== XGB GridSearchCV ===", flush=True)
    xgb_base = MultiOutputRegressor(
        xgb.XGBRegressor(random_state=42, tree_method="hist", n_jobs=-1),
        n_jobs=1,
    )
    xgb_gs = GridSearchCV(
        xgb_base, XGB_GRID, cv=5, scoring="neg_mean_squared_error",
        n_jobs=2, verbose=2,
    )
    xgb_gs.fit(Xtr, ytr)
    print("best XGB params:", xgb_gs.best_params_, flush=True)
    print(f"best XGB CV score (neg-MSE): {xgb_gs.best_score_:.6f}")
    xgb_m = xgb_gs.best_estimator_
    joblib.dump(xgb_m, MODELS / "xgb_best.joblib")

    # --- Evaluate ---
    y_naive = np.tile(ytr.mean(axis=0), (yte.shape[0], 1))
    y_rf = rf.predict(Xte)
    y_xgb = predict(xgb_m, Xte)

    df_naive = metrics_per_target(yte, y_naive)
    df_rf    = metrics_per_target(yte, y_rf)
    df_xgb   = metrics_per_target(yte, y_xgb)
    print("\n=== Test metrics ===")
    print("Naive:\n", df_naive)
    print("\nRF:\n",    df_rf)
    print("\nXGB:\n",   df_xgb)

    # --- Figures ---
    # pred_vs_actual_rf
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, j, name in zip(axes, range(3), TARGETS):
        ax.scatter(yte[:, j], y_rf[:, j], s=8, c=mte.astype(int),
                   cmap="coolwarm", alpha=0.6)
        lims = [min(yte[:, j].min(), y_rf[:, j].min()),
                max(yte[:, j].max(), y_rf[:, j].max())]
        ax.plot(lims, lims, "k--")
        ax.set_xlabel(f"log10 {name} true")
        ax.set_ylabel("predicted")
        ax.set_title(name)
    fig.suptitle("Random Forest — predicted vs actual   (blue=no FPM, red=FPM)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pred_vs_actual_rf.png", dpi=150)
    plt.close(fig)

    # residuals_rf
    residuals = y_rf - yte
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, j, name in zip(axes, range(3), TARGETS):
        ax.hist(residuals[:, j], bins=60, color="C2", edgecolor="k")
        mu, sig = residuals[:, j].mean(), residuals[:, j].std()
        ax.set_title(f"{name}   μ={mu:.3f}  σ={sig:.3f}")
        ax.set_xlabel("residual (log10 decades)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residuals_rf.png", dpi=150)
    plt.close(fig)

    # rf_top20_importance
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
    plt.close(fig)

    # importance_by_statistic
    labels = ["ADEV", "MDEV", "HDEV", "MHDEV"]
    raw_by_stat   = np.array([imp[i*20:(i+1)*20].sum() for i in range(4)])
    slope_by_stat = np.array([imp[80 + 19*k:80 + 19*(k+1)].sum() for k in range(4)])
    ratio_mvar_avar  = imp[156:176].sum()
    ratio_mhvar_hvar = imp[176:196].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(4)
    ax.bar(x - 0.2, raw_by_stat,   width=0.4, label="raw")
    ax.bar(x + 0.2, slope_by_stat, width=0.4, label="slopes")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("total importance")
    ax.set_title("Importance aggregated by stability statistic")
    ax.legend()
    ax.text(0.02, 0.95,
            f"ratio(MVAR/AVAR)={ratio_mvar_avar:.3f}\nratio(MHVAR/HVAR)={ratio_mhvar_hvar:.3f}",
            transform=ax.transAxes, va="top", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "importance_by_statistic.png", dpi=150)
    plt.close(fig)

    # rf_uq (50 lowest-q samples)
    var_rf = rf_prediction_variance(rf, Xte)
    std_rf = np.sqrt(np.maximum(var_rf, 0))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, j, name in zip(axes, range(3), TARGETS):
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
    plt.close(fig)

    # model_comparison_rmse
    comp = pd.concat({"RF": df_rf["rmse"], "XGB": df_xgb["rmse"], "Naive": df_naive["rmse"]}, axis=1)
    comp.index = list(TARGETS)
    ax = comp.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("RMSE (log10 decades)")
    ax.set_title("RMSE by model and target")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison_rmse.png", dpi=150)
    plt.close()

    # --- Metrics JSON ---
    def _metrics_dict(df):
        out = {}
        for i, name in enumerate(TARGETS):
            out[name] = {k: float(df[k].iloc[i]) for k in df.columns}
        return out

    metrics = {
        "targets": list(TARGETS),
        "naive":   _metrics_dict(df_naive),
        "rf":      _metrics_dict(df_rf),
        "xgb":     _metrics_dict(df_xgb),
        "rf_best_params":  {k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
                            for k, v in rf_gs.best_params_.items()},
        "xgb_best_params": {k.replace("estimator__", ""):
                                (v if not isinstance(v, (np.integer, np.floating)) else float(v))
                            for k, v in xgb_gs.best_params_.items()},
        "rf_best_cv_neg_mse":  float(rf_gs.best_score_),
        "xgb_best_cv_neg_mse": float(xgb_gs.best_score_),
        "rf_grid":  {k: [v if not isinstance(v, np.generic) else v.item() for v in vs] for k, vs in RF_GRID.items()},
        "xgb_grid": {k.replace("estimator__", ""): vs for k, vs in XGB_GRID.items()},
    }
    out = ML_DIR / "retune_metrics.json"
    out.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nwrote {out.relative_to(ML_DIR)}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
