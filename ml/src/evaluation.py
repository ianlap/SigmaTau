"""Metrics, uncertainty quantification, and model comparison."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


def metrics_per_target(y_true: np.ndarray, y_pred: np.ndarray,
                       target_names=("q_wpm", "q_wfm", "q_rwfm")) -> pd.DataFrame:
    """Per-target RMSE, MAE, R²."""
    rows = []
    for j, name in enumerate(target_names):
        rows.append({
            "target": name,
            "rmse":   float(np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))),
            "mae":    float(mean_absolute_error(y_true[:, j], y_pred[:, j])),
            "r2":     float(r2_score(y_true[:, j], y_pred[:, j])),
        })
    return pd.DataFrame(rows).set_index("target")


def rf_prediction_variance(model, X_new: np.ndarray, X_train: np.ndarray = None) -> np.ndarray:
    """Per-target prediction variance via tree disagreement.

    Each tree of a RandomForestRegressor predicts all targets; variance across
    trees is a simple, well-behaved uncertainty estimator that handles
    multi-output models natively (unlike IJ-style approaches).
    """
    # Shape: (n_trees, n_new, n_targets)
    all_preds = np.stack([t.predict(X_new) for t in model.estimators_], axis=0)
    return all_preds.var(axis=0, ddof=1)


def xgb_quantile_intervals(X_tr, y_tr, X_te, *,
                           alpha_low: float = 0.05,
                           alpha_high: float = 0.95,
                           n_estimators: int = 300,
                           max_depth: int = 6,
                           learning_rate: float = 0.05,
                           seed: int = 42):
    """Fit one XGBoost quantile regressor per (target, quantile) → (low, high)."""
    n_targets = y_tr.shape[1]
    lo = np.empty((X_te.shape[0], n_targets))
    hi = np.empty((X_te.shape[0], n_targets))
    for j in range(n_targets):
        for q, out in ((alpha_low, lo), (alpha_high, hi)):
            m = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective="reg:quantileerror",
                quantile_alpha=q,
                random_state=seed + int(100 * q),
                tree_method="hist",
            )
            m.fit(X_tr, y_tr[:, j])
            out[:, j] = m.predict(X_te)
    return lo, hi


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Empirical fraction of y_true falling within [lo, hi], per target."""
    return ((y_true >= lo) & (y_true <= hi)).mean(axis=0)
