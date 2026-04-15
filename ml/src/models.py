"""Random Forest and XGBoost training wrappers."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


def train_rf(X, y, *,
             n_estimators: int = 500,
             max_depth: int | None = None,
             min_samples_leaf: int = 5,
             max_features: str | float = "sqrt",
             n_jobs: int = -1,
             seed: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def train_xgb(X, y, *,
              n_estimators: int = 500,
              max_depth: int = 6,
              learning_rate: float = 0.05,
              subsample: float = 1.0,
              n_jobs: int = -1,
              seed: int = 42) -> MultiOutputRegressor:
    base = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        n_jobs=n_jobs,
        random_state=seed,
        tree_method="hist",
    )
    model = MultiOutputRegressor(base, n_jobs=1)  # base already uses all threads
    model.fit(X, y)
    return model


def predict(model, X) -> np.ndarray:
    return np.asarray(model.predict(X))
