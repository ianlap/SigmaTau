"""Model-wrapper smoke tests with tiny synthetic data."""
import numpy as np
from ml.src.models import train_rf, train_xgb, predict


def _tiny_xy(n=200, d=196, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    # Targets: linear combinations of a few input features
    y = np.column_stack([X[:, 0], X[:, 1] - X[:, 5], X[:, 2] + X[:, 10]])
    return X, y


def test_train_rf_runs_and_predicts():
    X, y = _tiny_xy()
    model = train_rf(X[:150], y[:150], n_estimators=50, max_depth=6, min_samples_leaf=2)
    pred = predict(model, X[150:])
    assert pred.shape == (50, 3)
    for j in range(3):
        r = np.corrcoef(pred[:, j], y[150:, j])[0, 1]
        assert r > 0.3, f"RF target {j} correlation {r:.3f} below threshold"


def test_train_xgb_runs_and_predicts():
    X, y = _tiny_xy()
    model = train_xgb(X[:150], y[:150], n_estimators=50, max_depth=4)
    pred = predict(model, X[150:])
    assert pred.shape == (50, 3)
    for j in range(3):
        r = np.corrcoef(pred[:, j], y[150:, j])[0, 1]
        assert r > 0.3, f"XGB target {j} correlation {r:.3f} below threshold"
