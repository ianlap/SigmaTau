import numpy as np
import pytest
from ml.src.evaluation import metrics_per_target, rf_prediction_variance, xgb_quantile_intervals, coverage


def test_metrics_per_target_shape_and_content():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(100, 3))
    y_pred = y_true + 0.1 * rng.normal(size=(100, 3))
    m = metrics_per_target(y_true, y_pred, target_names=("a", "b", "c"))
    for k in ("rmse", "mae", "r2"):
        assert k in m.columns
    assert set(m.index) == {"a", "b", "c"}
    # r2 should be high when prediction ≈ truth + small noise
    assert all(m["r2"] > 0.9)


def test_rf_variance_shape_and_positive():
    from ml.src.models import train_rf
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 20)).astype(np.float32)
    y = X @ rng.normal(size=(20, 3))
    model = train_rf(X[:150], y[:150], n_estimators=40, max_depth=6, min_samples_leaf=2)
    var = rf_prediction_variance(model, X[150:], X[:150])
    assert var.shape == (50, 3)
    assert (var >= 0).all()


def test_xgb_quantile_intervals_monotone_and_bracket():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 10)).astype(np.float32)
    y = X @ rng.normal(size=(10, 3))
    lo, hi = xgb_quantile_intervals(
        X[:150], y[:150], X[150:],
        alpha_low=0.05, alpha_high=0.95,
        n_estimators=40, max_depth=3,
    )
    assert lo.shape == hi.shape == (50, 3)
    assert (hi >= lo).all()


def test_coverage_correctness():
    y_true = np.array([[0.5], [1.5], [2.5]])
    lo     = np.array([[0.0], [1.0], [3.0]])
    hi     = np.array([[1.0], [2.0], [4.0]])
    cov = coverage(y_true, lo, hi)
    # Rows 0 & 1 covered, row 2 not; expected coverage 2/3
    assert cov[0] == pytest.approx(2 / 3)
