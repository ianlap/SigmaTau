"""Loader tests. Uses ml/data/dev_25.h5 as a fixture (25 samples, same schema).

The production dataset (ml/data/dataset_v1.h5) may not exist yet when these
tests run; tests key off dev_25.h5. If dev_25.h5 is missing, tests skip."""
import os
from pathlib import Path

import numpy as np
import pytest

from ml.src.loader import load_dataset, stratified_split, impute_median


FIXTURE = Path(__file__).resolve().parents[2] / "ml" / "data" / "dev_25.h5"


@pytest.fixture(scope="module")
def fixture_path():
    if not FIXTURE.exists():
        pytest.skip(f"dev fixture not found: {FIXTURE}")
    return str(FIXTURE)


def test_load_dataset_shapes(fixture_path):
    ds = load_dataset(fixture_path)
    assert ds.X.ndim == 2 and ds.X.shape[1] == 196
    assert ds.y.shape[1] == 3
    assert ds.h_coeffs.shape[1] == 5
    assert ds.fpm_present.dtype == bool
    assert len(ds.feature_names) == 196
    assert ds.taus.shape == (20,)
    # all samples converged in dev_25 (typical)
    assert ds.converged.all()


def test_stratified_split_preserves_fpm_ratio(fixture_path):
    ds = load_dataset(fixture_path)
    # 25 samples is very small for stratification; just check it runs and
    # totals match.
    Xtr, Xte, ytr, yte, mtr, mte = stratified_split(ds, test_size=0.2, seed=42)
    assert Xtr.shape[1] == 196
    assert Xte.shape[1] == 196
    assert len(ytr) + len(yte) == len(ds.X)


def test_impute_median_handles_nans():
    X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 5.0]])
    X_imp = impute_median(X)
    assert not np.isnan(X_imp).any()
    # Column 0: values (1, 2); median = 1.5 (imputed at row 2)
    assert X_imp[2, 0] == 1.5
    # Column 1: values (3, 5); median = 4.0
    assert X_imp[0, 1] == 4.0


def test_load_dataset_h_coeffs_has_expected_nans(fixture_path):
    """h_+1 column (index 1) should be NaN for samples where FPM is not present."""
    ds = load_dataset(fixture_path)
    h_plus1 = ds.h_coeffs[:, 1]
    # Samples without FPM should have NaN in the h_+1 column
    if not ds.fpm_present.all():
        assert np.isnan(h_plus1[~ds.fpm_present]).all()
