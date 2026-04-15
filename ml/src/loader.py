"""Dataset loader for the SigmaTau ML pipeline.

Reads HDF5 files produced by ml/dataset/generate_dataset.jl.
Schema:
  /features/X              float32  (n, 196)
  /labels/q_log10          float64  (n, 3)
  /labels/h_log10          float64  (n, 5)
  /labels/fpm_present      uint8    (n,)
  /diagnostics/nll_values  float64  (n,)
  /diagnostics/converged   uint8    (n,)
  /meta/taus               float64  (20,)
  /meta/feature_names      strings  (196,)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    X: np.ndarray                # (n, 196) float32
    y: np.ndarray                # (n, 3)   float64 — log10(q_wpm, q_wfm, q_rwfm)
    h_coeffs: np.ndarray         # (n, 5)   float64 — log10(h+2, h+1, h0, h-1, h-2), NaN where absent
    fpm_present: np.ndarray      # (n,)     bool
    nll_values: np.ndarray       # (n,)     float64
    converged: np.ndarray        # (n,)     bool
    feature_names: np.ndarray    # (196,)   str
    taus: np.ndarray             # (20,)    float64


def _maybe_transpose(arr: np.ndarray, expected_inner: int) -> np.ndarray:
    """Julia HDF5.jl writes column-major; h5py reads row-major. For a Julia
    Matrix(n_samples, n_inner) with n_inner small (e.g. 196 features, 3 q-labels,
    5 h-coeffs), h5py sees it as (n_inner, n_samples). Detect and transpose
    when the leading dimension matches `expected_inner`."""
    if arr.ndim == 2 and arr.shape[0] == expected_inner and arr.shape[1] != expected_inner:
        return arr.T
    return arr


def load_dataset(path: str | Path, filter_unconverged: bool = True) -> Dataset:
    """Load an HDF5 dataset and optionally drop samples that didn't converge."""
    with h5py.File(str(path), "r") as f:
        X            = f["features/X"][...]
        y            = f["labels/q_log10"][...]
        h            = f["labels/h_log10"][...]
        fpm_u8       = f["labels/fpm_present"][...]
        nll          = f["diagnostics/nll_values"][...]
        conv_u8      = f["diagnostics/converged"][...]
        taus         = f["meta/taus"][...]
        feat_names_b = f["meta/feature_names"][...]

    # Variable-length strings in h5py may come back as bytes; decode
    if feat_names_b.dtype.kind in ("S", "O"):
        feature_names = np.array(
            [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in feat_names_b],
            dtype=str,
        )
    else:
        feature_names = feat_names_b.astype(str)

    # Correct for Julia → h5py axis-order swap on matrices
    X = _maybe_transpose(X, expected_inner=len(feature_names))   # (n, 196)
    y = _maybe_transpose(y, expected_inner=3)                     # (n, 3)
    h = _maybe_transpose(h, expected_inner=5)                     # (n, 5)

    ds = Dataset(
        X=X.astype(np.float32),
        y=y.astype(np.float64),
        h_coeffs=h.astype(np.float64),
        fpm_present=fpm_u8.astype(bool),
        nll_values=nll.astype(np.float64),
        converged=conv_u8.astype(bool),
        feature_names=feature_names,
        taus=taus.astype(np.float64),
    )

    if filter_unconverged and not ds.converged.all():
        ok = ds.converged
        ds = Dataset(
            X=ds.X[ok], y=ds.y[ok], h_coeffs=ds.h_coeffs[ok],
            fpm_present=ds.fpm_present[ok], nll_values=ds.nll_values[ok],
            converged=ds.converged[ok], feature_names=ds.feature_names,
            taus=ds.taus,
        )
    return ds


def impute_median(X: np.ndarray) -> np.ndarray:
    """Column-median NaN imputation. Returns a copy; leaves input untouched."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
            X[:, j] = col
    return X


def stratified_split(ds: Dataset, test_size: float = 0.2, seed: int = 42):
    """Train/test split stratified by ds.fpm_present. NaN imputation applied per split.

    Returns:
      Xtr, Xte, ytr, yte, mtr, mte
    """
    idx = np.arange(len(ds.X))
    idx_tr, idx_te = train_test_split(
        idx, test_size=test_size, stratify=ds.fpm_present, random_state=seed
    )
    Xtr = impute_median(ds.X[idx_tr])
    Xte = impute_median(ds.X[idx_te])
    return Xtr, Xte, ds.y[idx_tr], ds.y[idx_te], ds.fpm_present[idx_tr], ds.fpm_present[idx_te]
