"""Real-data loader for GMR6000 phase records.

Handles unit detection (phase may be in seconds, nanoseconds, or similar),
window extraction for feature computation, and file I/O for the two
reference files.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


# Expected σ_y(1s) range for a GMR6000 Rb when phase is in seconds
_EXPECTED_ADEV_SECONDS = (1e-12, 1e-9)


def _adev_tau1(x: np.ndarray) -> float:
    """Estimate σ_y at τ=1 sample via second-differences of phase.

    ADEV² = ⟨(x[i+2] - 2 x[i+1] + x[i])²⟩ / 2  (for τ₀ = 1)
    """
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return float(np.sqrt(np.mean(d2**2) / 2.0))


def detect_units(x: np.ndarray, tau0: float = 1.0) -> tuple[str, float]:
    """Infer the unit of the phase column by matching ADEV(1s) to expected range.

    Returns (unit_name, scale_factor_to_seconds).
    """
    a = _adev_tau1(x)   # dimensionless if x is seconds, otherwise scaled
    candidates = [
        ("seconds",     1.0,  _EXPECTED_ADEV_SECONDS),
        ("nanoseconds", 1e-9, tuple(v * 1e9 for v in _EXPECTED_ADEV_SECONDS)),
        ("microseconds", 1e-6, tuple(v * 1e6 for v in _EXPECTED_ADEV_SECONDS)),
    ]
    for name, factor, (lo, hi) in candidates:
        # Allow a factor-of-3 slop on either side — Rb units may be noisier than spec
        if lo / 3.0 <= a <= hi * 3.0:
            return name, factor
    raise ValueError(
        f"Could not infer units: ADEV(1s)={a:.3e} is not within any expected range "
        f"for a Rb/HSO/OCXO phase record"
    )


def load_phase_record(path: str | Path, tau0: float = 1.0,
                      override_unit: str | None = None) -> np.ndarray:
    """Load an MJD/phase whitespace-delimited ASCII file; return phase in seconds.

    Lines starting with `#` or non-numeric are skipped. The τ₀ of the file
    must match the `tau0` argument; raises if it doesn't.

    If `override_unit` is provided (``"seconds"``, ``"nanoseconds"``, or
    ``"microseconds"``), skips auto-detection and uses that unit.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    mjd, ph = [], []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 2:
                continue
            try:
                mjd.append(float(toks[0]))
                ph.append(float(toks[1]))
            except ValueError:
                continue
    mjd = np.asarray(mjd, dtype=float)
    ph  = np.asarray(ph,  dtype=float)
    if len(mjd) < 3:
        raise ValueError(f"Too few samples in {path}")

    step_s = (mjd[1] - mjd[0]) * 86400.0
    if abs(step_s - tau0) > 0.01 * max(tau0, 1e-9):
        raise ValueError(
            f"τ₀ mismatch: file step = {step_s:.4f}s, expected {tau0:.4f}s"
        )

    if override_unit is not None:
        factors = {"seconds": 1.0, "nanoseconds": 1e-9, "microseconds": 1e-6}
        if override_unit not in factors:
            raise ValueError(f"Unknown override_unit: {override_unit}")
        factor = factors[override_unit]
        unit = override_unit
    else:
        unit, factor = detect_units(ph, tau0=tau0)

    print(f"[{path.name}] unit={unit}  factor={factor:g}  n={len(ph)}")
    return ph * factor


def extract_windows(x: np.ndarray, *, window_size: int = 524_288,
                    n_windows: int | None = None) -> np.ndarray:
    """Non-overlapping contiguous windows from the phase record.

    `window_size` defaults to 524288 (= 2^19) to match the synthetic N.
    If `n_windows` is None, returns every non-overlapping window the record
    can provide.
    """
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D phase vector; got shape {x.shape}")
    max_n = len(x) // window_size
    n = max_n if n_windows is None else min(n_windows, max_n)
    if n <= 0:
        raise ValueError(
            f"Record too short: len(x)={len(x)} < window_size={window_size}"
        )
    return np.stack([x[i*window_size:(i+1)*window_size] for i in range(n)])
