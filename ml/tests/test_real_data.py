"""Real-data loader tests. Uses synthetic phase for unit detection; tries
the actual reference file if present."""
from pathlib import Path

import numpy as np
import pytest

from ml.src.real_data import detect_units, extract_windows, load_phase_record


# Locate the reference file relative to the repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
REF_FEB = REPO_ROOT / "reference" / "raw" / "6k27febunsteered.txt"


def test_detect_units_synthetic_seconds():
    rng = np.random.default_rng(0)
    # Random-walk phase in seconds with σ_step=1e-11 → ADEV(1) ≈ 1e-11
    x = np.cumsum(1e-11 * rng.normal(size=131_072))
    unit, factor = detect_units(x, tau0=1.0)
    assert unit == "seconds"
    assert factor == 1.0


def test_detect_units_synthetic_ns():
    rng = np.random.default_rng(1)
    x = np.cumsum(1e-11 * rng.normal(size=131_072)) * 1e9  # same physics but in ns
    unit, factor = detect_units(x, tau0=1.0)
    assert unit == "nanoseconds"
    assert factor == 1e-9


def test_detect_units_raises_when_out_of_range():
    # ADEV way below any expected range → no candidate matches
    x = np.full(1024, 1e-30)  # zero variance essentially
    # use very small incremental noise to get a finite but tiny ADEV
    x = x + 1e-40 * np.arange(1024, dtype=float)
    with pytest.raises(ValueError):
        detect_units(x, tau0=1.0)


def test_extract_windows_shape():
    x = np.arange(10 * 131_072, dtype=float)
    wins = extract_windows(x, window_size=131_072, n_windows=5)
    assert wins.shape == (5, 131_072)
    assert wins[0, 0] == 0
    assert wins[1, 0] == 131_072


def test_extract_windows_all():
    x = np.arange(10 * 1024, dtype=float)
    wins = extract_windows(x, window_size=1024, n_windows=None)
    assert wins.shape == (10, 1024)


def test_extract_windows_too_short_raises():
    x = np.arange(100, dtype=float)
    with pytest.raises(ValueError):
        extract_windows(x, window_size=1024)


@pytest.mark.skipif(not REF_FEB.exists(),
                    reason="reference file not present")
def test_load_phase_record_feb():
    """Real-data smoke test: loads the 6k27feb file end-to-end."""
    ph = load_phase_record(REF_FEB, tau0=1.0)
    # File is in ns; post-scale ADEV(1) should be ~5e-10 (Rb WPM level)
    ad1 = np.sqrt(np.mean(np.diff(ph, n=2)**2) / 2.0)
    assert 1e-11 < ad1 < 1e-8
    assert len(ph) > 1_000_000
