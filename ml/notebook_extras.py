"""notebook_extras.py — Standalone script for the new ML-project analyses.

Runs two follow-ons to notebook.executed.ipynb without rerunning the 75-min
notebook. Loads the already-trained RF/XGB models and the per-window Julia
fits (mhdev_fit / optimize_nll / als_fit), then:

  A. Produces `figures/real_data_method_comparison.png` — measured ADEV vs
     theoretical ADEV from naive / RF / XGB / MHDEV / NLL / ALS per window.

  B. Writes `data/q_inits.csv` and kicks `warmstart_compare.jl` (Julia), then
     produces `figures/warmstart_compare.png` — NLL iterations and holdover
     RMS under naive vs ML warm starts.

Usage:
  cd ml && .venv/bin/python notebook_extras.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.src.loader import load_dataset, stratified_split, impute_median
from ml.src.models import predict
from ml.src.real_data import load_phase_record, extract_windows

ML_DIR   = Path(__file__).parent
DATA_H5  = ML_DIR / "data" / "dataset_v1.h5"
MODELS   = ML_DIR / "models"
FIG_DIR  = ML_DIR / "figures"
DATA_DIR = ML_DIR / "data"
REF_FILE = ML_DIR.parent / "reference" / "raw" / "6k27febunsteered.txt"

FIG_DIR.mkdir(exist_ok=True)

# ── Wu 2023 convention ──────────────────────────────────────────────────────
def analytical_adev(q_wpm: float, q_wfm: float, q_rwfm: float, tau):
    tau = np.asarray(tau, dtype=float)
    s2 = 3.0 * q_wpm / tau**2 + q_wfm / tau + q_rwfm * tau / 3.0
    return np.sqrt(np.maximum(s2, 0.0))


def compute_features_via_julia(windows: np.ndarray) -> np.ndarray:
    jl_project = str((ML_DIR / "dataset").resolve())
    jl_script = '''
using Pkg; Pkg.activate(ARGS[1])
using SigmaTau, HDF5
windows = HDF5.h5open(ARGS[2], "r") do f; f["windows"][]; end
window_size, n_windows = size(windows)
features = Matrix{Float32}(undef, n_windows, 196)
Threads.@threads for i in 1:n_windows
    features[i, :] = Float32.(compute_feature_vector(view(windows, :, i), 1.0))
end
open(ARGS[3], "w") do io
    for i in 1:n_windows
        for j in 1:196
            print(io, features[i, j]); j < 196 && print(io, ",")
        end
        println(io)
    end
end
'''
    with tempfile.TemporaryDirectory() as td:
        in_h5 = Path(td) / "windows.h5"
        out_csv = Path(td) / "feat.csv"
        with h5py.File(in_h5, "w") as f:
            f["windows"] = windows
        jl = Path(td) / "run.jl"; jl.write_text(jl_script)
        subprocess.run(
            ["julia", f"--project={jl_project}", "--threads=auto",
             str(jl), jl_project, str(in_h5), str(out_csv)],
            check=True, capture_output=True,
        )
        return np.loadtxt(out_csv, delimiter=",").astype(np.float32)


def phaseA_method_comparison(ds, ytr, rf, xgb_m, pred_rf, pred_xgb, F_real, windows):
    fits_csv = DATA_DIR / "real_per_window_fits.csv"
    adev_csv = DATA_DIR / "real_per_window_adev.csv"
    if not (fits_csv.exists() and adev_csv.exists()):
        print("Missing real_per_window_*.csv — run real_data_all_fits.jl first.")
        return None
    fits_df = pd.read_csv(fits_csv)
    adev_df = pd.read_csv(adev_csv)

    q_naive = 10.0 ** ytr.mean(axis=0)   # shape (3,)

    n_win = int(fits_df["window"].max()) + 1
    n_show = min(n_win, len(windows))
    fig, axes = plt.subplots(1, n_show, figsize=(5.2 * n_show, 5.2), sharey=True)
    if n_show == 1:
        axes = [axes]
    tau_fine = np.logspace(0, 5, 60)
    spec = {
        "naive":     ("C7", ":",  1.3),
        "RF":        ("C1", "--", 1.8),
        "XGB":       ("C2", "--", 1.8),
        "MHDEV fit": ("C3", "-.", 1.8),
        "ALS":       ("C5", "-",  1.8),
    }
    for i, ax in enumerate(axes):
        m = adev_df[adev_df["window"] == i]
        ax.loglog(m["tau"], m["adev"], "o-", color="k", ms=4, lw=1.2,
                  label="measured ADEV")
        row = fits_df[fits_df["window"] == i].iloc[0]
        q_rf  = 10.0 ** pred_rf[i]
        q_xgb = 10.0 ** pred_xgb[i]
        q_map = {
            "naive":     q_naive,
            "RF":        q_rf,
            "XGB":       q_xgb,
            "MHDEV fit": np.array([row.mhf_qwpm, row.mhf_qwfm, row.mhf_qrwfm]),
            "ALS":       np.array([row.als_qwpm, row.als_qwfm, row.als_qrwfm]),
        }
        for label, q in q_map.items():
            if not np.all(np.isfinite(q)):
                continue
            sig = analytical_adev(q[0], q[1], q[2], tau_fine)
            c, ls, lw = spec[label]
            ax.loglog(tau_fine, sig, ls=ls, lw=lw, color=c, label=label)
        ax.set_title(f"window {i}")
        ax.set_xlabel(r"$\tau$ (s)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0].set_ylabel(r"$\sigma_y(\tau)$")
    axes[0].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle("GMR6000 Rb — measured ADEV vs theoretical ADEV from 5 q-estimators")
    fig.tight_layout()
    out = FIG_DIR / "real_data_method_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(ML_DIR)}")

    print("\nPer-window log10(q) estimates:")
    for i in range(n_show):
        row = fits_df[fits_df["window"] == i].iloc[0]
        table = {
            "naive":     ytr.mean(axis=0),
            "RF":        pred_rf[i],
            "XGB":       pred_xgb[i],
            "MHDEV fit": np.log10(np.maximum([row.mhf_qwpm, row.mhf_qwfm, row.mhf_qrwfm], 1e-99)),
            "ALS":       np.log10(np.maximum([row.als_qwpm, row.als_qwfm, row.als_qrwfm], 1e-99)),
        }
        print(f"  ── window {i} ──")
        for k, v in table.items():
            print(f"    {k:10s}  [{v[0]:7.2f}, {v[1]:7.2f}, {v[2]:7.2f}]")
    return fits_df


def phaseB_holdover_trajectories(ytr, pred_rf, pred_xgb):
    """Trajectory-averaged 1-day holdover RMS for naive / RF / XGB q triples."""
    q_naive = 10.0 ** ytr.mean(axis=0)     # (3,)
    n_win = len(pred_rf)

    q_methods = DATA_DIR / "q_methods.csv"
    with q_methods.open("w") as f:
        f.write("window,method,q_wpm,q_wfm,q_rwfm\n")
        for i in range(n_win):
            q_rf  = 10.0 ** pred_rf[i]
            q_xgb = 10.0 ** pred_xgb[i]
            rows = [
                ("naive", q_naive),
                ("RF",    q_rf),
                ("XGB",   q_xgb),
            ]
            for name, q in rows:
                f.write(f"{i},{name},{q[0]:.6e},{q[1]:.6e},{q[2]:.6e}\n")
    print(f"wrote {q_methods.relative_to(ML_DIR)}")

    print("Running holdover_trajectories.jl ...")
    jl_project = str((ML_DIR / "dataset").resolve())
    subprocess.run(
        ["julia", f"--project={jl_project}", "--threads=auto",
         str(ML_DIR / "dataset" / "holdover_trajectories.jl")],
        check=True,
    )

    traj = pd.read_csv(DATA_DIR / "holdover_trajectories.csv")
    print("\nHoldover RMS summary (1-day horizon, trajectory-averaged):")
    print(traj.to_string(index=False))

    # grouped bars: per window, one bar per method; error bars = std across
    # trajectory starts
    methods = ["naive", "RF", "XGB"]
    colors  = {"naive": "C7", "RF": "C1", "XGB": "C2"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    wins = sorted(traj["window"].unique())
    xp = np.arange(len(wins))
    width = 0.27
    for j, m in enumerate(methods):
        ys, errs = [], []
        for w in wins:
            row = traj[(traj["window"] == w) & (traj["method"] == m)]
            if len(row) == 0:
                ys.append(np.nan); errs.append(0.0)
                continue
            ys.append(float(row.rms_mean.iloc[0]))
            errs.append(float(row.rms_std.iloc[0]))
        ax.bar(xp + (j - 1) * width, ys, width, yerr=errs,
               label=m, color=colors[m], capsize=3)
    ax.set_xticks(xp); ax.set_xticklabels(wins)
    ax.set_xlabel("window")
    ax.set_ylabel("1-day holdover phase RMS (s)")
    ax.set_title("Trajectory-averaged 1-day holdover RMS — naive / RF / XGB\n"
                 f"(stride-sampled starts across each 524 288-sample window; log-scale)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "warmstart_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(ML_DIR)}")


def main():
    print("Loading dataset + models ...")
    ds = load_dataset(str(DATA_H5))
    Xtr, Xte, ytr, yte, mtr, mte = stratified_split(ds, test_size=0.2, seed=42)
    rf    = joblib.load(MODELS / "rf_best.joblib")
    xgb_m = joblib.load(MODELS / "xgb_best.joblib")

    print("Loading & windowing real Rb record ...")
    ph  = load_phase_record(REF_FILE, tau0=1.0, override_unit="nanoseconds")
    windows = extract_windows(ph, window_size=524_288, n_windows=4)
    print(f"{len(windows)} windows × {windows.shape[1]} samples")

    print("Computing 196-feature vectors via Julia ...")
    F_real = compute_features_via_julia(windows)
    F_imp  = impute_median(F_real)
    pred_rf  = rf.predict(F_imp)
    pred_xgb = predict(xgb_m, F_imp)

    print("\n── Phase A: method comparison plot ──")
    phaseA_method_comparison(ds, ytr, rf, xgb_m, pred_rf, pred_xgb, F_real, windows)

    print("\n── Phase B: trajectory-averaged 1-day holdover RMS ──")
    phaseB_holdover_trajectories(ytr, pred_rf, pred_xgb)


if __name__ == "__main__":
    main()
