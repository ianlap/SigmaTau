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


def phaseB_warmstart(ytr, pred_rf):
    q_naive = 10.0 ** ytr.mean(axis=0)
    n_win = len(pred_rf)
    inits = DATA_DIR / "q_inits.csv"
    with inits.open("w") as f:
        f.write("window,naive_qwpm,naive_qwfm,naive_qrwfm,ml_qwpm,ml_qwfm,ml_qrwfm\n")
        for i in range(n_win):
            q_ml = 10.0 ** pred_rf[i]
            f.write(f"{i},{q_naive[0]:.6e},{q_naive[1]:.6e},{q_naive[2]:.6e},"
                    f"{q_ml[0]:.6e},{q_ml[1]:.6e},{q_ml[2]:.6e}\n")
    print(f"wrote {inits.relative_to(ML_DIR)}")

    print("Running warmstart_compare.jl ...")
    jl_project = str((ML_DIR / "dataset").resolve())
    subprocess.run(
        ["julia", f"--project={jl_project}", "--threads=auto",
         str(ML_DIR / "dataset" / "warmstart_compare.jl")],
        check=True,
    )

    cmp = pd.read_csv(DATA_DIR / "warmstart_compare.csv")
    print("\nWarm-start comparison:")
    print(cmp.to_string(index=False))
    print(f"\nMean speedup  (N_iter_naive / N_iter_ml):   {cmp.speedup_iter.mean():.2f}×")
    print(f"Mean RMS ratio (RMS_naive  / RMS_ml):        {cmp.rms_ratio.mean():.3f}×")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    w = cmp["window"].to_numpy()
    xp = np.arange(len(w))
    width = 0.38
    rms_naive = np.where(cmp.rms_naive > 0, cmp.rms_naive, np.nan)
    rms_ml    = np.where(cmp.rms_ml    > 0, cmp.rms_ml,    np.nan)
    ax.bar(xp - width/2, rms_naive, width, label="naive seed",  color="C7")
    ax.bar(xp + width/2, rms_ml,    width, label="ML (RF) seed", color="C1")
    ax.set_xticks(xp); ax.set_xticklabels(w)
    ax.set_xlabel("window")
    ax.set_ylabel("Holdover phase RMS (s)")
    ax.set_title("Predicted holdover RMS after ALS tuning — ML seed vs naive seed\n"
                 "(20% tail, ≈ 104 858 s horizon, log-scale)")
    ax.set_yscale("log")
    ax.legend()
    for i, row in cmp.iterrows():
        ymax = np.nanmax([row.rms_naive, row.rms_ml])
        if np.isfinite(ymax) and ymax > 0:
            ratio_str = (f"{row.rms_ratio:.1f}× better" if row.rms_ratio >= 1
                         else f"{1/row.rms_ratio:.1f}× worse")
            ax.text(i, ymax * 1.5, ratio_str, ha="center", fontsize=10)
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

    print("\n── Phase B: warm-start (iter + holdover RMS) comparison ──")
    phaseB_warmstart(ytr, pred_rf)


if __name__ == "__main__":
    main()
