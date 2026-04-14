#!/usr/bin/env python3
"""Plot the Kalman filter pipeline results written by kf_pipeline.jl.

Produces two figures:

  kf_optimization.png — 6-panel layout of the legacy MATLAB
    plot_optimization_results.m:
      (1,1) RMS prediction error vs horizon — initial vs optimized (log-log)
      (1,2) percent improvement vs horizon
      (1,3) Q-parameter bar chart (log y, initial vs optimized)
      (2,1) NLL surface contour over (q_wfm, q_rwfm) with markers
      (2,2) NLL slice along q_wfm at optimal q_rwfm
      (2,3) NLL slice along q_rwfm at optimal q_wfm

  kf_mhdev_comparison.png — three MHDEV curves on one log-log axis:
      (i)   data (mhdev_preview.csv)
      (ii)  theoretical from the fitted q (stage-3 output)
      (iii) theoretical from the NLL-optimal q (stage-5 output)
    Useful to see whether the NLL optimum still tracks the measured MHDEV or
    has moved to compensate for unmodelled structure.

Usage (from repo root):
  python3 scripts/python/plot_kf.py <dataset>

Reads:
  results/<dataset>/kf/kf_pipeline_summary.csv
  results/<dataset>/kf/kf_pipeline_prediction.csv
  results/<dataset>/kf/kf_nll_surface.csv
  results/<dataset>/kf/mhdev_preview.csv

Writes:
  results/<dataset>/kf/kf_optimization.png
  results/<dataset>/kf/kf_mhdev_comparison.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
if len(sys.argv) < 2:
    raise SystemExit("usage: plot_kf.py <dataset>")
DATASET    = sys.argv[1]
KF_DIR     = HERE / "results" / DATASET / "kf"
SUMMARY    = KF_DIR / "kf_pipeline_summary.csv"
PREDICTION = KF_DIR / "kf_pipeline_prediction.csv"
SURFACE    = KF_DIR / "kf_nll_surface.csv"
MHDEV_CSV  = KF_DIR / "mhdev_preview.csv"
OUT        = KF_DIR / "kf_optimization.png"
OUT_MHDEV  = KF_DIR / "kf_mhdev_comparison.png"

# Legacy MHDEV power-law coefficients (matlab/legacy/kflab/mhdev_fit.m):
#   σ²_MHDEV = (10/3)·q_wpm·τ^-3 + (7/16)·q_wfm·τ^-1 + (1/9)·q_rwfm·τ^+1
MHDEV_COMPONENTS = (
    ("wpm",  -3, 10 / 3, "tab:red"),
    ("wfm",  -1, 7 / 16, "tab:green"),
    ("rwfm", +1, 1 / 9,  "tab:purple"),
)

# Target horizons to mark on the RMS plot (seconds) — match
# HORIZONS_TO_REPORT in kf_pipeline.jl so the two stay in sync.
TARGET_HORIZONS_S = (1, 10, 60, 300, 3_600, 10_000)


def load_summary(path: Path) -> dict:
    """Return {'fitted': {...}, 'optimal': {...}} with q/innov/nll fields."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    out = {}
    for row in data:
        stage = row["stage"]
        out[stage] = {name: row[name] for name in data.dtype.names if name != "stage"}
    return out


def panel_rms(ax, prediction, target_horizons_s) -> None:
    h_s  = prediction["horizon_s"]
    rms0 = prediction["rms_fitted"]
    rms1 = prediction["rms_optimal"]

    ax.loglog(h_s, rms0, "-", color="tab:red",  lw=1.6, label="Fitted")
    ax.loglog(h_s, rms1, "-", color="tab:blue", lw=1.6, label="NLL-optimal")

    for h in target_horizons_s:
        idx = np.searchsorted(h_s, h)
        if idx < len(h_s) and h_s[idx] == h:
            ax.plot(h, rms1[idx], "o", mec="tab:blue", mfc="none",
                    ms=7, mew=1.3)
    ax.plot([], [], "o", mec="tab:blue", mfc="none", ms=7, mew=1.3,
            label="target horizons")

    ax.set_xlabel(r"prediction horizon $\tau$ [s]")
    ax.set_ylabel("RMS error")
    ax.set_title("Prediction performance")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)


def panel_improvement(ax, prediction) -> None:
    h_s  = prediction["horizon_s"]
    rms0 = prediction["rms_fitted"]
    rms1 = prediction["rms_optimal"]
    pct  = 100.0 * (rms0 - rms1) / rms0

    ax.semilogx(h_s, pct, "-", color="tab:green", lw=1.6)
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xlabel(r"prediction horizon $\tau$ [s]")
    ax.set_ylabel("improvement [%]")
    ax.set_title("Optimization gain")
    ax.grid(True, which="both", ls=":", alpha=0.5)


def panel_q_bars(ax, summary) -> None:
    names  = ("q_wpm", "q_wfm", "q_rwfm")
    fitted  = [summary["fitted"][n]  for n in names]
    optimal = [summary["optimal"][n] for n in names]
    idx    = np.arange(len(names))
    width  = 0.35

    ax.bar(idx - width/2, fitted,  width, color="tab:red",  label="Fitted")
    ax.bar(idx + width/2, optimal, width, color="tab:blue", label="Optimal")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([r"$q_\mathrm{wpm}$ (fixed)",
                        r"$q_\mathrm{wfm}$",
                        r"$q_\mathrm{rwfm}$"])
    ax.set_ylabel("value")
    ax.set_title("Q parameters")
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)


def panel_surface(ax, surface, summary) -> None:
    qw  = np.unique(surface["q_wfm"])
    qr  = np.unique(surface["q_rwfm"])
    # CSV is emitted with q_wfm outer, q_rwfm inner (see kf_pipeline.jl) so the
    # first index of the reshape is q_wfm and the second is q_rwfm.
    Z = surface["nll"].reshape(len(qw), len(qr))
    QW, QR = np.meshgrid(qw, qr, indexing="ij")

    cs = ax.contourf(QW, QR, Z, 20, cmap="turbo")
    plt.colorbar(cs, ax=ax, label="NLL")

    qw0, qr0 = summary["fitted"]["q_wfm"],  summary["fitted"]["q_rwfm"]
    qw1, qr1 = summary["optimal"]["q_wfm"], summary["optimal"]["q_rwfm"]
    ax.plot(qw0, qr0, marker="^", color="red",  ms=10, mec="white",
            mew=1.2, label="Fitted")
    ax.plot(qw1, qr1, marker="*", color="lime", ms=16, mec="black",
            mew=0.9, label="Optimal")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$q_\mathrm{wfm}$")
    ax.set_ylabel(r"$q_\mathrm{rwfm}$")
    ax.set_title(r"NLL surface  ($q_\mathrm{wpm}$ fixed)")
    ax.legend(loc="upper left", fontsize=9)


def panel_slice(ax, surface, summary, axis: str) -> None:
    """1D slice through the surface at the optimum along the other axis."""
    if axis == "wfm":
        vary, fixed = "q_wfm",  "q_rwfm"
        xlabel       = r"$q_\mathrm{wfm}$"
        initial      = summary["fitted"]["q_wfm"]
        optimal      = summary["optimal"]["q_wfm"]
        fixed_opt    = summary["optimal"]["q_rwfm"]
    else:
        vary, fixed = "q_rwfm", "q_wfm"
        xlabel       = r"$q_\mathrm{rwfm}$"
        initial      = summary["fitted"]["q_rwfm"]
        optimal      = summary["optimal"]["q_rwfm"]
        fixed_opt    = summary["optimal"]["q_wfm"]

    fixed_values = np.unique(surface[fixed])
    near         = fixed_values[np.argmin(np.abs(np.log10(fixed_values) -
                                                  np.log10(fixed_opt)))]
    mask         = np.isclose(surface[fixed], near)
    sel          = surface[mask]
    order        = np.argsort(sel[vary])

    ax.semilogx(sel[vary][order], sel["nll"][order],
                 "-", color="tab:blue", lw=1.5, marker=".")
    ax.axvline(initial, color="tab:red",  ls="--", lw=1.3, label="Fitted")
    ax.axvline(optimal, color="tab:green", ls="--", lw=1.3, label="Optimal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("NLL")
    ax.set_title(f"NLL slice along {vary}")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)


def theoretical_mhdev(tau, q_wpm, q_wfm, q_rwfm):
    """Return σ_MHDEV(τ) from the three-component power-law model."""
    var = np.zeros_like(tau, dtype=float)
    for name, slope, coeff, _ in MHDEV_COMPONENTS:
        q = {"wpm": q_wpm, "wfm": q_wfm, "rwfm": q_rwfm}[name]
        if q > 0:
            var = var + coeff * q * tau ** slope
    return np.sqrt(var)


def plot_mhdev_comparison(summary, mhdev_path: Path, out: Path) -> None:
    """Overlay data MHDEV with the fitted and NLL-optimal theoretical curves."""
    if not mhdev_path.exists():
        print(f"(skip {out.name}: {mhdev_path} missing)")
        return

    data = np.genfromtxt(mhdev_path, delimiter=",", names=True)
    keep = np.isfinite(data["sigma"]) & (data["sigma"] > 0)
    data = data[keep]
    tau  = data["tau"]

    q0 = summary["fitted"]
    q1 = summary["optimal"]
    tau_model = np.geomspace(tau.min(), tau.max(), 300)
    sig_fit   = theoretical_mhdev(tau_model, q0["q_wpm"], q0["q_wfm"], q0["q_rwfm"])
    sig_opt   = theoretical_mhdev(tau_model, q1["q_wpm"], q1["q_wfm"], q1["q_rwfm"])

    fig, (ax, ax_comp) = plt.subplots(1, 2, figsize=(14, 6))

    # Main comparison
    ax.loglog(tau, data["sigma"], "o-", color="tab:blue",   lw=1.4, ms=6,
              label="MHDEV (data)")
    ax.loglog(tau_model, sig_fit, "--", color="tab:red",   lw=1.6,
              label=(f"MHDEV (fitted theoretical)\n"
                     f"q_wpm={q0['q_wpm']:.2e} q_wfm={q0['q_wfm']:.2e} "
                     f"q_rwfm={q0['q_rwfm']:.2e}"))
    ax.loglog(tau_model, sig_opt, "-", color="tab:green", lw=1.8,
              label=(f"MHDEV (NLL-optimal theoretical)\n"
                     f"q_wpm={q1['q_wpm']:.2e} q_wfm={q1['q_wfm']:.2e} "
                     f"q_rwfm={q1['q_rwfm']:.2e}"))
    ax.set_xlabel(r"$\tau$ [s]")
    ax.set_ylabel(r"MHDEV $\sigma_{y,H}(\tau)$")
    ax.set_title("MHDEV: data vs fitted vs NLL-optimal")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)

    # Component breakdown of the optimal fit
    ax_comp.loglog(tau, data["sigma"], "o", color="tab:blue", ms=5,
                   label="data")
    total_var_opt = np.zeros_like(tau_model)
    for name, slope, coeff, color in MHDEV_COMPONENTS:
        q = {"wpm": q1["q_wpm"], "wfm": q1["q_wfm"], "rwfm": q1["q_rwfm"]}[name]
        if q <= 0:
            continue
        comp_var = coeff * q * tau_model ** slope
        total_var_opt += comp_var
        ax_comp.loglog(tau_model, np.sqrt(comp_var),
                       "--", color=color, lw=1.2,
                       label=f"{name.upper()} (q={q:.2e})")
    ax_comp.loglog(tau_model, np.sqrt(total_var_opt),
                   "-", color="black", lw=1.8, label="Total (optimal)")
    ax_comp.set_xlabel(r"$\tau$ [s]")
    ax_comp.set_ylabel(r"MHDEV $\sigma$")
    ax_comp.set_title("NLL-optimal components")
    ax_comp.grid(True, which="both", ls=":", alpha=0.5)
    ax_comp.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


MIN_NSAMPLES = 1000  # drop prediction rows averaged over fewer starting epochs
                     # — the tail is single-digit samples per horizon and the
                     # RMS estimate there is dominated by one trajectory, not
                     # a statistical mean.


def main() -> None:
    for p in (SUMMARY, PREDICTION, SURFACE):
        if not p.exists():
            raise SystemExit(f"missing {p}; run kf_pipeline.jl first")

    summary    = load_summary(SUMMARY)
    prediction = np.genfromtxt(PREDICTION, delimiter=",", names=True)
    surface    = np.genfromtxt(SURFACE,    delimiter=",", names=True)

    n_raw = len(prediction)
    prediction = prediction[prediction["n_samples"] >= MIN_NSAMPLES]
    n_keep = len(prediction)
    if n_keep < n_raw:
        h_cut = prediction["horizon_s"][-1]
        print(f"trimmed {n_raw - n_keep} tail rows (n_samples < {MIN_NSAMPLES}); "
              f"plotting horizons up to {h_cut:.3g}s")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panel_rms(axes[0, 0], prediction, TARGET_HORIZONS_S)
    panel_improvement(axes[0, 1], prediction)
    panel_q_bars(axes[0, 2], summary)
    panel_surface(axes[1, 0], surface, summary)
    panel_slice(axes[1, 1], surface, summary, "wfm")
    panel_slice(axes[1, 2], surface, summary, "rwfm")

    gain_rms = prediction["rms_optimal"].mean() / prediction["rms_fitted"].mean()
    fig.suptitle(
        f"KF Q-parameter optimization — {DATASET}   •   "
        f"mean-RMS ratio optimal/fitted = {gain_rms:.3f}   •   "
        f"NLL(opt) = {summary['optimal']['nll']:.3f}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT}")

    plot_mhdev_comparison(summary, MHDEV_CSV, OUT_MHDEV)


if __name__ == "__main__":
    main()
