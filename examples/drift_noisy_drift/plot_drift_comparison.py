#!/usr/bin/env python3
"""Create three comparison plots for drift/noisy-drift example."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path(__file__).resolve().parent / "out"

CASES = [
    ("baseline", "Baseline (WFM + RWFM)", "#1f77b4"),
    ("deterministic_drift", "Deterministic drift (quadratic phase)", "#ff7f0e"),
    ("noisy_drift", "Noisy drift (quadratic + RRFM-like)", "#2ca02c"),
]

COMPARISONS = [
    (("adev", "hdev"), "comparison_plot_1_adev_hdev.png", "Comparison 1: ADEV vs HDEV"),
    (("mdev", "mhdev"), "comparison_plot_2_mdev_mhdev.png", "Comparison 2: MDEV vs MHDEV"),
    (("tdev", "ldev"), "comparison_plot_3_tdev_ldev.png", "Comparison 3: TDEV vs LDEV"),
]


def load_sigma(case: str, dev: str):
    p = OUTDIR / f"{case}_{dev}.csv"
    data = np.genfromtxt(p, delimiter=",", names=True)
    tau = np.asarray(data["tau"], dtype=float)
    sig = np.asarray(data["sigma"], dtype=float)
    good = np.isfinite(tau) & np.isfinite(sig) & (tau > 0) & (sig > 0)
    return tau[good], sig[good]


for (dev_a, dev_b), outname, title in COMPARISONS:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)

    for ax, dev in zip(axes, (dev_a, dev_b)):
        for case, label, color in CASES:
            tau, sig = load_sigma(case, dev)
            ax.loglog(tau, sig, marker="o", markersize=2.8, linewidth=1.2, color=color, label=label)

        ax.set_xlabel("$\\tau$ (s)")
        ax.set_ylabel(r"$\\sigma(\\tau)$")
        ax.set_title(dev.upper())
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title)
    outpath = OUTDIR / outname
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    print(f"wrote {outpath}")
