#!/usr/bin/env python3
"""Plot MHDEV preview curve (from mhdev_preview.jl) with point indices labeled.

Labels match the `idx` column that mhdev_fit_interactive.py (and the inline
fallback in kf_pipeline.jl) uses to select fit regions. Also draws reference
slope lines for WPM (σ slope −3/2), WFM (−1/2), RWFM (+1/2).

Usage (from repo root):
    python3 scripts/python/plot_mhdev_preview.py <dataset>

Reads:  results/<dataset>/kf/mhdev_preview.csv
Writes: results/<dataset>/kf/mhdev_preview.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent

if len(sys.argv) < 2:
    raise SystemExit("usage: plot_mhdev_preview.py <dataset>")
DATASET = sys.argv[1]
CSV = HERE / "results" / DATASET / "kf" / "mhdev_preview.csv"
OUT = HERE / "results" / DATASET / "kf" / "mhdev_preview.png"


def slope_guide(ax, tau_ref, sigma_ref, slope, label, color, xmax):
    """Draw a guide line sigma ∝ τ^slope anchored at (tau_ref, sigma_ref)."""
    tau = np.geomspace(tau_ref, xmax, 25)
    sig = sigma_ref * (tau / tau_ref) ** slope
    ax.loglog(tau, sig, ls=":", color=color, lw=1.2, alpha=0.7, label=label)


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}; run mhdev_preview.jl first")

    data = np.genfromtxt(CSV, delimiter=",", names=True)
    keep = np.isfinite(data["sigma"]) & (data["sigma"] > 0)
    data = data[keep]
    idx  = data["idx"].astype(int)
    tau  = data["tau"]
    sig  = data["sigma"]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.loglog(tau, sig, "o-", color="tab:blue", lw=1.4, ms=6,
              label=f"MHDEV ({DATASET})")

    # Index annotations — offset to the upper-right of each marker.
    for i, t, s in zip(idx, tau, sig):
        if s <= 0 or not np.isfinite(s):
            continue
        ax.annotate(f"{i}", xy=(t, s),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9, color="black")

    # Slope guides for WPM (σ ∝ τ^-3/2 in MHDEV), WFM (τ^-1/2), RWFM (τ^+1/2).
    # Anchor them at the first valid point so the user can mentally shift.
    valid = np.isfinite(sig) & (sig > 0)
    if valid.any():
        i0 = np.argmax(valid)
        t0, s0 = tau[i0], sig[i0]
        xmax = tau[valid].max()
        slope_guide(ax, t0, s0,          -1.5, "WPM  (slope σ = -3/2)", "tab:red",    xmax)
        slope_guide(ax, t0, s0 / 5,      -0.5, "WFM  (slope σ = -1/2)", "tab:green",  xmax)
        slope_guide(ax, t0, s0 / 50_000, +0.5, "RWFM (slope σ = +1/2)", "tab:purple", xmax)

    ax.set_xlabel(r"$\tau$ [s]")
    ax.set_ylabel(r"MHDEV $\sigma_{y,H}(\tau)$")
    ax.set_title(f"MHDEV preview — {DATASET} — indices label each point for fit regions")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
