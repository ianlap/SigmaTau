#!/usr/bin/env python3
"""Plot each SigmaTau deviation written by compute_devs.jl on its own figure.

Usage (from repo root):
  python3 examples/kf_pipeline/plot_devs.py <dataset>

Reads  results/<dataset>/devs/<dev>.csv and writes <dev>.png alongside.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
if len(sys.argv) < 2:
    raise SystemExit("usage: plot_devs.py <dataset>")
DATASET = sys.argv[1]
RES     = HERE / "results" / DATASET / "devs"
DEVS   = [
    "adev", "mdev", "hdev", "mhdev",
    "tdev", "ldev",
    "totdev", "mtotdev", "htotdev", "mhtotdev",
]
YLABEL = {
    "adev":     r"$\sigma_y(\tau)$  (overlapping Allan)",
    "mdev":     r"mod $\sigma_y(\tau)$",
    "hdev":     r"$\sigma_{y,H}(\tau)$  (overlapping Hadamard)",
    "mhdev":    r"mod $\sigma_{y,H}(\tau)$",
    "tdev":     r"$\sigma_x(\tau)$  (time deviation)",
    "ldev":     r"$\sigma_{x,H}(\tau)$  (Hadamard time dev)",
    "totdev":   r"$\sigma_{y,\mathrm{total}}(\tau)$",
    "mtotdev":  r"mod total $\sigma_y(\tau)$",
    "htotdev":  r"Hadamard total $\sigma_y(\tau)$",
    "mhtotdev": r"mod Hadamard total $\sigma_y(\tau)$",
}


def plot_one(name: str) -> Path | None:
    csv = RES / f"{name}.csv"
    if not csv.exists():
        print(f"skip {name}: {csv} not found")
        return None
    data = np.genfromtxt(csv, delimiter=",", names=True)
    tau  = data["tau"]
    dev  = data["deviation"]
    lo   = data["ci_lo"]
    hi   = data["ci_hi"]

    ok = np.isfinite(tau) & np.isfinite(dev) & (dev > 0)
    tau, dev, lo, hi = tau[ok], dev[ok], lo[ok], hi[ok]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ci_ok = np.isfinite(lo) & np.isfinite(hi) & (lo > 0)
    if ci_ok.any():
        ax.fill_between(tau[ci_ok], lo[ci_ok], hi[ci_ok],
                        color="C0", alpha=0.18, label="1σ CI")
    ax.loglog(tau, dev, "o-", color="C0", lw=1.6, ms=5, label=name)

    ax.set_xlabel(r"$\tau$ [s]")
    ax.set_ylabel(YLABEL.get(name, name))
    ax.set_title(f"{name}  —  6krb25apr  (N={len(tau)} τ-points)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    out = RES / f"{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    wrote = []
    for name in DEVS:
        p = plot_one(name)
        if p is not None:
            wrote.append(p)
    print(f"wrote {len(wrote)} plots to {RES}/")
    for p in wrote:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
