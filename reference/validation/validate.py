"""validate.py — SigmaTau vs Stable32 vs allantools comparison & plots.

Loads:
  - sigmatau_results.csv           (from compute_sigmatau_deviations.jl)
  - stable32out/stable32_data_full.csv   (pre-parsed Stable32 output)
  - stable32gen.DAT                (raw phase data, for allantools)

Computes allantools equivalents where available, then produces one log-log
comparison PNG per deviation in ./plots/ with SigmaTau + Stable32 + allantools
overlaid. Also prints a relative-error summary table.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import allantools

HERE = Path(__file__).parent
DAT = HERE / "stable32gen.DAT"
SIGMATAU_CSV = HERE / "sigmatau_results.csv"
STABLE32_CSV = HERE / "stable32out" / "stable32_data_full.csv"
PLOT_DIR = HERE / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Phase data load ──────────────────────────────────────────────────────────

def load_phase(path: Path) -> np.ndarray:
    x = []
    in_data = False
    with path.open() as f:
        for line in f:
            if line.strip().startswith("# Header End"):
                in_data = True
                continue
            if in_data:
                s = line.strip()
                if s:
                    x.append(float(s))
    return np.asarray(x, dtype=float)


# ── Deviation dispatch table ─────────────────────────────────────────────────
# Each entry: sigmatau dev name → {stable32 label, allantools fn or None, display}

DEV_MAP = {
    "adev": {
        "display": "ADEV (overlapping)",
        "stable32_label": "Overlapping Allan",
        "allantools": lambda x, r, taus: allantools.oadev(x, rate=r, data_type="phase", taus=taus),
    },
    "mdev": {
        "display": "MDEV",
        "stable32_label": "Modified Allan",
        "allantools": lambda x, r, taus: allantools.mdev(x, rate=r, data_type="phase", taus=taus),
    },
    "tdev": {
        "display": "TDEV",
        "stable32_label": "Time",
        "allantools": lambda x, r, taus: allantools.tdev(x, rate=r, data_type="phase", taus=taus),
    },
    "hdev": {
        "display": "HDEV (overlapping)",
        "stable32_label": "Overlapping Hadamard",
        "allantools": lambda x, r, taus: allantools.ohdev(x, rate=r, data_type="phase", taus=taus),
    },
    "mhdev": {
        "display": "MHDEV (modified Hadamard)",
        "stable32_label": None,  # Stable32 has no plain MHDEV
        "allantools": None,      # allantools has no MHDEV either
    },
    "ldev": {
        "display": "LDEV",
        "stable32_label": None,
        "allantools": None,
    },
    "totdev": {
        "display": "TOTDEV",
        "stable32_label": "Total",
        "allantools": lambda x, r, taus: allantools.totdev(x, rate=r, data_type="phase", taus=taus),
    },
    "mtotdev": {
        "display": "MTOTDEV",
        "stable32_label": "Modified Total",
        "allantools": lambda x, r, taus: allantools.mtotdev(x, rate=r, data_type="phase", taus=taus),
    },
    "htotdev": {
        "display": "HTOTDEV",
        "stable32_label": "Hadamard Total",
        "allantools": lambda x, r, taus: allantools.htotdev(x, rate=r, data_type="phase", taus=taus),
    },
    "mhtotdev": {
        "display": "MHTOTDEV",
        "stable32_label": None,
        "allantools": None,
    },
}


def _rel_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| / |b|, with zeros in b returning nan."""
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[mask] = np.abs(a[mask] - b[mask]) / np.abs(b[mask])
    return out


def main() -> int:
    x = load_phase(DAT)
    rate = 1.0  # τ₀ = 1 s
    print(f"Loaded {len(x)} phase samples from {DAT.name}")

    sig = pd.read_csv(SIGMATAU_CSV)
    s32_all = pd.read_csv(STABLE32_CSV)

    summary_rows = []

    for dev_name, info in DEV_MAP.items():
        print(f"\n── {dev_name} ({info['display']}) ──")
        st = sig[sig["dev"] == dev_name].sort_values("tau").reset_index(drop=True)
        st = st[np.isfinite(st["sigma"]) & (st["neff"] >= 3)].reset_index(drop=True)
        if len(st) == 0:
            print("  no SigmaTau rows — skipping")
            continue

        # Stable32
        s32 = None
        if info["stable32_label"] is not None:
            s32 = s32_all[s32_all["Type"] == info["stable32_label"]].sort_values("Tau").reset_index(drop=True)
            if len(s32) == 0:
                s32 = None

        # allantools
        at_taus, at_sigma = None, None
        if info["allantools"] is not None:
            try:
                taus_req = st["tau"].to_numpy()
                at_tau_out, at_dev, _, _ = info["allantools"](x, rate, taus_req)
                at_taus = at_tau_out
                at_sigma = at_dev
            except Exception as e:
                print(f"  allantools failed: {e}")

        # ── plot ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        # SigmaTau with CI error bars
        sig_err_lo = st["sigma"].to_numpy() - st["ci_lo"].to_numpy()
        sig_err_hi = st["ci_hi"].to_numpy() - st["sigma"].to_numpy()
        ax.errorbar(
            st["tau"], st["sigma"],
            yerr=[sig_err_lo, sig_err_hi],
            fmt="o-", color="C0", label="SigmaTau (Julia)",
            capsize=3, markersize=6, linewidth=1.5,
        )

        if s32 is not None:
            s32_err_lo = s32["Sigma"].to_numpy() - s32["MinSigma"].to_numpy()
            s32_err_hi = s32["MaxSigma"].to_numpy() - s32["Sigma"].to_numpy()
            ax.errorbar(
                s32["Tau"], s32["Sigma"],
                yerr=[s32_err_lo, s32_err_hi],
                fmt="s--", color="C1", label="Stable32",
                capsize=3, markersize=5, linewidth=1.2, alpha=0.85,
            )

        if at_sigma is not None:
            ax.plot(at_taus, at_sigma,
                    "^:", color="C2", label="allantools",
                    markersize=6, linewidth=1.2, alpha=0.85)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\tau$ (s)")
        ax.set_ylabel(rf"$\sigma_{{{dev_name}}}(\tau)$")
        ax.set_title(f"{info['display']}  —  stable32gen.DAT ({len(x)} phase points)")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        out = PLOT_DIR / f"{dev_name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  plot → {out.relative_to(HERE)}")

        # ── relative-error summary ──────────────────────────────────────────
        if s32 is not None:
            # match on tau by inner join
            merged = pd.merge(
                st[["tau", "sigma"]].rename(columns={"sigma": "sig_st"}),
                s32[["Tau", "Sigma"]].rename(columns={"Tau": "tau", "Sigma": "sig_s32"}),
                on="tau", how="inner",
            )
            if len(merged):
                rel = _rel_err(merged["sig_st"].to_numpy(), merged["sig_s32"].to_numpy())
                summary_rows.append({
                    "dev": dev_name, "ref": "Stable32",
                    "n": len(merged),
                    "max_relerr": np.nanmax(rel),
                    "median_relerr": np.nanmedian(rel),
                })

        if at_sigma is not None:
            # allantools returns its own τ grid; interpolate-free match by tau
            df_at = pd.DataFrame({"tau": at_taus, "sig_at": at_sigma})
            merged = pd.merge(
                st[["tau", "sigma"]].rename(columns={"sigma": "sig_st"}),
                df_at, on="tau", how="inner",
            )
            if len(merged):
                rel = _rel_err(merged["sig_st"].to_numpy(), merged["sig_at"].to_numpy())
                summary_rows.append({
                    "dev": dev_name, "ref": "allantools",
                    "n": len(merged),
                    "max_relerr": np.nanmax(rel),
                    "median_relerr": np.nanmedian(rel),
                })

    # Print summary
    summary = pd.DataFrame(summary_rows)
    if len(summary):
        summary["max_relerr"] = summary["max_relerr"].map(lambda v: f"{v:.2e}")
        summary["median_relerr"] = summary["median_relerr"].map(lambda v: f"{v:.2e}")
        print("\n" + "=" * 64)
        print("RELATIVE ERROR SUMMARY (SigmaTau vs reference)")
        print("=" * 64)
        print(summary.to_string(index=False))
        (HERE / "validation_summary.csv").write_text(summary.to_csv(index=False))
        print(f"\nSummary → validation_summary.csv")
    print(f"\nPlots written to {PLOT_DIR.relative_to(HERE)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
