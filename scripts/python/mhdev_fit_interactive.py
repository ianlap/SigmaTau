#!/usr/bin/env python3
"""Interactive MHDEV noise-component fit — port of mhdev_fit.m.

Reads MHDEV from `results/<dataset>/kf/mhdev_preview.csv` (produced by
mhdev_preview.jl), shows a live-updating log-log plot, and lets the user
pick noise type + index range repeatedly. Each fit is subtracted from the
residual (variance for power-law types, σ for flicker types) so later
fits operate on what's left, matching the legacy MATLAB flow.

When the user exits with 'done', the total coefficients are written to
`results/<dataset>/kf/mhdev_fit.csv` so kf_pipeline.jl can pick them up as
its initial Q values.

Noise-type menu:
    1 WPM  (slope σ² = -3, coeff 10/3)
    2 WFM  (slope σ² = -1, coeff  7/16)
    3 RWFM (slope σ² = +1, coeff  1/9)
    4 RRFM (slope σ² = +3, coeff 11/120)
    5 FFM  (slope σ  =  0)      [flicker, fits sig0 in σ-space]
    6 FPM  (slope σ  = -2)      [flicker]
    7 undo
    0 done

Run from repo root:
    python3 scripts/python/mhdev_fit_interactive.py <dataset>
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent

if len(sys.argv) < 2:
    raise SystemExit("usage: mhdev_fit_interactive.py <dataset>")
DATASET = sys.argv[1]
KF_DIR  = HERE / "results" / DATASET / "kf"
CSV_IN  = KF_DIR / "mhdev_preview.csv"
CSV_OUT = KF_DIR / "mhdev_fit.csv"
PNG_OUT = KF_DIR / "mhdev_fit.png"

# Legacy noise-type table (matlab/legacy/kflab/mhdev_fit.m)
POWERLAW = {
    "wpm":  {"slope": -3, "coeff": 10 / 3,    "color": "tab:red"},
    "wfm":  {"slope": -1, "coeff": 7 / 16,    "color": "tab:green"},
    "rwfm": {"slope": +1, "coeff": 1 / 9,     "color": "tab:purple"},
    "rrfm": {"slope": +3, "coeff": 11 / 120,  "color": "tab:brown"},
}
FLICKER = {
    "ffm": {"slope":  0, "color": "tab:orange"},
    "fpm": {"slope": -2, "color": "tab:pink"},
}
MENU = [
    (1, "wpm"),  (2, "wfm"),  (3, "rwfm"), (4, "rrfm"),
    (5, "ffm"),  (6, "fpm"),
]


def load_preview(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    keep = np.isfinite(data["sigma"]) & (data["sigma"] > 0)
    data = data[keep]
    return (data["idx"].astype(int), data["tau"], data["sigma"])


def print_table(idx, tau, sig, var_resid) -> None:
    print("\n  idx    tau[s]       sigma        sigma^2     resid_var      slope(σ²)")
    for k in range(len(idx)):
        if k + 1 < len(idx):
            slope = np.log(var_resid[k + 1] / var_resid[k]) / np.log(tau[k + 1] / tau[k]) \
                if var_resid[k] > 0 and var_resid[k + 1] > 0 else np.nan
            slope_s = f"{slope:+6.2f}" if np.isfinite(slope) else "  —   "
        else:
            slope_s = "  —   "
        print(f"  {idx[k]:3d}  {tau[k]:10.4g}  {sig[k]:10.4g}  {sig[k]**2:11.4g}  "
              f"{var_resid[k]:11.4g}     {slope_s}")


def weighted_mean(y, w):
    sw = w.sum()
    if sw <= 0:
        raise ValueError("all weights are zero")
    mu = (w * y).sum() / sw
    n_eff = sw**2 / (w**2).sum()
    data_var = (w * (y - mu) ** 2).sum() / sw
    se = np.sqrt(data_var / max(n_eff, 1.0))
    return mu, se


def fit_powerlaw(noise_type, tau_sub, var_sub, w_sub):
    """Fit σ² = coeff·q·τ^slope via log-space weighted mean. Returns (q, q_std)."""
    m   = POWERLAW[noise_type]
    keep = var_sub > 0
    if not keep.any():
        return 0.0, np.nan
    y = np.log(var_sub[keep]) - m["slope"] * np.log(tau_sub[keep])
    mu, se = weighted_mean(y, w_sub[keep])
    q     = np.exp(mu) / m["coeff"]
    q_std = q * se
    return q, q_std


def fit_flicker(noise_type, tau_sub, sig_sub, w_sub):
    """Fit σ = sig0·τ^slope. Returns (sig0, sig0_std)."""
    m   = FLICKER[noise_type]
    keep = sig_sub > 0
    if not keep.any():
        return 0.0, np.nan
    y = np.log(sig_sub[keep]) - m["slope"] * np.log(tau_sub[keep])
    mu, se = weighted_mean(y, w_sub[keep])
    sig0     = np.exp(mu)
    sig0_std = sig0 * se
    return sig0, sig0_std


def subtract_powerlaw(var_resid, tau, q, noise_type):
    m = POWERLAW[noise_type]
    var_new = var_resid - m["coeff"] * q * tau ** m["slope"]
    return np.maximum(var_new, 0.0)


def subtract_flicker(sig_resid, tau, sig0, noise_type):
    m = FLICKER[noise_type]
    comp = sig0 * tau ** m["slope"]
    return np.sqrt(np.maximum(sig_resid**2 - comp**2, 0.0))


def render(ax_main, ax_resid, ax_text,
           idx, tau, sig_orig, var_orig, sig_resid, var_resid,
           q, sig0, history_msg):
    """Redraw the three panels. Called on every iteration."""
    ax_main.clear()
    ax_resid.clear()
    ax_text.clear()
    ax_text.axis("off")

    # Main: MHDEV data + fitted components + total model
    ax_main.loglog(tau, sig_orig, "o-", color="tab:blue", lw=1.4, ms=6,
                   label="Data")
    for i, t, s in zip(idx, tau, sig_orig):
        ax_main.annotate(f"{i}", xy=(t, s),
                         xytext=(5, 5), textcoords="offset points",
                         fontsize=8, color="black")

    if any(q.values()) or any(sig0.values()):
        tau_model = np.geomspace(tau.min(), tau.max(), 200)
        total_var = np.zeros_like(tau_model)
        for name, val in q.items():
            if val <= 0:
                continue
            comp_var = POWERLAW[name]["coeff"] * val * tau_model**POWERLAW[name]["slope"]
            total_var += comp_var
            ax_main.loglog(tau_model, np.sqrt(comp_var),
                           ls="--", color=POWERLAW[name]["color"], lw=1.2,
                           label=f"{name.upper()} (q={val:.2e})")
        for name, val in sig0.items():
            if val <= 0:
                continue
            comp_sig = val * tau_model**FLICKER[name]["slope"]
            total_var += comp_sig**2
            ax_main.loglog(tau_model, comp_sig,
                           ls="--", color=FLICKER[name]["color"], lw=1.2,
                           label=f"{name.upper()} (σ₀={val:.2e})")
        ax_main.loglog(tau_model, np.sqrt(total_var),
                       "-", color="black", lw=2, label="Total model")

    ax_main.set_xlabel(r"$\tau$ [s]")
    ax_main.set_ylabel(r"MHDEV $\sigma$")
    ax_main.set_title(f"MHDEV fit — live — {DATASET}")
    ax_main.grid(True, which="both", ls=":", alpha=0.5)
    ax_main.legend(loc="lower left", fontsize=8)

    # Residual variance
    ax_resid.loglog(tau, var_orig,  "o-", color="tab:blue", lw=1.3, label="Original σ²")
    pos = var_resid > 0
    ax_resid.loglog(tau[pos], var_resid[pos], "o-", color="tab:red", lw=1.3,
                    label="Residual σ²")
    ax_resid.set_xlabel(r"$\tau$ [s]")
    ax_resid.set_ylabel(r"$\sigma^2$")
    ax_resid.set_title("Residual after fits")
    ax_resid.grid(True, which="both", ls=":", alpha=0.5)
    ax_resid.legend(loc="best", fontsize=8)

    # Summary
    ax_text.text(0.02, 0.95, "Fitted coefficients", fontsize=13, fontweight="bold",
                 transform=ax_text.transAxes, va="top")
    lines = [
        f"q_wpm   = {q['wpm']:.3e}",
        f"q_wfm   = {q['wfm']:.3e}",
        f"q_rwfm  = {q['rwfm']:.3e}",
        f"q_rrfm  = {q['rrfm']:.3e}",
        f"σ₀_ffm  = {sig0['ffm']:.3e}",
        f"σ₀_fpm  = {sig0['fpm']:.3e}",
    ]
    for k, line in enumerate(lines):
        ax_text.text(0.02, 0.83 - k * 0.09, line, fontsize=11, family="monospace",
                     transform=ax_text.transAxes, va="top")
    if history_msg:
        ax_text.text(0.02, 0.14, history_msg, fontsize=10, color="tab:blue",
                     transform=ax_text.transAxes, va="top", wrap=True)

    plt.pause(0.05)


def prompt_choice() -> int:
    print("\n  1 WPM   2 WFM   3 RWFM   4 RRFM   5 FFM   6 FPM   7 undo   0 done")
    while True:
        raw = input("Choice: ").strip()
        if raw == "":
            continue
        try:
            return int(raw)
        except ValueError:
            print("  not a number — try again")


def prompt_range(n: int) -> tuple[int, int] | None:
    while True:
        raw = input("Index range [start end]: ").strip()
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("  enter two numbers, e.g. '1 4'")
            continue
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            print("  not numbers — try again")
            continue
        if not (1 <= a < b <= n):
            print(f"  range must satisfy 1 <= a < b <= {n}")
            continue
        return a, b


def main() -> None:
    if not CSV_IN.exists():
        raise SystemExit(f"missing {CSV_IN}; run mhdev_preview.jl first")

    idx, tau, sig = load_preview(CSV_IN)
    print(f"loaded {len(idx)} usable MHDEV points  "
          f"tau {tau[0]:.3g} … {tau[-1]:.3g}")

    # Flat weights (uniform); mhdev_preview.csv does not carry CI. If we want
    # legacy-grade weighting later, plumb CI through the preview CSV and swap
    # this for the ci2weights 'conservative' rule.
    w = np.ones_like(sig)

    sig_orig = sig.copy()
    var_orig = sig ** 2
    var_resid = var_orig.copy()
    sig_resid = sig.copy()
    q    = {"wpm": 0.0, "wfm": 0.0, "rwfm": 0.0, "rrfm": 0.0}
    sig0 = {"ffm": 0.0, "fpm": 0.0}
    history = []   # stack of prior states for undo

    plt.ion()
    fig = plt.figure(figsize=(15, 8))
    gs  = fig.add_gridspec(2, 3)
    ax_main  = fig.add_subplot(gs[:, 0:2])
    ax_resid = fig.add_subplot(gs[0, 2])
    ax_text  = fig.add_subplot(gs[1, 2])

    print_table(idx, tau, sig_orig, var_resid)
    render(ax_main, ax_resid, ax_text,
           idx, tau, sig_orig, var_orig, sig_resid, var_resid,
           q, sig0, history_msg="")

    while True:
        choice = prompt_choice()
        if choice == 0:
            break
        if choice == 7:
            if not history:
                print("  nothing to undo")
                continue
            state = history.pop()
            var_resid = state["var_resid"]
            sig_resid = state["sig_resid"]
            q         = state["q"]
            sig0      = state["sig0"]
            msg = "undone"
        elif 1 <= choice <= 6:
            noise_type = dict(MENU)[choice]
            rng = prompt_range(len(idx))
            if rng is None:
                continue
            a, b = rng
            mask = (idx >= a) & (idx <= b)
            tau_sub = tau[mask]
            w_sub   = w[mask]

            history.append({
                "var_resid": var_resid.copy(),
                "sig_resid": sig_resid.copy(),
                "q":    copy.deepcopy(q),
                "sig0": copy.deepcopy(sig0),
            })

            if noise_type in POWERLAW:
                q_est, q_std = fit_powerlaw(noise_type, tau_sub,
                                             var_resid[mask], w_sub)
                q[noise_type] += q_est
                var_resid = subtract_powerlaw(var_resid, tau, q_est, noise_type)
                sig_resid = np.sqrt(var_resid)
                msg = (f"fit {noise_type.upper()}  idx [{a},{b}]  "
                       f"q = {q_est:.3e} ± {q_std:.2e}")
            else:  # flicker
                sig0_est, sig0_std = fit_flicker(noise_type, tau_sub,
                                                  sig_resid[mask], w_sub)
                sig0[noise_type] += sig0_est
                sig_resid = subtract_flicker(sig_resid, tau, sig0_est, noise_type)
                var_resid = sig_resid ** 2
                msg = (f"fit {noise_type.upper()}  idx [{a},{b}]  "
                       f"σ₀ = {sig0_est:.3e} ± {sig0_std:.2e}")
            print("  " + msg)
        else:
            print("  unknown choice")
            continue

        print_table(idx, tau, sig_orig, var_resid)
        render(ax_main, ax_resid, ax_text,
               idx, tau, sig_orig, var_orig, sig_resid, var_resid,
               q, sig0, history_msg=msg)

    # Persist final fit for kf_pipeline.jl — flat key,value CSV so Julia can
    # read it with a two-line readdlm + Dict comprehension (no new deps).
    out = {
        "q_wpm":    q["wpm"],
        "q_wfm":    q["wfm"],
        "q_rwfm":   q["rwfm"],
        "q_rrfm":   q["rrfm"],
        "sig0_ffm": sig0["ffm"],
        "sig0_fpm": sig0["fpm"],
    }
    with CSV_OUT.open("w") as f:
        f.write("param,value\n")
        for k, v in out.items():
            f.write(f"{k},{v:.17g}\n")
    fig.savefig(PNG_OUT, dpi=130)
    plt.ioff()
    print(f"\nFinal coefficients:")
    for k, v in out.items():
        print(f"  {k:8s} = {v:.4e}")
    print(f"wrote {CSV_OUT}")
    print(f"wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
