#!/usr/bin/env python3
"""Overlay sigmatau ADEV vs allantools oadev on the same white-FM phase series."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import allantools

HERE = Path(__file__).parent

x   = np.loadtxt(HERE / "wfm_phase.csv")
sig = np.loadtxt(HERE / "sigmatau_adev.csv", delimiter=",")
tau_s, dev_s, ci_lo, ci_hi = sig.T

tau0 = 1.0
m    = tau_s.astype(int)
taus_at, ad_at, ade_at, ns_at = allantools.oadev(
    x, rate=1/tau0, data_type="phase", taus=tau_s
)

fit_mask = tau_s >= 4
slope_s  = np.polyfit(np.log10(tau_s[fit_mask]), np.log10(dev_s[fit_mask]),   1)[0]
slope_at = np.polyfit(np.log10(taus_at[fit_mask]), np.log10(ad_at[fit_mask]), 1)[0]

theo = dev_s[0] * (tau_s / tau_s[0]) ** (-0.5)

fig, ax = plt.subplots(figsize=(7.5, 5.5))

ax.fill_between(tau_s, ci_lo, ci_hi, color="C0", alpha=0.15,
                label="sigmatau 1σ CI")

ax.loglog(tau_s,   dev_s,  "o-",  color="C0", lw=1.5, ms=6,
          label=f"sigmatau adev (slope={slope_s:+.3f})")
ax.loglog(taus_at, ad_at,  "s--", color="C3", lw=1.2, ms=5, mfc="none",
          label=f"allantools oadev (slope={slope_at:+.3f})")
ax.loglog(tau_s,   theo,   ":",   color="gray", lw=1.2,
          label="theoretical τ$^{-1/2}$ (white FM)")

ax.set_xlabel(r"$\tau$ [s]")
ax.set_ylabel(r"$\sigma_y(\tau)$ (ADEV)")
ax.set_title(f"ADEV on white FM phase (N={len(x)}, seed=12345)")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(loc="lower left", framealpha=0.9)

rel_err = np.abs(dev_s - ad_at) / ad_at
ax.text(0.98, 0.97,
        f"max |sigmatau−allantools|/allantools = {rel_err.max():.2e}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))

fig.tight_layout()
out = HERE / "adev_wfm.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
print(f"sigmatau slope   (m>=4): {slope_s:+.4f}")
print(f"allantools slope (m>=4): {slope_at:+.4f}")
print(f"max relative diff:       {rel_err.max():.3e}")
