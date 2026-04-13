# SigmaTau — Equation Reference

Maps each algorithm to the authoritative source equation, notes any discrepancies
between sources, and records audit status.

**Primary references**

| Sigil | Full citation |
|-------|---------------|
| **MB23** | Banerjee & Matsakis, *An Introduction to Modern Timekeeping and Time Transfer*, Springer 2023 (ISBN 978-3-031-30779-9). Local copy: `matsakis_banerjee.pdf`. |
| **SP1065** | Riley & Howe, *Handbook of Frequency Stability Analysis*, NIST SP1065, 2008. |
| **G03** | Greenhall & Riley, "Uncertainty of Stability Variances Based on Finite Samples," PTTI 2003. |
| **FCS01** | Howe & Schlossberger, "A new method of measuring… modified Hadamard total variance," FCS 2001. |

---

## Frequency Stability Deviations

Phase data convention throughout: `x[i]` are phase samples in seconds; averaging
factor `m`; basic interval `τ₀`; averaging time `τ = m·τ₀`; `N` = number of phase
samples.

### ADEV — Allan deviation

**Formula** (overlapping, SP1065 Eq. 14; MB23 §4.2 Eq. 4.2.3.1):

```
AVAR(τ) = 1 / (2(N-2m)(mτ₀)²) · Σᵢ (x[i+2m] - 2x[i+m] + x[i])²
```

**Implementation** (`julia/src/deviations.jl:_adev_kernel`, `matlab/+sigmatau/+dev/adev.m:adev_kernel`):

```julia
d2 = x[1+2m:end] - 2*x[1+m:end-m] + x[1:L]   # L = N-2m
v  = sum(abs2, d2) / (L * 2 * m^2 * tau0^2)
```

**Status**: ✓ Verified. Second-difference kernel and `2m²τ₀²` normalization match
SP1065 Eq. 14 and MB23 §4.2.

---

### MDEV — Modified Allan deviation

**Formula** (SP1065 Eq. 16; MB23 §4.4.3 Eq. 4.4.3.1):

```
MVAR(τ) = 1 / (2N_e(mτ₀)²) · Σⱼ [1/m · Σₖ₌₀^{m-1} (x[j+k+2m] - 2x[j+k+m] + x[j+k])]²
```

where `N_e = N - 3m + 1` (number of complete windows).

**Implementation** (cumsum / prefix-sum form; `julia/src/deviations.jl:_mdev_kernel`,
`matlab/+sigmatau/+dev/mdev.m:mdev_kernel`):

```julia
# Prefix sums s, then sliding window of length m
d = (s[j+2m] - 2*s[j+m] + s[j]) / m    # inner 1/m average
v = sum(abs2, d) / (Ne * 2 * m^2 * tau0^2)
```

**Status**: ✓ Verified. The `1/m` factor inside the brackets is correct; it is
**absent** from MB23 Eq. 4.4.3.2 — that appears to be a typo in the book. The code
matches SP1065 Eq. 16.  Mathematical check: for white FM noise,
MVAR/AVAR → 1/2 asymptotically (correct); omitting `1/m` gives ratio → ∞ (wrong).

---

### TDEV — Time deviation

**Formula** (SP1065 Eq. 17; MB23 §4.4.4):

```
TVAR(τ) = (τ² / 3) · MVAR(τ)
TDEV(τ) = √TVAR(τ)
```

**Implementation** (`julia/src/deviations.jl:tdev`, `matlab/+sigmatau/+dev/tdev.m`):
Calls `mdev`, then `dev = sqrt(tau^2 / 3) * mdev_val`.

**Status**: ✓ Verified. Scaling factor matches SP1065 §5.4.

---

### HDEV — Hadamard deviation

**Formula** (SP1065 §5.7, Eq. 18; MB23 §4.5):

```
HVAR(τ) = 1 / (6(N-3m)(mτ₀)²) · Σᵢ (x[i+3m] - 3x[i+2m] + 3x[i+m] - x[i])²
```

**Implementation** (`julia/src/deviations.jl:_hdev_kernel`, `matlab/+sigmatau/+dev/hdev.m:hdev_kernel`):

```julia
d3 = x[1+3m:end] - 3*x[1+2m:end-m] + 3*x[1+m:end-2m] - x[1:L]  # L = N-3m
v  = sum(abs2, d3) / (L * 6 * m^2 * tau0^2)
```

**Status**: ✓ Verified. Third-difference kernel and `6m²τ₀²` normalization match
SP1065 Eq. 18.

---

### MHDEV — Modified Hadamard deviation

**Formula** (SP1065 §5.8; MB23 §4.5, modified form):

```
MHVAR(τ) = 1 / (6N_e(mτ₀)²) · Σⱼ [1/m · Σₖ₌₀^{m-1} (x[j+k+3m] - 3x[j+k+2m] + 3x[j+k+m] - x[j+k])]²
```

where `N_e = N - 4m + 1`.

**Implementation** (`julia/src/deviations.jl:_mhdev_kernel`, `matlab/+sigmatau/+dev/mhdev.m:mhdev_kernel`):
Cumsum-based; analogous to MDEV but with third-difference kernel; denominator `6m²τ₀²`.

**Status**: ✓ Verified structurally. Mirrors MDEV/HDEV relationship.

---

### LDEV — Loran-C deviation

**Formula** (SP1065 §5.6):

```
LVAR(τ) = (τ² / 6) · MHVAR(τ)
LDEV(τ) = √LVAR(τ)
```

**Implementation** (`julia/src/deviations.jl:ldev`, `matlab/+sigmatau/+dev/ldev.m`):
Calls `mhdev`, then scales by `sqrt(tau^2 / 6)`.

**Status**: ✓ Verified. Scaling factor matches SP1065 §5.6.

---

### TOTDEV — Total deviation

**Formula** (SP1065 §5.11, Eq. 26):

Data extended by symmetric reflection at both ends, then overlapping AVAR applied
to extended sequence.

```
TOTVAR(τ) = 1 / (2(N-1)(mτ₀)²) · Σᵢ (x*[i+2m] - 2x*[i+m] + x*[i])²
```

where `x*` is the reflected-extended sequence and sum is over `i = 1 … N-2` (using
original-index positions in the extended array).

**Implementation** (`julia/src/deviations.jl:_totdev_kernel`,
`matlab/+sigmatau/+dev/totdev.m:totdev_kernel`): denominator is
`2*(N-2)*(m*tau0)^2`.

**Status**: ⚠ **Discrepancy flagged.** SP1065 §5.11 uses `2(N-1)(mτ₀)²`; both
MATLAB and Julia use `2(N-2)(mτ₀)²`. Source is the legacy codebase; needs
verification against SP1065 and MB23. Until resolved, cross-validate against a
known-good reference implementation (e.g., Stable32 or NIST TimeLab).

---

### MTOTDEV — Modified total deviation

**Formula** (SP1065 §5.12; MB23 §4.4.3 total variant):

For each of `N-3m+1` subsegments of length `3m`: half-average detrend → symmetric
reflection → modified Allan sum (cumsum second differences averaged over `m`).

```
MTOTVAR(τ) = 1 / (2(mτ₀)² · (N-3m+1)) · Σₙ Σⱼ (aⱼ₊₂ - 2aⱼ₊₁ + aⱼ)²/(6m)
```

where `aⱼ` = m-point averages of the cumsum-reflected extended segment.

**Implementation** (`julia/src/deviations.jl:_mtotdev_kernel`,
`matlab/+sigmatau/+dev/mtotdev.m:mtotdev_kernel`): identical algorithms — half-avg
detrend, `[rev; seg; rev]` extension, cumsum prefix sums, `Σd2²/(6m)` per sub,
divided by `2(mτ₀)²·nsubs`.

**Status**: ✓ Verified. MATLAB and Julia kernels are structurally identical.

---

### HTOTDEV — Hadamard total deviation

**Formula** (SP1065 §5.13; MB23 §4.5 total variant):

**m = 1**: Uses standard HDEV (third differences on phase). Not the total-deviation
algorithm — this is a documented intentional exception (CLAUDE.md).

**m > 1**: Convert to frequency `y = diff(x)/τ₀`; for each of `Ny-3m+1` segments of
length `3m`: half-average detrend → `[rev; seg; rev]` extension → cumsum Hadamard
differences.

**Implementation** (`julia/src/deviations.jl:_htotdev_kernel`,
`matlab/+sigmatau/+dev/htotdev.m:htotdev_kernel`): MATLAB and Julia are structurally
identical, including the m=1 branch.

**Bias correction**: Applied by engine (inferred from method name `htotdev`). Direction (multiply vs
divide) is flagged in CLAUDE.md as needing verification against SP1065 table and
Julia output.

**Status**: ✓ Kernels match between MATLAB and Julia. ⚠ Bias correction direction
requires cross-validation (see CLAUDE.md known bugs).

---

### MHTOTDEV — Modified Hadamard total deviation

**Formula** (FCS 2001, Howe & Schlossberger; no MB23 coverage):

For each of `N-4m+1` subsegments of phase length `3m+1`: linear detrend → symmetric
reflection → third differences + length-m moving average.

```
MHTOTVAR(τ) = 1 / ((mτ₀)² · (N-4m+1)) · Σₙ block_var
block_var = Σ avg² / (n_avg · 6m²)    where avg is m-point cumsum window of third diffs
```

**Implementation** (`julia/src/deviations.jl:_mhtotdev_kernel`,
`matlab/+sigmatau/+dev/mhtotdev.m:mhtotdev_kernel`): identical algorithms — linear
(`detrend_linear`) detrend, `[rev; seg; rev]` extension, cumsum third-diffs, m-point
moving average via cumsum.

**EDF**: No published analytical model. Engine uses approximate coefficients from
FCS 2001 (inferred total EDF mode for `mhtotdev`).

**Status**: ✓ MATLAB and Julia kernels are structurally identical. EDF is
approximate (FCS 2001 only).

---

## Kalman Filter

Reference: MB23 Chapter 13, §13.5–§13.6 (pp. 265–276).

### State vector

3-state model (nstates = 3) — default. States: phase offset (s), fractional
frequency offset (f), frequency drift (d).

```
x = [s, f, d]ᵀ
```

2-state (s, f) and 5-state (s, f, d, sin-diurnal, cos-diurnal) variants supported.

### State transition matrix Φ

(MB23 §13.5.6, kinematic clock model):

```
Φ = [1  τ  τ²/2]
    [0  1   τ  ]
    [0  0   1  ]
```

Encodes constant-velocity/constant-acceleration kinematics over interval τ.

**Implementation** (`julia/src/filter.jl:build_phi!`):

```julia
Φ[1,2] = τ;  Φ[1,3] = τ^2/2;  Φ[2,3] = τ
```

**Status**: ✓ Verified against MB23 §13.5.6.

---

### Process noise matrix Q

Continuous-time power-law noise model integrated over [0, τ] (Van Loan integration;
MB23 §13.5.4; SP1065 noise model):

| Element | Formula | Noise source |
|---------|---------|-------------|
| Q[1,1] | `q_wfm·τ + q_rwfm·τ³/3 + q_irwfm·τ⁵/20` | WFM + RWFM + IRWFM on phase |
| Q[1,2] = Q[2,1] | `q_rwfm·τ²/2 + q_irwfm·τ⁴/8` | RWFM + IRWFM cross |
| Q[2,2] | `q_rwfm·τ + q_irwfm·τ³/3` | RWFM + IRWFM on freq |
| Q[1,3] = Q[3,1] | `q_irwfm·τ³/6` | IRWFM cross (phase-drift) |
| Q[2,3] = Q[3,2] | `q_irwfm·τ²/2` | IRWFM cross (freq-drift) |
| Q[3,3] | `q_irwfm·τ` | IRWFM on drift |
| Q[4,4] = Q[5,5] | `q_diurnal` | Diurnal (nstates=5 only) |

White PM noise (`q_wpm`) does not enter Q — it is the measurement noise R.

**Implementation** (`julia/src/filter.jl:build_Q!`): exact formulas as above.

**Status**: ✓ Verified. τ-power coefficients (τ, τ³/3, τ⁵/20, τ²/2, τ⁴/8, τ³/6)
are exact results of continuous-time integration — do not approximate.

---

### Measurement model H

Phase-only measurement:

```
H = [1, 0, 0, ...]    (1 × nstates row vector)
```

For nstates = 5, H[4] = sin(2πk/T), H[5] = cos(2πk/T) at time step k.

**Status**: ✓ Verified.

---

### Kalman update equations

Standard linear Kalman filter (MB23 §13.5; no Joseph form — simplified form matches
legacy):

```
Innovation:   ν = z_k - H·x̂⁻
Gain:         K = P⁻·Hᵀ / (H·P⁻·Hᵀ + R)
State update: x̂ = x̂⁻ + K·ν
Cov update:   P = (I - K·H)·P⁻
```

**Status**: ✓ Verified. Simplified (non-Joseph) covariance update matches legacy.

---

### PID steering

(MB23 §13.6; Masterclock internal convention):

```
sumx  += x[1]                          # integrate phase error
steer  = -g_p·x[1] - g_i·sumx - g_d·x[2]
```

Phase error (not frequency) is integrated. This is the Masterclock convention and
must not be changed.

Steering fed back into prediction step:

```
x_pred[1] += steer · τ    # phase correction
x_pred[2] += steer        # frequency correction
```

**Status**: ✓ Verified against legacy `filter.jl` lines 138–146 and MB23 §13.6.

---

## Known Discrepancies and Open Issues

| # | Location | Issue | Status |
|---|----------|-------|--------|
| 1 | `totdev` kernel | Denominator `2(N-2)(mτ₀)²` — SP1065 §5.11 specifies `2(N-1)(mτ₀)²` | ⚠ Unresolved — needs verification against SP1065 and reference implementation |
| 2 | MDEV | MB23 Eq. 4.4.3.2 omits `1/m` normalization factor inside brackets | ✓ Code is correct (matches SP1065 Eq. 16); book has a typo |
| 3 | `htotdev` bias correction | Multiply vs. divide direction not confirmed | ⚠ Verify against SP1065 bias table and Julia output |
| 4 | `htotdev` EDF loop | CLAUDE.md flags potential off-by-one: loop over `numel(tau)` vs `numel(valid)` after trimming | ⚠ Not audited in this pass |
| 5 | `mhtotdev` Neff | CLAUDE.md flags: is segment count `N-4m+1` or `N-3m`? | ✓ Both MATLAB and Julia use `N-4m+1`; consistent with FCS 2001 |
| 6 | MATLAB KF | `matlab/+sigmatau/+kf/` is empty — no MATLAB Kalman filter implementation | ⚠ Not yet ported from Julia |
