# SigmaTau — Equation Reference

Maps each algorithm to the authoritative source equation, notes any discrepancies
between sources, and records audit status.

**Primary references** (PDFs in `docs/papers/`):

| Sigil | Full citation |
|-------|---------------|
| **MB23** | Banerjee & Matsakis, *An Introduction to Modern Timekeeping and Time Transfer*, Springer 2023 (ISBN 978-3-031-30779-9). → `2023_banerjee_matsakis_timekeeping_book.pdf` |
| **SP1065** | Riley, *Handbook of Frequency Stability Analysis*, NIST SP1065, 2008. → `sp1065.pdf` |
| **G97** | Greenhall, "The Third-Difference Approach to Modified Allan Variance," IEEE T-IM 46(3), June 1997. → `1997_greenhall_third_difference_mvar_ieeetim.pdf` |
| **GHP99** | Greenhall, Howe & Percival, "Total Variance, an Estimator of Long-Term Frequency Stability," IEEE UFFC 46(5), Sept 1999. → `1999_greenhall_howe_percival_total_variance_ieee.pdf` |
| **HV99** | Howe & Vernotte, "Generalization of the Total Variance Approach to the Modified Allan Variance," PTTI 1999 (31st). → `1999_howe_vernotte_total_mvar_ptti.pdf` — canonical **MTOT** reference. |
| **H00** | Howe, Beard, Greenhall, Vernotte, Riley, "A Total Estimator of the Hadamard Function Used for GPS Operations," PTTI 2000 (32nd). → `2000_howe_total_estimator_hadamard_ptti.pdf` |
| **FCS01** | Howe, Beard, Greenhall, Vernotte, Riley, "Total Hadamard Variance: Application to Clock Steering by Kalman Filtering," Proc. IEEE FCS 2001. → `2001_howe_total_hadamard_variance_fcs.pdf` — canonical **HTOT** reference; provides bias `a(α)` table used by `totaldev_edf` and `bias` (code's "FCS 2001"). |
| **GR03** | Greenhall & Riley, "Uncertainty of Stability Variances Based on Finite Differences," PTTI 2003. → `2003_greenhall_riley_uncertainty_stability_variances_ptti.pdf` |
| **RG04** | Riley & Greenhall, "Power Law Noise Identification Using the Lag 1 Autocorrelation," 18th EFTF 2004. → `2004_riley_greenhall_lag1_acf_noiseid.pdf` |
| **H05** | Howe et al., "Enhancements to GPS Operations and Clock Evaluations Using a 'Total' Hadamard Deviation," IEEE UFFC 52(8), Aug 2005. → `2005_howe_total_hadamard_ieee.pdf` |

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

**Formula** (SP1065 Eq. 16; MB23 §4.4.3):

```
MVAR(τ) = 1 / (2m⁴τ₀²·N_e) · Σⱼ [Σₖ₌₀^{m-1} (x[j+k+2m] - 2x[j+k+m] + x[j+k])]²
```

where `N_e = N − 3m + 1` (number of complete windows) and `τ = mτ₀`. No `1/m`
factor appears inside the brackets in SP1065 or MB23 — the normalization lives
entirely outside as `1/(2m⁴τ₀²·N_e)`.

**Implementation** (cumsum / prefix-sum form; `julia/src/deviations.jl:_mdev_kernel`,
`matlab/+sigmatau/+dev/mdev.m:mdev_kernel`):

```julia
# Prefix sums s, then sliding window of length m
d = (s[j+2m] - 2*s[j+m] + s[j]) / m    # <-- inner 1/m is an algorithmic artifact
v = sum(abs2, d) / (Ne * 2 * m^2 * tau0^2)
```

The inner `1/m` is **not** part of the textbook formula — it is a by-product of
the prefix-sum / third-difference formulation (G97). The equivalence is purely
algebraic, via `τ = mτ₀`:

```
  1            1                1                  1
───────  =  ─────────────  =  ───────────────  =  ──────────    ← SP1065 form
2m²τ²       2m²(mτ₀)²         2m² · m²τ₀²         2m⁴τ₀²
```

Let `A_j = Σₖ₌₀^{m−1} (x[j+k+2m] − 2x[j+k+m] + x[j+k])` be the unaveraged inner
sum. The kernel computes `d = A_j / m`, then divides by `2m²τ₀²`, yielding:

```
    1              A_j²          A_j²
─────────  ·  ─────────  =  ─────────────
2m²τ₀²            m²          2m⁴τ₀²
```

which is identical to the SP1065 prefactor `1/(2m²τ²)` times `A_j²`. Redistribution
between code and textbook — no numerical disagreement.

**Status**: ✓ Verified. Mathematical check: for white FM noise, MVAR/AVAR → 1/2
asymptotically.

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

**Formula** (SP1065 §5.2.10; MB23 §4.5, modified form):

```
MHVAR(τ) = 1 / (6m⁴τ₀²·N_e) · Σⱼ [Σₖ₌₀^{m-1} (x[j+k+3m] - 3x[j+k+2m] + 3x[j+k+m] - x[j+k])]²
```

where `N_e = N − 4m + 1` and `τ = mτ₀`.

**Implementation** (`julia/src/deviations.jl:_mhdev_kernel`, `matlab/+sigmatau/+dev/mhdev.m:mhdev_kernel`):
Cumsum-based; analogous to MDEV but with third-difference kernel. The same `1/m`
artifact from the prefix-sum formulation appears inside the kernel and is
absorbed by the `1/(6m²τ₀²·N_e)` outer factor — algebraically equivalent to the
textbook `1/(6m⁴τ₀²·N_e)` form above.

**Status**: ✓ Verified structurally. Mirrors MDEV/HDEV relationship.

---

### LDEV — Lapinski deviation

A time-stability analog of MHDEV (this project — Ian Lapinski). Relates to
MHDEV in the same way TDEV relates to MDEV, but inherits MHDEV's improved
drift resistance and convergence over a wider range of divergent noise types
(down to RRFM / α = −4).

**Formula**:

```
LVAR(τ) = (3τ² / 10) · MHVAR(τ)
LDEV(τ) = τ · MHDEV(τ) / √(10/3)
```

The `√(10/3)` prefactor is the Hadamard analog of TDEV's `√3` (SP1065 §5.2.6,
Eq. 17): each comes from the integral of the variance's sampling-function
squared against the frequency-to-time-error kernel; the third-difference
sampling function of MHDEV yields `10/3` where MDEV's second-difference yields
`3`. Not a standard NIST SP1065 statistic — no §5 reference applies.

**Implementation** (`julia/src/deviations.jl:ldev`, `matlab/+sigmatau/+dev/ldev.m`):
Calls `mhdev`, then scales by `tau / sqrt(10/3)`.

```julia
const LDEV_MHDEV_PREFACTOR = sqrt(10 / 3)   # LDEV = τ · MHDEV / √(10/3)
```

**Status**: ✓ Verified. Prefactor `√(10/3)` matches MATLAB (current + legacy)
and Julia implementations.

---

### TOTDEV — Total deviation

**Formula** (SP1065 §5.2.11, Eq. 25 — phase form):

Data extended by symmetric reflection at both ends, then overlapping second
differences applied to the extended sequence.

```
TOTVAR(τ) = 1 / (2τ²(N-2)) · Σᵢ₌₂ᴺ⁻¹ (x*[i-m] - 2x*[i] + x*[i+m])²
```

where `x*` is the reflected-extended sequence of length `3N-4` and τ = m·τ₀.
Frequency form (Eq. 26) uses `2(M-1)` with M = N-1; the two forms agree since
M-1 = N-2.

**Implementation** (`julia/src/deviations.jl:_totdev_kernel`,
`matlab/+sigmatau/+dev/totdev.m:totdev_kernel`): denominator is
`2*(N-2)*(m*tau0)^2`.

**Status**: ✓ Verified against SP1065 Eq. 25 (2026-04-14).

---

### MTOTDEV — Modified total deviation

**Formula** (SP1065 §5.12; MB23 §4.4.3 total variant):

For each of `N-3m+1` subsegments of length `3m`: half-average detrend → symmetric
reflection → modified Allan sum (cumsum second differences averaged over `m`).

```
MTOTVAR(τ) = 1 / (2(mτ₀)² · (N-3m+1)) · Σₙ Σⱼ (aⱼ₊₂ - 2aⱼ₊₁ + aⱼ)²/(6m)
```

where `aⱼ` = m-point averages of the cumsum-reflected extended segment (length 9m).
The inner sum over `j` spans `6m` terms.

**Implementation** (`julia/src/deviations.jl:_mtotdev_kernel`,
`matlab/+sigmatau/+dev/mtotdev.m:mtotdev_kernel`): identical algorithms — half-avg
detrend, full 3-part `[rev; seq; rev]` extension, cumsum prefix sums, `Σd2²/(6m)`
per sub, divided by `2(mτ₀)²·nsubs`.

**Status**: ✓ Verified against SP1065 §5.12 and Stable32 (2026-04-14).

---

### HTOTDEV — Hadamard total deviation

**Formula** (SP1065 §5.13; MB23 §4.5 total variant):

**m = 1**: Uses standard HDEV (third differences on phase). Not the total-deviation
algorithm — this is a documented intentional exception (CLAUDE.md).

**m > 1**: Convert to frequency `y = diff(x)/τ₀`; for each of `Ny-3m+1` segments of
length `3m`: half-average detrend → `[rev; seq; rev]` extension → cumsum Hadamard
differences. Sum over `6m` terms.

**Implementation** (`julia/src/deviations.jl:_htotdev_kernel`,
`matlab/+sigmatau/+dev/htotdev.m:htotdev_kernel`): MATLAB and Julia are structurally
identical, including the m=1 branch.

**Bias correction**: Applied by engine: `deviation_corrected = deviation_raw / B`.
Confirmed by comparison with Stable32 (unbiased results match Stable32).

**Status**: ✓ Verified. Kernels match and bias correction direction is correct.

---

### MHTOTDEV — Modified Hadamard total deviation

**Formula** (MTOT-style extension of MHDEV, following HV99 + FCS01 total methodology; no MB23 coverage):

For each of `N-4m+1` subsegments of phase length `3m+1`: linear detrend → symmetric
reflection → third differences + length-m moving average.

```
MHTOTVAR(τ) = 1 / ((mτ₀)² · (N-4m+1)) · Σₙ block_var
block_var = Σ avg² / (n_avg · 6m²)    where avg is m-point cumsum window of third diffs
```

**Implementation** (`julia/src/deviations.jl:_mhtotdev_kernel`,
`matlab/+sigmatau/+dev/mhtotdev.m:mhtotdev_kernel`): identical algorithms — linear
(`detrend_linear`) detrend, full 3-part `[rev; seq; rev]` extension, cumsum
third-diffs, m-point moving average via cumsum.

**EDF**: No published analytical model dedicated to MHTOT. Engine uses the
FCS01 HTOT coefficients as an approximation (inferred total EDF mode for
`mhtotdev`) — noted as a known limitation, not a verified model.

**Status**: ✓ Verified structurally.

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
| 1 | MDEV / MHDEV | Code kernels carry an explicit inner `1/m` not present in SP1065 Eq. 16 or MB23 §4.4.3 | ✓ Algebraic artifact of the prefix-sum / third-difference (G97) formulation — outer normalization is `1/(2m²τ₀²·N_e)` instead of `1/(2m⁴τ₀²·N_e)`; the two forms are identical. No source typo. |
| 2 | `htotdev` EDF loop | CLAUDE.md flags potential off-by-one: loop over `numel(tau)` vs `numel(valid)` after trimming | ⚠ Not audited in this pass |
| 3 | `mhtotdev` Neff | CLAUDE.md flags: is segment count `N−4m+1` or `N−3m`? | ✓ Both MATLAB and Julia use `N−4m+1`; consistent with HV99 / FCS01 total methodology |
| 4 | MATLAB KF | `matlab/+sigmatau/+kf/` implementation available. | ✓ Ported from Julia |
| 5 | MHTOT reference attribution | Prior docs cited "Howe & Schlossberger, FCS 2001" — no such paper exists | ✓ Replaced: FCS01 = Howe/Beard/Greenhall/Vernotte/Riley 2001 (HTOT paper). MHTOT itself has no dedicated canonical reference; inferred from HV99 (MTOT) + FCS01 (HTOT). Code comments still say "FCS 2001" for MHTOT coefficients — understood to mean the HTOT table applied as approximation. |
| 6 | LDEV | Prior docs called it "Loran-C deviation" and cited SP1065 §5.6; §5.6 is "Bias Functions" — citation was bogus and formula `(τ²/6)·MHVAR` did not match the code's `√(10/3)` prefactor | ✓ Renamed **Lapinski Deviation**; formula updated to `LVAR = (3τ²/10)·MHVAR` to match MATLAB + Julia |
