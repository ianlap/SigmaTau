# Hadamard Family Deviations

Phase data convention throughout: `x[i]` are phase samples in seconds; averaging
factor `m`; basic interval `τ₀`; averaging time `τ = m·τ₀`; `N` = number of phase
samples.

## HDEV — Hadamard deviation

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

## MHDEV — Modified Hadamard deviation

**Formula** (SP1065 §5.2.10; MB23 §4.5, modified form):

```
MHVAR(τ) = 1 / (6m⁴τ₀²·N_e) · Σⱼ [Σₖ₌₀^{m-1} (x[j+k+3m] - 3x[j+k+2m] + 3x[j+k+m] - x[j+k])]²
```

where `N_e = N − 4m + 1` and `τ = mτ₀`.

**Implementation** (`julia/src/deviations.jl:_mhdev_kernel`,
`matlab/+sigmatau/+dev/mhdev.m:mhdev_kernel`): Cumsum-based; analogous to MDEV
but with third-difference kernel:

```julia
# Prefix sums s of third differences, then sliding window of length m
d = (s[j+3m] - 3*s[j+2m] - 3*s[j+m] - s[j]) / m   # <-- inner 1/m artifact
v = sum(abs2, d) / (Ne * 6 * m^2 * tau0^2)
```

Same `τ = mτ₀` redistribution as MDEV:

```
  1            1                1                  1
───────  =  ─────────────  =  ───────────────  =  ──────────    ← SP1065 form
6m²τ²       6m²(mτ₀)²         6m² · m²τ₀²         6m⁴τ₀²
```

Let `B_j = Σₖ₌₀^{m−1} (x[j+k+3m] − 3x[j+k+2m] + 3x[j+k+m] − x[j+k])` be the
unaveraged inner third-difference sum. The kernel computes `d = B_j / m`, then
divides by `6m²τ₀²`, yielding:

```
    1              B_j²          B_j²
─────────  ·  ─────────  =  ─────────────
6m²τ₀²            m²          6m⁴τ₀²
```

— identical to the textbook `1/(6m²τ²) · B_j²` form. No numerical disagreement.

**Status**: ✓ Verified structurally. Mirrors MDEV/HDEV relationship.

---

## LDEV — Lapinski deviation

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
