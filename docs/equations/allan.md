# Allan Family Deviations

Phase data convention throughout: `x[i]` are phase samples in seconds; averaging
factor `m`; basic interval `τ₀`; averaging time `τ = m·τ₀`; `N` = number of phase
samples.

## ADEV — Allan deviation

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

## MDEV — Modified Allan deviation

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

## TDEV — Time deviation

**Formula** (SP1065 Eq. 17; MB23 §4.4.4):

```
TVAR(τ) = (τ² / 3) · MVAR(τ)
TDEV(τ) = √TVAR(τ)
```

**Implementation** (`julia/src/deviations.jl:tdev`, `matlab/+sigmatau/+dev/tdev.m`):
Calls `mdev`, then `dev = sqrt(tau^2 / 3) * mdev_val`.

**Status**: ✓ Verified. Scaling factor matches SP1065 §5.4.
