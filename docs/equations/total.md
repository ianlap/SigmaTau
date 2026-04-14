# Total Family Deviations

Phase data convention throughout: `x[i]` are phase samples in seconds; averaging
factor `m`; basic interval `τ₀`; averaging time `τ = m·τ₀`; `N` = number of phase
samples.

## TOTDEV — Total deviation

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

## MTOTDEV — Modified total deviation

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

## HTOTDEV — Hadamard total deviation

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

## MHTOTDEV — Modified Hadamard total deviation

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
