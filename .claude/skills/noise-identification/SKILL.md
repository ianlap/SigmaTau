---
name: noise-identification
description: >
  Use when building, modifying, or debugging noise identification — the
  SP1065 §5.6 dual-path estimator (lag-1 ACF + B1/R(n) fallback),
  NEFF_RELIABLE threshold, carry-forward rule, or the identify_b1rn
  inlined-kernel pattern. Trigger when working in julia/src/noise.jl or
  matlab/+sigmatau/+noise/noise_id.m.
---

# Noise Identification

SP1065 §5.6 dual-path dominant-power-law-noise estimator. Returns a vector of
α exponents (one per averaging factor) used to select the right EDF formula
and bias correction downstream.

## Dual-path dispatch

Per τ (averaging factor `m`), compute `N_eff = floor(N / m)`:

- `N_eff ≥ NEFF_RELIABLE` → **lag-1 ACF** (primary; SP1065 §5.6)
- `N_eff <  NEFF_RELIABLE` → **B1 ratio + R(n) fallback** (SP1065 §5.6)
- Both above return NaN → **carry forward** the most recent reliable α
  (Stable32 convention — "use the previous noise type estimate at the
  longest averaging time"). GEMINI.md §3.1.

`julia/src/noise.jl:34-40`, `matlab/+sigmatau/+noise/noise_id.m:38-40`.

## NEFF_RELIABLE = 50 (Goal G5, direction open)

```text
NEFF_RELIABLE = 50
```

`julia/src/noise.jl:24`, `matlab/+sigmatau/+noise/noise_id.m:28`.

SP1065 §5.6 cites a theoretical lag-1 ACF floor of **30**, but the code uses
**50** with an inline empirical rationale: the ACF estimator is still
high-variance just above 30, and 50 stabilises the long-τ tail. This is a
deliberate product choice, not unmigrated drift.

GEMINI.md §7 Goal G5 tracks resolution — three options named, direction
deliberately open. **Do not silently migrate the constant to 30 without
sign-off.** Resolution requires either a new ACF-variance measurement over
the 30-50 range or an explicit product decision. Deferred until post-PH-551.

## Recursion trap — identify_b1rn MUST use inlined kernels (hard rule)

The B1/R(n) path needs ADEV at m and MDEV at m. Calling `sigmatau.dev.adev`
or `sigmatau.dev.mdev` from inside `identify_b1rn` would re-enter
`sigmatau.dev.engine` → `noise_id` → `identify_b1rn` → **infinite recursion**
whenever the argument data is short enough to fall into the B1/R(n) path
itself.

The fix is to **inline the kernels**:

- **Julia** (`noise.jl:237-260`): `_simple_avar(x, m)` and
  `_simple_mdev(x, m, tau0)` — dedicated minimal kernels that don't go back
  through the engine. Commit `7231782` established this rule.
- **MATLAB** (`noise_id.m:127-133` and `191-202`): inlined ADEV (second
  differences on decimated phase) and inlined MDEV (prefix-sum form). The
  warning comments at lines 122-127 and 187-189 are load-bearing — they
  explain why `sigmatau.dev.adev`/`mdev` must not be called here.

**If you change the B1/R(n) path, do not call any function in `sigmatau.dev.*`
or `noise_id.*` from inside `identify_b1rn`. Use the local `_simple_*`
kernels or inline the computation.** This is the "infinite recursion just
bit us" rule made durable.

## Preprocessing (both paths)

```text
x_mean = mean(x); x_std = std(x)
mask = |(x - x_mean) / x_std| < 5           # z-score outlier removal
x = detrend_linear(x[mask])                 # linear detrend after
```

Guard: if `x_std < eps`, skip outlier removal and go straight to detrend.
`julia/src/noise.jl:64-73`.

## Lag-1 ACF method

```text
detrend_quadratic(x)            # phase: quadratic detrend after decimation at m
# -or-
mean-in-block + detrend_linear  # frequency: m-block average + linear detrend

loop:
    r1 = lag1_acf(x)
    rho = r1 / (1 + r1)
    if d >= dmin && (rho < 0.25 || d >= dmax):
        p = -2 * (rho + d)
        alpha = p + 2 * (data_type == "phase" ? 1 : 0)
        return alpha
    x = diff(x); d += 1; require length(x) >= 5
```

`julia/src/noise.jl:83-112`, `matlab/+sigmatau/+noise/noise_id.m:68-99`.

- `rho < 0.25` is the convergence criterion (flat enough → p uniquely determined)
- `dmax` caps the differencing depth (default 2; engine passes 0..2)
- The `+2·(phase)` term shifts α from frequency-noise convention (μ) to
  SigmaTau's signed α (frequency: α = p; phase: α = p + 2)

Guard `lag1_acf`: when `ssx < eps*N` (constant input after detrend), return NaN.

## B1 ratio / R(n) fallback

```text
B1_obs = var_classical / var_Allan

mu_list    = [+1, 0, -1, -2]            # μ: AVAR slope exponent
alpha_list = [-2, -1, 0,  2]            # α: SigmaTau's signed label
# geometric-mean boundaries between adjacent B1 theory values
for i in 1..3:
    if B1_obs > sqrt(B1_theory(N, mu_i) · B1_theory(N, mu_{i+1})):
        pick (mu_i, alpha_i)
        break
```

Closed forms for B1_theory (`noise.jl:204-212`):

| μ  | Noise | B1_theory(N) |
|----|-------|--------------|
| +2 | FW FM (α=-3) | `N(N+1)/6` |
| +1 | RWFM  (α=-2) | `N/2` |
|  0 | FLFM  (α=-1) | `N·log(N) / (2(N-1)·log 2)` |
| -1 | WHFM  (α= 0) | `1.0` (reference) |
| -2 | WHPM/FLPM (α=2 or 1) | `(N² - 1) / (1.5·N·(N-1))` |

`_b1_theory` uses SP1065 Eq 73 / Howe-Beard 1998 for other μ values.

### R(n) refinement (phase data only, initial α=2)

When the B1 step initially picks `mu_best == -2` and `data_type == "phase"`,
compute `Rn_obs = (MDEV/ADEV)²` on the full data and refine:

- `R_hi = rn_theory(m, 0)` — α=2 (White PM)
- `R_lo = rn_theory(m, -1)` — α=1 (Flicker PM)
- `alpha = (Rn_obs > sqrt(R_hi · R_lo)) ? 1 : 2`

This is the only place inlined MDEV is used. `rn_theory` (`noise.jl:216-227`)
closed forms:

- `b =  0`: `R → 1/m` (WHPM asymptotic)
- `b = -1`: closed form via leading-order AVAR and MVAR expansions

## Carry-forward rule (last resort)

When both lag-1 ACF and B1/R(n) return NaN for the current `m`, propagate
the most recent reliable α forward:

```text
if !isnan(alpha):
    alpha_list[k] = alpha
    last_reliable = alpha
elif !isnan(last_reliable):
    alpha_list[k] = last_reliable
```

`noise.jl:51-56`, `noise_id.m:57-63`.

Matches Stable32's behaviour; prevents a single τ's estimator failure from
cascading to NaN EDFs across the tail. The engine rounds NaN α to 0 for EDF
purposes so the final result is still numerically valid.
