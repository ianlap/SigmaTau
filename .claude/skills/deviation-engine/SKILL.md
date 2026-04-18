---
name: deviation-engine
description: >
  Use when building, modifying, or debugging any deviation function (adev, mdev,
  hdev, mhdev, tdev, ldev, totdev, mtotdev, htotdev, mhtotdev) or the shared
  engine that powers them. Trigger when working in matlab/+sigmatau/+dev/ or
  julia/src/engine.jl or julia/src/deviations/{allan,hadamard,total}.jl.
---

# Deviation Engine

All 10 deviation functions dispatch through a shared engine. Each wrapper is thin
— a kernel plus a `DevParams`/params struct — and all boilerplate (validation,
noise-id, EDF, bias, CI) lives in the engine.

## Kernel contract (4-arg, both languages)

```text
kernel(x, m, tau0, x_cs) → (variance, neff)
```

- `x` — full phase vector
- `m` — averaging factor
- `tau0` — sampling interval
- `x_cs` — prefix sum `cumsum([0; x])`, length `N+1`, precomputed once by the engine and shared across all kernel calls in the m-loop

Kernels return **variance** (σ²). The engine takes `sqrt(max(var_val, 0))` and
clamps small negative variance (`< -eps*N`) to NaN to guard against catastrophic
cancellation in the second/third-difference sums.

MATLAB migrated to the 4-arg contract in Apr 2026 (commits `69210f3`, `2d87b83`,
`bb699cd` — engine plus all 10 kernels). GEMINI.md §2.3 and Goal G2 are stale on
this point (logged in FIX_PARKING_LOT.md); ignore them, use current code as
source of truth.

Kernels that don't use `x_cs` still take it — use `~` (MATLAB) or `_x_cs`
(Julia) for the unused arg so every kernel presents the same signature.

## Engine responsibilities

See `julia/src/engine.jl:30` and `matlab/+sigmatau/+dev/engine.m:1`. In order:

1. Frequency → phase via `cumsum(y)*tau0` when `data_type` is `:freq`/`"freq"`
2. `validate_phase_data`, `validate_tau0`
3. Default `m_list` from `params.min_factor` (octave-spaced); filter `m >= 1`
4. `noise_id(x, m_list, ...)` → per-m α exponents
5. Precompute `x_cs` once
6. Kernel dispatch in the m-loop (engine takes the sqrt, not the kernel)
7. EDF: `totaldev_edf(params.total_type, ...)` when `total_type` is set, else
   `calculate_edf(alpha, params.d, m, params.F_fn(m), 1, N)`
8. Bias correction when `params.bias_type` is set (total/mtot/htot only)
9. `compute_ci` — chi-squared with Gaussian fallback
10. Return a `DeviationResult` / struct with `(tau, dev, edf, ci, alpha, neff, tau0, N, method, confidence)`

Wrappers supply **only** the kernel + params. Everything else is the engine's
job. Do not duplicate engine boilerplate in a wrapper.

## Wrapper example (Julia)

```julia
function adev(x, tau0; m_list=nothing, data_type=:phase)
    params = DevParams(name="adev", min_factor=2, d=2,
                       F_fn = m -> m, dmin=0, dmax=1)
    return engine(x, tau0, m_list, _adev_kernel, params; data_type)
end

function _adev_kernel(x, m, tau0, _x_cs)
    N = length(x); L = N - 2m
    L <= 0 && return (NaN, 0)
    d2 = @view(x[1+2m:N]) .- 2 .* @view(x[1+m:N-m]) .+ @view(x[1:L])
    return (sum(abs2, d2) / (2.0 * Float64(m)^2 * tau0^2 * L), L)
end
```

## Deviation parameters quick reference

| Dev      | d | min_factor | F_fn   | total_type | bias_type |
|----------|---|------------|--------|------------|-----------|
| adev     | 2 | 2          | m      | —          | —         |
| mdev     | 2 | 3          | 1      | —          | —         |
| hdev     | 3 | 4          | m      | —          | —         |
| mhdev    | 3 | 4          | 1      | —          | —         |
| totdev   | 2 | 2          | m      | totvar     | totvar    |
| mtotdev  | 2 | 3          | 1      | mtot       | mtot      |
| htotdev  | 3 | 3          | m      | htot       | htot      |
| mhtotdev | 3 | 4          | 1      | mhtot      | —         |
| tdev     | — | —          | —      | wraps mdev  | wraps mdev  |
| ldev     | — | —          | —      | wraps mhdev | wraps mhdev |

## Gotchas

- **Totdev denominator.** Phase form uses `2(N−2)(mτ₀)²` per SP1065 Eq 25
  (equivalently `2(M−1)` in frequency form, `M = N−1`). Institutional memory;
  do not change. GEMINI.md §2.9.

- **htotdev at m=1.** Uses the overlapping HDEV formula, not the total-deviation
  reflection algorithm (reflection is ill-defined at m=1). `julia/src/deviations/total.jl:184,217`,
  `matlab/+sigmatau/+dev/htotdev.m:12`. GEMINI.md §2.7.

- **Int64 overflow in MDEV/MHDEV denominators** (fixed — don't regress). The
  denominators `N_e·2·m⁴` and `N_e·6·m⁴` silently wrap Int64 at `m ≳ 10⁴`. At
  `m=11461, N=131072`, the true value is `3.35×10²¹` but wraps to `~10¹⁸`,
  making MDEV ~200× too large and MHDEV go slightly negative (engine clamped to
  0, MHDEV fit then reported `q_rwfm = 3.5×10⁻²⁴` on a record with true value
  `~10⁻³⁰`). Fix: promote `2 → 2.0` and `m → Float64(m)` in the denominators.
  Commit `ead11ea` across `allan.jl`, `hadamard.jl`, `noise.jl`, `total.jl`.
  ml/STATE.md:170-181 has the narrative.

- **tdev and ldev don't use the engine directly.** They call `mdev`/`mhdev`,
  rescale the result, and package it. No kernel of their own.

- **Bias-correction tables must match SP1065 exactly** (Riley & Howe §5.2).
  Deviations from tabulated coefficients without a new citation are regressions.
  GEMINI.md §2.8.

- **Variance, not deviation.** If a kernel returns `sqrt(variance)`, the result
  is off by a square root across the whole τ-grid and every downstream caller
  sees wrong numbers. The engine does the sqrt.

- **`data_type` keyword, not positional.** Both engines accept `data_type` as a
  keyword/name-value. Passing `"freq"` as a positional third argument will be
  mistaken for `m_list`.
