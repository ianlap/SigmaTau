# API Consistency Report — Julia vs MATLAB

**Generated**: 2026-04-11  
**Julia source**: `julia/src/deviations.jl` + `julia/src/engine.jl` + `julia/src/types.jl`  
**MATLAB source**: `matlab/legacy/stablab/` (pre-refactor reference)

---

## Summary

| Category | Status |
|---|---|
| Function set (10 functions) | ✅ Match |
| Parameter names (`x`, `tau0`, `m_list`) | ✅ Match |
| `m_list` default generation logic | ✅ Match |
| `data_type` keyword | ⚠️ Julia-only — MATLAB has no equivalent |
| Return type / structure | ⚠️ Intentional redesign — see §3 |
| Deviation result field name | ❌ Mismatch — MATLAB per-function name vs Julia `.deviation` |
| `ci` computed eagerly vs lazily | ❌ Mismatch — MATLAB eager, Julia lazy |
| `neff` / `Neff` exposure | ⚠️ Partial — only `totdev` exposes it in MATLAB |
| Confidence level default | ✅ Both `p = 0.683` |
| `alpha` field type | ⚠️ MATLAB `float`, Julia `Int` (rounded) |
| `ldev` description | ⚠️ Name mismatch in docstrings |
| Namespace | ⚠️ MATLAB `allanlab.*` (legacy) → target `sigmatau.dev.*` |

---

## 1. Function Set

Both APIs implement all 10 SP1065 deviation functions:

| Function | Julia | MATLAB |
|---|---|---|
| `adev` | ✅ | ✅ |
| `mdev` | ✅ | ✅ |
| `tdev` | ✅ | ✅ |
| `hdev` | ✅ | ✅ |
| `mhdev` | ✅ | ✅ |
| `ldev` | ✅ | ✅ |
| `totdev` | ✅ | ✅ |
| `mtotdev` | ✅ | ✅ |
| `htotdev` | ✅ | ✅ |
| `mhtotdev` | ✅ | ✅ |

**No missing functions on either side.**

---

## 2. Input Parameters

### 2a. Positional parameters — ✅ Match

All 10 functions in both languages share the same first two positional parameters:

| Parameter | MATLAB | Julia | Notes |
|---|---|---|---|
| `x` | positional 1 | positional 1 | Phase data vector |
| `tau0` | positional 2 | positional 2 | Sampling interval |

### 2b. `m_list` — ✅ Match (name), ⚠️ Style difference

| | MATLAB | Julia |
|---|---|---|
| Name | `m_list` | `m_list` |
| Passing style | 3rd positional arg (optional) | keyword arg (`m_list=nothing`) |
| Default | `2.^(0:floor(log2(N/min_factor)))` | `[2^k for k in 0:floor(Int, log2(N/min_factor))]` |

The default generation logic is numerically identical. The `min_factor` used per function:

| Function | MATLAB `min_factor` | Julia `min_factor` |
|---|---|---|
| `adev` | 2 (`N/2`) | 2 |
| `mdev` | 3 (`N/3`) | 3 |
| `tdev` | delegates to `mdev` | delegates to `mdev` |
| `hdev` | 4 (`N/4`) | 4 |
| `mhdev` | 4 (`N/4`) | 4 |
| `ldev` | delegates to `mhdev` | delegates to `mhdev` |
| `totdev` | 2 (`N/2`) | 2 |
| `mtotdev` | 3 (`N/3`) | 3 |
| `htotdev` | 3 (`N/3`) | 3 |
| `mhtotdev` | 4 (`N/4`) | 4 |

### 2c. `data_type` — ❌ Julia-only parameter

**Julia** exposes `data_type :: Symbol = :phase` on all 10 functions. Passing `:freq` triggers a frequency-to-phase conversion (`cumsum(y) * tau0`) inside the engine before any computation.

**MATLAB** has no `data_type` parameter. All legacy functions accept only phase data (`x`). Frequency-to-phase conversion must be done by the caller beforehand.

**Action required for refactored MATLAB (`+sigmatau/+dev/`)**: add a `data_type` name-value argument defaulting to `'phase'` to every function, delegating conversion to `engine.m`. This is the single largest input-API gap.

---

## 3. Return Type / Structure

### 3a. MATLAB — multiple positional outputs

Every MATLAB function returns a fixed-position tuple:

```matlab
[tau, <devname>, edf, ci, alpha] = adev(x, tau0)
```

Where `<devname>` is the function name (`adev`, `mdev`, …). `totdev` uniquely returns a 6th output:

```matlab
[tau, totdev, edf, ci, alpha, Neff] = totdev(x, tau0)
```

### 3b. Julia — single `DeviationResult` struct

```julia
r = adev(x, tau0)
# r.tau, r.deviation, r.edf, r.ci, r.alpha, r.neff, r.tau0, r.N, r.method, r.confidence
```

The Julia struct carries additional metadata (`tau0`, `N`, `method`, `confidence`) not available in the MATLAB tuple.

**This is an intentional redesign, not a bug.** `unpack_result(r, Val(5))` provides MATLAB-style tuple destructuring.

### 3c. Result field name for deviation value — ❌ Mismatch

| | MATLAB | Julia |
|---|---|---|
| Deviation value field | per-function name (e.g. `adev`, `mdev`) | always `.deviation` |

Julia deliberately uses a uniform field name `.deviation` for all functions, improving composability. The MATLAB target API (`+sigmatau/+dev/`) should follow the same convention (returning a struct), or document this difference explicitly.

---

## 4. Confidence Interval Computation

### ❌ Timing mismatch

| | MATLAB | Julia |
|---|---|---|
| CI computed | Immediately, inside each function | Not computed — `ci` is `NaN` until `compute_ci(result)` is called |
| Confidence level default | `p = 0.683` (all functions) | `confidence = 0.683` stored in struct |

**Impact**: Code that does `ci = adev(x, tau0); ci(1,:)` works in MATLAB but in Julia requires an extra `compute_ci` call first.

**Action required for refactored MATLAB**: If returning a struct (recommended), CIs could be computed eagerly to maintain backward compatibility; or adopt the Julia lazy pattern and document it.

---

## 5. `neff` / `Neff` Exposure — ⚠️ Partial match

| | MATLAB | Julia |
|---|---|---|
| `adev` | Internal `Neff` (not returned) | `r.neff` |
| `mdev` | Internal `N_eff` (not returned) | `r.neff` |
| `totdev` | Returned as 6th output `Neff` | `r.neff` |
| All others | Internal, not returned | `r.neff` |

**Action required for refactored MATLAB**: expose `neff` consistently across all functions (as a struct field or additional output).

---

## 6. `alpha` Field Type — ⚠️ Type difference

| | MATLAB | Julia |
|---|---|---|
| Type | `double` (float) | `Vector{Int}` (rounded integer) |
| Values | e.g. `2.0`, `-2.0` | e.g. `2`, `-2` |

Both represent the same SP1065 power-law noise exponent α. Julia rounds to integer for cleaner indexing into EDF lookup tables. This is a minor implementation detail — no numerical difference in practice.

---

## 7. `ldev` Description — ⚠️ Docstring mismatch

| | Description |
|---|---|
| MATLAB `ldev.m` | "Lapinski deviation" (`σ_L`) |
| Julia `ldev` docstring | "Hadamard time deviation (LDEV)" |

The formula (`τ · MHDEV(τ) / √(10/3)`) is the same in both. The name "Lapinski" appears only in the legacy MATLAB docstring; SP1065 does not use this name. The refactored MATLAB should align with Julia and use "Hadamard time deviation (LDEV)".

---

## 8. Namespace

| | Current | Target (CLAUDE.md) |
|---|---|---|
| MATLAB | `allanlab.adev(x, tau0)` (legacy) | `sigmatau.dev.adev(x, tau0)` |
| Julia | `SigmaTau.adev(x, tau0)` | same |

The legacy MATLAB namespace is `allanlab`; the refactored target per CLAUDE.md is `sigmatau.dev`. This is a pending refactor, not a bug.

---

## 9. `mhtotdev` EDF

Both implementations agree: no closed-form published EDF model exists for MHTOTDEV. Both attempt `totaldev_edf("mhtot", …)` with fallback to `NaN`.

| | MATLAB | Julia |
|---|---|---|
| EDF model | Falls back to `NaN` per τ if formula unavailable | Uses `totaldev_edf("mhtot", …)` via engine |

✅ Behaviour is consistent.

---

## 10. `htotdev` m=1 Special Case

Both implementations agree: when `m == 1`, `htotdev` uses the overlapping HDEV (third differences on phase) rather than the total deviation algorithm.

✅ Behaviour is consistent (matches CLAUDE.md critical rule).

---

## Action Items for Refactored MATLAB (`+sigmatau/+dev/`)

1. **Add `data_type` name-value argument** (default `'phase'`) to all 10 functions.  
   Place frequency-to-phase conversion (`cumsum(y)*tau0`) in `engine.m`.

2. **Return a struct** (e.g. `result`) rather than positional outputs, with a uniform `.deviation` field (instead of per-function names like `.adev`, `.mdev`).

3. **Expose `neff`** consistently in the struct for all 10 functions, not just `totdev`.

4. **Align `ldev` docstring** to "Hadamard time deviation (LDEV)" — drop "Lapinski deviation".

5. **Update namespace** from `allanlab.*` to `sigmatau.dev.*` per CLAUDE.md.

6. **Decide CI timing**: document whether CI is computed eagerly (as in legacy MATLAB) or lazily (as in Julia). Recommend adopting the Julia lazy pattern for consistency.
