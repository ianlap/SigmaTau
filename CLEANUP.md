# CLEANUP — Dead Code & Redundancy Audit

Scope: `julia/src/` (the `matlab/+sigmatau/` refactor has not been written yet; only
`matlab/legacy/` exists, so no MATLAB cleanup is applicable here).

---

## 1. Dead code: `_make_result` in `julia/src/types.jl`

**Before** (types.jl lines 86–106):
```julia
"""
    _make_result(tau, dev, alpha_float, neff, tau0, N, method, confidence)

Construct a `DeviationResult` with NaN EDF and CI placeholders.
EDF and CI are filled by `compute_ci` in stats.jl.
"""
function _make_result(tau, dev, alpha_float, neff, tau0, N::Int,
                      method::String, confidence)
    L = length(dev)
    alpha_int = Vector{Int}(undef, L)
    for i in 1:L
        alpha_int[i] = isnan(alpha_float[i]) ? 0 : round(Int, alpha_float[i])
    end
    DeviationResult(
        Float64.(tau), Float64.(dev),
        fill(NaN, L), fill(NaN, L, 2),
        alpha_int, Vector{Int}(neff),
        Float64(tau0), N, method, Float64(confidence)
    )
end
```

**After**: function removed entirely.

**Why**: `_make_result` was the result-construction helper in the old StabLab
deviation wrappers (each wrapper called it directly). After the refactor, the
shared `engine` builds `DeviationResult` itself; no call to `_make_result`
exists anywhere in `julia/src/`. It survived the refactor as an orphan.

---

## 2. Bug: operator-precedence guard in `julia/src/noise.jl`

**Before** (noise.jl line 132):
```julia
isnan(avar_val) || avar_val <= 0 && return (0, -2, NaN)
```

**After**:
```julia
(isnan(avar_val) || avar_val <= 0) && return (0, -2, NaN)
```

**Why**: In Julia `&&` binds tighter than `||`, so the original line parsed as:
```julia
isnan(avar_val) || (avar_val <= 0 && return (0, -2, NaN))
```
When `avar_val` is `NaN`, `||` short-circuits and returns `true` *without*
triggering the `return`. Execution continued into `B1_obs = var_class / avar_val`
(which yields `NaN`) and into the theory table, producing silently wrong noise
classifications instead of a clean early exit. The fix adds explicit parentheses
so both conditions are covered by the `&&` early-return guard.

---

## 3. Redundant import: `using Statistics` in `julia/src/noise.jl`

**Before** (noise.jl line 5):
```julia
using Statistics
```

**After**: line removed.

**Why**: `noise.jl` is `include`d into the `SigmaTau` module. The module-level
`using Statistics` in `SigmaTau.jl` already brings `Statistics` into scope for
every included file. The duplicate `using` inside `noise.jl` is a legacy
copy-paste from when it was a standalone file; it is harmless but misleading.

---

## 4. Structural note: EDF recomputation in `compute_ci`

`compute_ci` in `stats.jl` calls `edf_for_result`, which re-dispatches on
`result.method` (a string) and reruns `calculate_edf` / `totaldev_edf` — the
same formulas the engine already ran and stored in `result.edf`.

**No change was made here.** A test in `runtests.jl` constructs a
`DeviationResult` with `edf = fill(NaN, L)` and calls `compute_ci` expecting
EDF to be populated. This is a documented public-API contract: `compute_ci`
is the authoritative way to obtain EDF for any `DeviationResult`, including
hand-constructed ones. Changing `compute_ci` to bypass `edf_for_result` would
break that contract.

The redundancy for engine-produced results is minor; the recomputation is O(L)
arithmetic and not performance-sensitive.

---

## 5. Stale comment in `julia/src/SigmaTau.jl`

**Before**:
```julia
include("types.jl")      # DeviationResult, DevParams, _make_result, helpers
```

**After**:
```julia
include("types.jl")      # DeviationResult, DevParams, helpers
```

**Why**: `_make_result` was removed (item 1 above); the comment was updated to
match.

---

## Structural notes (no change made)

### `_edf_dispatch` in `stats.jl` duplicates EDF logic from `DevParams`

`_edf_dispatch` (called only by the public `edf_for_result`) re-encodes the
EDF parameters (d, F, is_total, total_type) by switching on the method string.
The same information is carried in each deviation's `DevParams` struct, which
the engine uses directly. This is an intentional trade-off: keeping `edf_for_result`
as a self-contained public API that does not require access to `DevParams`.
No change recommended unless `DevParams` is exposed publicly.

### `data_type` lowercase normalisation called repeatedly in `noise.jl`

`lowercase(data_type)` is evaluated at every branch point inside
`_noise_id_lag1acf` and `_noise_id_b1rn`. The engine always passes `"phase"` (already
lowercase), so these calls are no-ops in practice. A single normalisation at the
top of each function would be cleaner but is a minor style issue, not a bug.

### Legacy `matlab/+sigmatau/` does not exist yet

The CLAUDE.md architecture describes a refactored `matlab/+sigmatau/` package.
At the time of this audit only `matlab/legacy/` is present; no MATLAB refactor
has been written, so there is nothing to audit or clean up on the MATLAB side.
