# Audit 02 — Architecture, Module Boundaries, and the "One Package or Two" Question

**Scope:** module boundaries across Julia/MATLAB/Python/ML subsystems, one-vs-two-package verdict, MATLAB engine-refactor asymmetry, optimizer default divergence, module-level dead code, CLI placement, ML subsystem placement. Performance, test-coverage, and API redesign are out of scope. No file except this one is modified.

---

## 1. Executive summary

The single most architecturally pathological thing in this repo is that the stability-analysis half (deviations + noise-id + MHDEV fit) and the KF/ML half (clock models + filter + optimize + ALS + ml_features + noise_gen) have **zero code-level coupling** — no symbol in `julia/src/{clock_model,filter,predict,optimize,als_fit}.jl` imports or calls any symbol from `julia/src/{deviations*,noise*,noise_fit,stats,engine,validate,types}.jl`, and the reverse direction has zero edges as well. The *only* file bridging the two halves is `julia/src/ml_features.jl`, which calls `adev/mdev/hdev/mhdev` at lines 49–52 and is itself the entry into the separately-managed `ml/` subsystem. SigmaTau advertises itself as an integrated stability-and-KF package, but the Julia source tree is shaped like two sibling packages that happen to share a module namespace, a README, and a CHANGELOG.

The single cleanest fix is **to acknowledge the cut that already exists in the code**: declare `SigmaTau` (Julia + MATLAB) to be the stability-analysis package (the ten deviations, noise_id, MHDEV fit, EDF/CI, noise generation), extract KF + ML features + clock models + prediction + ALS into `SigmaTauKF.jl` as a sibling Julia package that depends on `SigmaTau`, and either move the MATLAB `+kf/` package into a sibling MATLAB package or leave it in place with an explicit "legacy parity frozen" banner in the README. The ML feature-extraction subsystem (`ml/`) should become a separate repository entirely — its subprocess-based coupling to Julia via `ml/notebook.py:396-438` is already the coupling shape of an external consumer.

The question to answer before committing to any fix: **is MATLAB first-class or legacy-parity?** Every other architectural decision on this list resolves trivially once this one is settled. If MATLAB is first-class, you owe months of work to build `+sigmatau/+clock_model/`, `+kf/predict_holdover`, `+kf/als_fit`, `+ml/compute_feature_vector`, `+dev/mhdev_fit`, etc. If MATLAB is legacy-parity, every asymmetry finding below resolves with a banner and a `docs/archive/matlab_legacy.md` note, not new code.

---

## 2. Post-Prompt-1 state reconciliation

Between AUDIT_01 and this run: **nothing**. The chronology proves it:

- Last git commit: `9cd015d` at 2026-04-16 13:59:14 -0500 ("fix: clean up stale API artifacts from main→dev merge").
- `AUDIT_01_docs_and_drift.md` mtime: 2026-04-16 14:44:29.
- `git diff HEAD --stat`: empty. No uncommitted edits to tracked files.
- Every file AUDIT_01 flagged (README.md, scripts/julia/kf_pipeline.jl, docs/handbook/*, docs/design/kf_architecture.md) has a pre-AUDIT_01 mtime. The only recent mtime is `TODO.md` (13:59:25), which matches the last commit.
- The two untracked files relevant to the audit lifecycle (`AUDIT_01_docs_and_drift.md`, `docs/design/`) are at the repo root and in an untracked subdirectory, not in a commit.

The prompt says the repo "is not the one Prompt 1 audited." The evidence says it is. I'll treat Prompt 1's findings as still-live wherever they haven't been re-addressed here — the prompt's framing is wrong on this point and I am not fabricating fixes to match it.

What this tells me about my own intent: I wrote Prompt 1's audit, saw it, and did not act on the twelve-ish concrete drift items before running Prompt 2. Either the interim time was spent on other work (the untracked PDFs in `ml/` and `notebook.ipynb`'s Apr 16 14:21 mtime are consistent with notebook work on the ML side), or I am using the audit backlog to build up a picture before deciding where to act. Either way, the drift cost is now ~36 hours deeper than it was yesterday.

---

## 3. Module boundary map

### Julia `julia/src/` (16 source files + 4 in `deviations/`)

| File | Cluster | Imports (non-stdlib, non-StaticArrays/FFTW) | External callers (outside src/ and test/) |
|------|---------|---------------------------------------------|-------------------------------------------|
| `SigmaTau.jl` | (root module) | — | — |
| `types.jl` | stability | — | — |
| `validate.jl` | stability | — | `cli/SigmaTauCLI/src/loader.jl` |
| `noise.jl` | stability | `types.jl` | — (via dev wrappers) |
| `noise_gen.jl` | **KF/ML** | Random, FFTW | `ml/dataset/*.jl` |
| `stats.jl` | stability | — | — |
| `engine.jl` | stability | `types.jl`, `validate.jl`, `noise.jl`, `stats.jl` | — |
| `deviations.jl` | stability | includes `deviations/{common,allan,hadamard,total}.jl` | — |
| `deviations/*.jl` | stability | `engine.jl`, `types.jl` | `cli/SigmaTauCLI/src/commands/common.jl` (imports all 10); `ml_features.jl` (imports 4); `ml/dataset/*.jl`; `scripts/julia/*.jl` |
| `ml_features.jl` | **bridge** | calls `adev`, `mdev`, `hdev`, `mhdev` | `ml/notebook.py:405-417` (via subprocess) |
| `noise_fit.jl` | stability | — | `scripts/julia/mhdev_preview.jl`; `ml/dataset/*.jl` |
| `clock_model.jl` | **KF/ML** | LinearAlgebra, StaticArrays | `scratch_holdover.jl:42` only (outside tests) |
| `filter.jl` | **KF/ML** | LinearAlgebra, Statistics | `scratch_holdover.jl:43`; `docs/design/kf_architecture.md` (doc only) |
| `predict.jl` | **KF/ML** | LinearAlgebra | `scratch_holdover.jl:44` only |
| `optimize.jl` | **KF/ML** | LinearAlgebra, Statistics, StaticArrays | `scratch_holdover.jl:29` only |
| `als_fit.jl` | **KF/ML** | LinearAlgebra | `scratch_holdover.jl:35` only |

Cross-cluster edges: **one.** `ml_features.jl` (KF/ML cluster) calls `adev, mdev, hdev, mhdev` (stability cluster). Nothing else crosses.

### MATLAB `matlab/+sigmatau/`

| Package | Contents | Counterpart in Julia | Status |
|---------|----------|----------------------|--------|
| `+dev/` | 10 deviation wrappers + `engine.m` | `julia/src/deviations*` + `engine.jl` | Mirror (but engine signature diverged — see §5) |
| `+noise/` | `noise_id.m`, `generate.m`, `private/` | `julia/src/noise.jl`, partial `noise_gen.jl` | Mirror (single-component generate only; Julia has composite) |
| `+stats/` | 5 EDF/CI files | `julia/src/stats.jl` | Mirror |
| `+util/` | 4 validators | `julia/src/validate.jl` | Mirror |
| `+kf/` | `build_phi`, `build_Q`, `predict_covariance`, `predict_state`, `update_pid`, `kalman_filter`, `optimize` | `julia/src/{clock_model,filter,predict,optimize}.jl` | **Legacy paradigm.** Config-struct based; no `ClockModel` types, no `predict_holdover`, no `als_fit`, no `innovation_nll` public function, no `mhdev_fit`, no `ml_features`, no `noise_gen` composite |
| `+plot/` | **empty** | Plots.jl (in `[deps]`, not extension) | Dead directory |
| `+steering/` | **empty** | — | Dead directory |

MATLAB mirror coverage: **stability half: yes; KF half: pre-refactor frozen; ML half: entirely absent.** Prompt 1 Sweep 7 noted this; confirmed here unchanged.

### Python `ml/src/` and `scripts/python/`

`ml/src/` (library, 4 files + `__init__.py`):
- `loader.py` — HDF5 dataset reader for training data produced by `ml/dataset/generate_dataset.jl`. Imports: h5py, numpy, sklearn.
- `models.py` — RF + XGBoost wrappers. Imports: numpy, sklearn, xgboost.
- `evaluation.py` — metrics + UQ (RF ensemble variance, XGBoost quantile). Imports: numpy, pandas, sklearn, xgboost.
- `real_data.py` — GMR6000 phase-record loader + unit detection + window extraction. Imports: numpy (and `_adev_tau1` reimplements ADEV locally — small duplication with `julia/src/deviations/allan.jl`).

`scripts/python/` (7 scripts, user-facing):
- `plot_devs.py`, `plot_kf.py`, `plot_mhdev_preview.py` — plotting. `plot_kf.py` hard-codes CSV filenames produced by the **broken** `scripts/julia/kf_pipeline.jl` (Prompt 1 parking-lot item, still unaddressed).
- `mhdev_fit_interactive.py` — interactive MHDEV power-law fit UI.
- `parse_stable32.py` — Stable32 output parser.
- `generate_comprehensive_report.py` — Stable32/allantools/SigmaTau cross-validation.

Division of labor: `ml/src/` is ML-training library; `scripts/python/` is stability-analysis plotting + parsing. The two groups have no shared idioms and no shared imports. They are two unrelated Python codebases sitting in the same repo. Style consistency: both use `from __future__ import annotations` and Python 3 typing; neither imports the other.

### `ml/` as a subsystem

| Component | Language | Kind | Depends on | Depended on by |
|-----------|----------|------|------------|----------------|
| `ml/dataset/*.jl` | Julia | Driver scripts | `using SigmaTau` (5 of 7 files) | — |
| `ml/src/*.py` | Python | Library | HDF5 output of `ml/dataset/` | `ml/notebook.py`, `ml/tests/` |
| `ml/notebook.py` / `.ipynb` | Python + embedded Julia | Training/eval + real-data validation | `ml/src/`, Julia subprocess (ll.396–438) | — |
| `ml/tests/` | Python | pytest | `ml/src/` | pytest |
| `ml/data/` | HDF5 + CSV | Data | — | notebook, loader |
| `ml/STATE.md` | Doc | Sub-project state | — | — |
| `ml/README.md` | Doc | Setup + reproduction | — | — |
| `ml/ml_pipeline_spec_v2.md` | Doc | Design spec | — | — |

`ml/dataset/Project.toml` has **its own SigmaTau dependency declaration** (line 5 of the `[deps]` section) separate from `julia/Project.toml`. This is the honest architectural statement: `ml/dataset/` is a separate Julia project that consumes SigmaTau.

Coupling shape to the rest of the repo:
1. **File-based** (HDF5): `ml/dataset/generate_dataset.jl` writes `ml/data/dataset_v1.h5`; `ml/src/loader.py` reads it.
2. **Process-based** (subprocess): `ml/notebook.py:396-438` writes a ~25-line Julia program to a tempfile, spawns `julia --project=ml/dataset` to run it, parses a CSV from stdout. The Julia program itself depends on `SigmaTau.compute_feature_vector` and `HDF5`.

Both directions treat SigmaTau as an opaque external library.

### CLI `julia/cli/SigmaTauCLI/`

Separate Julia package (own `Project.toml`, own uuid, own `src/`). Deps beyond stdlib: `DelimitedFiles`, `Plots`, `REPL`, `SigmaTau` (path-linked to `../..`), `UnicodePlots`. Source modules: `SigmaTauCLI.jl`, `parser.jl`, `loader.jl`, `plotting.jl`, `types.jl`, `commands/{common,compute,data,output}.jl`. Uses only `SigmaTau: adev, mdev, hdev, mhdev, tdev, ldev, totdev, mtotdev, htotdev, mhtotdev, DeviationResult, validate_phase_data, validate_tau0`. **No KF symbol.**

---

## 4. "One package or two" verdict

Answering in order:

1. **Does the stability half depend on the KF/ML half?** No. Grep for `ClockModel|ClockNoise|optimize_nll|als_fit|kalman_filter|predict_holdover|innovation_nll|KalmanResult|HoldoverResult|build_phi|build_Q|build_H` in `julia/src/{deviations,noise,noise_fit,noise_gen,stats,engine,validate,types}.jl` and `julia/src/deviations/*.jl` returns **zero matches**. The stability half is a closed set under imports.

2. **Does the KF/ML half depend on the stability half?** Yes, but narrowly. The KF half proper (`clock_model.jl`, `filter.jl`, `predict.jl`, `optimize.jl`, `als_fit.jl`) calls **nothing** from the stability side — grep returns no matches. The single bridging file is `ml_features.jl`, which calls `adev, mdev, hdev, mhdev` at lines 49–52. That is the entire cross-module coupling of the Julia codebase.

3. **Is MATLAB a twin-language implementation of the whole package?** No. MATLAB mirrors only the stability half and a frozen pre-refactor KF. Specifically missing from MATLAB: `ClockNoiseParams`, `ClockModel{2,3,Diurnal}`, `predict_holdover`, `als_fit`, `innovation_nll` (exists privately inside `optimize.m`), `mhdev_fit`, `ml_features`, `noise_gen` composite, `CANONICAL_TAU_GRID`, `FEATURE_NAMES`. The MATLAB `+kf/` package is not an intentionally different design; `ml/STATE.md:72-74,430-432` describes the Julia refactor as an explicit replacement for the old `KalmanConfig` paradigm, and `docs/design/kf_architecture.md:13-16` (the only in-repo acknowledgement) says MATLAB is not migrated. MATLAB is frozen at the pre-2026-04-16 Julia paradigm, without a decision having been made about whether to follow.

4. **How is ML coupled to the rest?** Through `ml_features.jl` (published export `compute_feature_vector`) plus a subprocess shell-out. The hard-coded CSV paths in `scripts/python/plot_kf.py` tie to `scripts/julia/kf_pipeline.jl` which is still broken — that coupling chain is load-bearing for nothing real today. The `compute_feature_vector` subprocess invocation is the only working ML↔SigmaTau link, and it runs through a fresh Julia process each time rather than any library-level import.

5. **If you extracted the KF+ML half as `SigmaTauKF.jl`, what would the cut look like?**
   - **Moves to `SigmaTauKF.jl`:** `clock_model.jl`, `filter.jl`, `predict.jl`, `optimize.jl`, `als_fit.jl`, `ml_features.jl`, `noise_gen.jl`.
   - **Stays in `SigmaTau.jl`:** `types.jl`, `validate.jl`, `noise.jl`, `stats.jl`, `engine.jl`, `deviations.jl`, `deviations/{common,allan,hadamard,total}.jl`, `noise_fit.jl`.
   - **New dep:** `SigmaTauKF` adds `using SigmaTau` to `ml_features.jl` for the four deviation imports; everything else stands alone.
   - **MATLAB `+sigmatau/+kf/`:** either moves to a new `matlab/+sigmataukf/` package or stays under `+sigmatau/+kf/` with an explicit "legacy frozen" banner. Leaving it under `+sigmatau/+kf/` after the Julia cut would make the namespace actively misleading; the cleaner move is into `+sigmataukf/+kf/` as a one-time rename.
   - **Exports that move out of `SigmaTau.jl`:** 18 of the 48 current exports — `KalmanResult, kalman_filter, kf_filter, HoldoverResult, predict_holdover, optimize_nll, innovation_nll, als_fit, ClockNoiseParams, ClockModel2, ClockModel3, ClockModelDiurnal, build_phi, build_Q, build_H, sigma_y_theory, h_to_q, q_to_h, steady_state_covariance, steady_state_gain, nstates, generate_power_law_noise, generate_composite_noise, CANONICAL_TAU_GRID, CANONICAL_M_LIST, FEATURE_NAMES, compute_feature_vector` (some of these 27 symbols re-exported from the KF package; final count depends on granularity).

6. **Verdict.** **(c) — it's effectively two packages already.** The single piece of evidence that makes this (c) and not (a): grep for any KF-side symbol in the stability-side files returns zero hits. That is not "the packages happen not to have many shared imports"; that is "the packages share nothing." The one bridging edge (`ml_features.jl`) is itself on the KF/ML side of the cut.

   The disagreement isn't whether they're two packages — the code has already answered that. The disagreement is whether to ratify the existing cut as a formal sibling-package structure (option ⓒ-formalize) or preserve the shared namespace because it preserves branding and README coherence (option ⓒ-status-quo). The cost of ⓒ-status-quo: every refactor has to reason about the whole tree as if it were coupled when it isn't; the README and handbook keep lying about integration that doesn't exist (see Prompt 1 Sweep 7 verdict); the MATLAB asymmetry lives in a limbo of "is this legacy or is this pending work" that no one can resolve without the cut being named.

   Recommendation: ratify (c)-formalize. Extract `SigmaTauKF.jl` as a sibling Julia package; leave MATLAB `+sigmatau/+kf/` alone with a "legacy parity, frozen at pre-refactor" banner; extract `ml/` to its own repo (see §9). If you can't stomach extracting `ml/` yet, at minimum move `ml_features.jl` and `noise_gen.jl` out of the core package — they have zero dependents outside the ML pipeline and they're the reason the core module docstring (`SigmaTau.jl:1-13`) can't say what SigmaTau *is* in six lines.

---

## 5. Engine refactor asymmetry

Julia `engine.jl:20-22` declares the kernel contract as `kernel(x, m, tau0, x_cs) → (variance, neff)`. Julia kernels receive the prefix sum `x_cs` computed once in the engine. MATLAB `engine.m:11` declares the kernel contract as `@(x, m, tau0) → [variance, neff]`, 3-arg. MATLAB kernels do their own `cumsum([0; x(:)])` inside the kernel body.

**Is this architectural divergence or pending-work divergence?** Pending work. Evidence:
- `CLAUDE.md` § Architecture (visible from file paths in comments) is referenced by both engines as the shared "one engine" architecture. Neither engine's docstring says the signatures are intentionally different.
- `CHANGELOG.md` `[Unreleased]` claims "Engine kernel signature refactored to `(x, m, tau0, x_cs)` in both." Prompt 1 Sweep 3 row 1 caught this as half-false: Julia was, MATLAB was not. AUDIT_01 §5 row 1 is still live.
- No TODO.md entry names the MATLAB engine refactor as intentional.
- `docs/design/` contains only `kf_architecture.md` — no engine-design doc exists to claim MATLAB "stays simple" intentionally.

**Redundancy cost.** Grepping for `cumsum([0` in `matlab/+sigmatau/+dev/`:
- Files containing the pattern: `mdev.m`, `mhdev.m`, `htotdev.m`, `mhtotdev.m`, `mtotdev.m` — **5 kernels**.
- `adev.m`, `hdev.m` use a second/third-difference form that doesn't need prefix sums.
- `tdev.m` and `ldev.m` are thin wrappers around `mdev.m`/`mhdev.m`; they inherit the prefix-sum cost without adding one.
- `totdev.m` builds an extended-data array inside the kernel and then does its own cumulative logic.

Redundant work per engine call: up to 5× `cumsum(x)` over a full N-length vector. At N=2^17 (ml_features default) that's ~5 × 131k additions per engine run when a user calls multiple deviations on the same series. Julia's refactor eliminated exactly this cost.

**What this says about "one engine, ten deviations."** The README wording is aspirational. Julia realized it; MATLAB has ten kernels each with their own prefix-sum implementation plus a shared outer loop. "One engine" in MATLAB is the outer loop, not the arithmetic — the shared engine only handles validation, m-list defaulting, noise-id dispatch, EDF, bias, CI. The actual variance arithmetic is per-kernel and not shared.

**Parity implications.** Float summation in a different order gives different roundoff. MATLAB kernels do `cumsum([0; x(:)])` (pre-pad with 0), then index into the cumulative-sum array. Julia's engine computes `pushfirst!(x_cs, 0.0)` on the cumulative sum of `x` and passes into the kernel. Both produce identical intermediate sums for Float64 input *if the underlying cumsum implementation is identical*, which it likely is (both call BLAS-adjacent libs). The 2e-10 tolerance in `test_crossval_julia.m` is generous enough to absorb any differences — the test isn't precision-sensitive. Concretely: the MATLAB-Julia cross-validation finding would not flip on a refactor that moved MATLAB to the 4-arg kernel contract.

**Recommendation.** This is not a structural disagreement between the two implementations. It is half of a migration. Either finish it (preferred; the Julia engine demonstrates the target), or delete the `[Unreleased]` CHANGELOG line that claims it was finished in both. The MATLAB engine refactor should be on TODO.md; it is not.

---

## 6. Optimizer default divergence

`matlab/+sigmatau/+kf/optimize.m:13,46,55,71`: q_wpm = R is held fixed. `theta0 = [log10(q_wfm); log10(q_rwfm)]`, IRWFM optional. Docstring says "q_wpm (= R) is held fixed" and prints "(fixed, R)" at optimization start.

`julia/src/optimize.jl:187,211`: `optimize_qwpm::Bool = true`. `theta0` includes `log10(q_wpm)` by default. Julia's optimizer treats R as free by default.

**Which default is correct?** The question I owe an answer to: is R known from short-τ ADEV / datasheet / prior calibration, or is it unknown and needs to be fitted from the data? For a GMR6000 Rb, R is observable from the flat τ⁻¹ slope at short τ (WPM floor) and comes out within ~3 dB of the datasheet value before the filter ever runs. That's the canonical case in this codebase — R is known when the user reaches `optimize_nll`.

In the textbook Kalman-filter tuning literature (and in Zucca-Tavella's 2005 framing which `clock_model.jl:4-7` cites), R is treated as known and the diffusion parameters q₁, q₂, q₃ are what you optimize. Allowing R to be optimized is a "hyperparameter sweep" feature, not the canonical case. MATLAB's default matches the canonical case; Julia's does not.

**Is the divergence documented?** Grep `optimize.jl` and `optimize.m`: MATLAB docstring says q_wpm fixed. Julia docstring (`optimize.jl:180-183`) says only "Optimize parameters over NelderMead. Returns optimal ClockNoiseParams" — one line, no mention of which params get optimized. Julia's docstring doesn't name the default at all, which means a reader looking at the signature and running it against their MATLAB workflow will get silently different answers. AUDIT_01 §8 already flagged the docstring gap; the default divergence compounds it.

**Failure mode if a user calls both with the same data.** Silent wrong answer. The returned `ClockNoiseParams` objects will differ in `q_wpm`. The filter runs cleanly. The innovation NLL will be lower on the Julia result (by construction — more free parameters). The user will read both and conclude "Julia's optimizer is slightly better." That's the worst category of divergence: two correct implementations of two *different* objectives presented as the same thing.

**Fix direction.** **(a) Make Julia's default match MATLAB's.** Flip `optimize_qwpm::Bool = true` → `false`. Reason: the canonical KF tuning problem fixes R. Users who want the hyperparameter sweep can opt in with `optimize_qwpm=true`. That matches the textbook and it matches the established MATLAB behavior, which is the older and more heavily-used side of the implementation. Option (b) (flip MATLAB to optimize q_wpm) inverts a published MATLAB docstring and changes behavior for users who have been running MATLAB against their own calibrated R values. Option (c) (declare non-parity, document) preserves the divergence and burdens every user with a mental table. Option (d): none applicable.

Same flip applies to `als_fit.jl:31` — `optimize_qwpm::Bool = true` default — for symmetry with `optimize_nll`.

---

## 7. Module-level dead code

Treating "dead" as "no public entry point reaches this file/module":

| Item | Status | Recommendation |
|------|--------|----------------|
| `matlab/+sigmatau/+plot/` | Empty directory | **Delete.** README line 12 claims "In Julia, loaded as a package extension so `using SigmaTau` stays light" with no corresponding MATLAB plotting story; the empty directory is the promise of one that was never built. |
| `matlab/+sigmatau/+steering/` | Empty directory | **Delete.** Steering logic lives inside `matlab/+sigmatau/+kf/update_pid.m` already. The empty package is a vestige of an earlier layout. |
| `scripts/julia/kf_pipeline.jl` | Broken (AUDIT_01 §3); references 6 removed symbols | **Rewrite** against the new `optimize_nll` / `predict_holdover` / `ClockModel3` API, or **archive to `docs/archive/scripts/`** with a banner. Do not leave in `scripts/julia/` routed to by README line 70 and handbook. |
| `ml/dataset/generate_dataset.jl` | Broken — line 86 calls `optimize_kf_nll` which was removed in `1789413` | **Rewrite** to `optimize_nll(...)`. (New finding beyond AUDIT_01: that audit dismissed these hits because it was searching for `optimize_kf`, not `optimize_kf_nll`. Both names are removed.) |
| `ml/dataset/real_data_fit.jl` | Broken — line 109 same issue | **Rewrite.** |
| `ml/dataset/real_data_fit_file2.jl` | Broken — line 116 same issue | **Rewrite.** |
| `scratch_holdover.jl` (repo root) | **Only** caller of new KF API outside tests and `docs/design/kf_architecture.md` | **Promote to `examples/kf_holdover.jl`** or `scripts/julia/kf_holdover.jl`. Leaving it as a scratch file at repo root, while it's the only working demo of the new KF surface, actively hides the API. |
| `julia/scratch_als.jl` | Scratch at Julia package root | **Delete** or promote to `examples/`. |
| `docs/design/kf_architecture.md` | Untracked in git (per `git ls-files docs/design/`); describes the redesign as if current, references pre-refactor line numbers (`filter.jl:69-85` etc.) | **Move to `docs/archive/kf_architecture_2026-04-16.md`** and commit with a tombstone banner. Its current state (untracked + pre-refactor line numbers + `KalmanConfig` references + uses `build_phi!/build_Q!` names that no longer exist) is the worst of three categories: it's a living file (not archived) and isn't version-controlled and isn't current. |
| `scripts/julia/generate_comprehensive_report.jl` | Not linked from README/handbook (AUDIT_01 §6) | Low priority — the Python twin `generate_comprehensive_report.py` is handbook-linked and likely the user-facing path. **Delete** the Julia version if confirmed redundant. |
| `scripts/python/parse_stable32.py` | Not linked from README/handbook (AUDIT_01 §6) | **Promote** via a one-paragraph handbook entry or confirm it's internal to `generate_comprehensive_report.py`. |
| `julia/ext/` | Empty directory, Project.toml has no `[weakdeps]`/`[extensions]` (AUDIT_01 §4 row 6) | **Delete.** The README claim is independent of whether the directory stays. |
| `julia/scripts/gen_crossval_data.jl` | Lives in `julia/scripts/`, not `scripts/julia/` — two conventions (AUDIT_01 parking lot) | **Move to `scripts/julia/` or `julia/test/fixtures/`** for consistency. |
| `ml_notebook_exec.log` (repo root) | Tracked log file | Parking-lot for hygiene — not architecture. |

Tombstone check: nothing else in `docs/design/` or `docs/equations/` rises to the same level as `kf_architecture.md`. `docs/equations/kalman.md` has out-of-date symbol references (AUDIT_01 parking lot noted `build_phi!`/`build_Q!`) but the file is a mathematical reference, not a design doc — the fix is a find-and-replace, not an archive.

---

## 8. CLI placement

`julia/cli/SigmaTauCLI/` is a separate Julia package (own uuid `03c35b2b-…`, own `Project.toml`, path-linked to `../..` for SigmaTau). Its `src/` contains: `SigmaTauCLI.jl`, `parser.jl`, `loader.jl`, `plotting.jl`, `types.jl`, `commands/{common,compute,data,output}.jl`.

**Why separate and not submodule of SigmaTau.jl?** Unstated. No design doc names the rationale. The weak-form argument is that the CLI depends on `UnicodePlots` and `REPL`, which SigmaTau should not. The strong-form argument is that the CLI is an end-user tool and not a library. Both arguments are fine but neither is written down.

**What it depends on.** `DelimitedFiles`, `Plots`, `REPL`, `SigmaTau`, `UnicodePlots`. From SigmaTau specifically: only the ten deviation wrappers, `DeviationResult`, `validate_phase_data`, `validate_tau0`. **No KF symbol.** The CLI is a pure deviation-analysis tool.

**Is there meant to be a MATLAB CLI or Python CLI?** No evidence. `bin/sigmatau` resolves to the Julia CLI. `scripts/matlab/` and `scripts/python/` contain function-style scripts, not CLIs. If MATLAB were first-class, you'd expect a `matlab/+sigmatau/+cli/` or a `bin/sigmatau_matlab` — neither exists. This is consistent with Julia being the primary CLI language and MATLAB being source-reachable only through its own scripts.

**Does the CLI reach into KF or ML?** No. grep of `julia/cli/SigmaTauCLI/src/` for `kalman_filter|optimize_nll|als_fit|predict_holdover|ClockModel` returns zero matches. The CLI is a stability-analysis-only tool that happens to live inside the same repo as a KF subsystem it doesn't touch.

**Placement verdict.** The CLI as a separate package is fine. Under a formalized cut (§4), the CLI would depend on `SigmaTau` (stability) only — it would not need to pull in `SigmaTauKF`. That's additional evidence for the cut: the CLI already chose its side.

---

## 9. ML subsystem placement

`ml/` is a sub-project grown out of coursework (PH 551 — per user memory) with a real external driver (GMR6000 phase records in `reference/raw/`). Decomposing the question:

1. **Is `ml/` a sub-project, a sibling project, or a coursework artifact?** It behaves as a sibling project with coursework origin. Evidence: own `Project.toml` in `ml/dataset/` with SigmaTau as a dep (not a development path), own `requirements.txt`, own `.venv/`, own `STATE.md`, own `README.md`, own `tests/`, own `.gitignore`. It is a complete mini-repo embedded inside SigmaTau.

2. **Subprocess shell-out stability.** `ml/notebook.py:396-438` writes a ~25-line Julia script into a tempfile at runtime, then spawns `julia --project={ml/dataset} --threads=auto <tempfile>`. The invocation is stable in shape (it writes HDF5 in → CSV out) but fragile at the edges: the script text is embedded as a Python string literal, Julia's col-major / h5py's row-major order mismatch is handled inline at lines 413–414 and the comment (`h5py wrote (n_windows, window_size) C-order; Julia reads column-major → (window_size, n_windows)`) would rot the moment either library changes. It's not versioned against a specific SigmaTau version — it assumes `compute_feature_vector(view(windows, :, i), 1.0)` matches whatever is in `julia/src/ml_features.jl` at run time.

3. **`ml/STATE.md` scope.** Scoped to ML sub-project. This is genuinely sub-project-scoped (timeline of h-range tuning, quantization-dominated real-data diagnostic, RF/XGB results, specific dataset version `dataset_v1.h5`). It does *not* duplicate what a repo-level `STATE.md` would cover. But its existence is also evidence that the user recognizes `ml/` as a thing that needs its own state tracking — that's already most of the way to "it's a separate project."

4. **Reproduction instructions.** `ml/README.md` exists (5352 bytes) and describes setup, h-ranges, schema, reproduction steps. Against the current state of `julia/src/`: it drives through `ml/dataset/generate_dataset.jl`, which is **broken** (line 86: `optimize_kf_nll`). A user following the README today gets a MethodError at step one. This is architectural rot: sub-project README stale against parent refactor.

5. **If `ml/` were extracted to a sibling repo `SigmaTauML`.** Breakage surface:
   - `ml/dataset/*.jl` would need SigmaTau as a registered package (via local path during transition) — trivial.
   - `ml/notebook.py`'s subprocess call would resolve SigmaTau via the external project's Project.toml — trivial.
   - `compute_feature_vector` and `CANONICAL_TAU_GRID`, `CANONICAL_M_LIST`, `FEATURE_NAMES` would need to be guaranteed-stable in SigmaTau (or SigmaTauKF). They already have frozen values, so this is a documentation obligation, not a code one.
   - Subsystem tests (`ml/tests/`) travel with the subsystem, no change.
   - `reference/raw/` would need to be either duplicated or lifted to a shared data directory. Given these are ~100 MB of GMR6000 phase records, a shared `data/` or external download would be better than duplication.

   Minimum public-API surface SigmaTau would need to freeze: the feature-extraction API (`compute_feature_vector`, `CANONICAL_TAU_GRID`, `CANONICAL_M_LIST`, `FEATURE_NAMES`, plus the 10 deviation wrappers and `DeviationResult`). That is already what SigmaTau is exporting.

**Verdict.** Extract `ml/` to a sibling repo `SigmaTauML`. Subprocess coupling is already the coupling shape of an external consumer. `ml/dataset/` has its own `Project.toml` already. Coursework is not a reason to keep it in — it's a reason to make it easy to hand off when the course is over. Leaving it in causes visible rot (broken driver scripts against parent refactor) and silent cost (parent audits must reason about a subsystem that doesn't belong).

If extracting `ml/` is too disruptive right now, at minimum: **promote `ml_features.jl`'s `CANONICAL_*` and `FEATURE_NAMES` to a declared stable-API block in SigmaTau's exports** (the one thing that must not churn for `ml/` to keep running), and **add a test that verifies `compute_feature_vector` still produces 196 features in the canonical order** so refactors of the stability side can't silently break the ML pipeline.

---

## 10. Parking lot for Prompt 3

Observations noted during this pass, out of scope for this audit:

- **Docstring decay (API level).** `optimize_nll` docstring (`optimize.jl:180-183`) is one line, doesn't mention cost function or which params are optimized. `als_fit` mentions Åkesson/Odelson but not its relation to `optimize_nll`. Prompt 3 candidate: API redesign that clarifies when to use which of `mhdev_fit`, `optimize_nll`, `als_fit`.
- **Performance: 5 MATLAB kernels recompute `cumsum`** (§5). If MATLAB engine is refactored, that's a measurable speedup at large N. Benchmark candidate for Prompt 3.
- **Test-coverage gap.** No cross-validation test exists for the KF half between MATLAB and Julia — only for deviations. The KF defaults diverge (§6) yet neither test suite catches it because they're not tested against each other. If the cross-validation claim in README line 5 were taken seriously, the KF would fail it today.
- **Test-coverage gap (ML).** No test exercises the `ml_features.jl` → Julia subprocess → Python CSV path end-to-end. The coupling is the highest-risk path in the repo and it has no regression coverage.
- **Signature redesign candidate.** Julia `kalman_filter(data, model; ...)` uses keyword args; MATLAB `kalman_filter(data, config)` uses a struct. If MATLAB stays first-class, a thin `kalman_filter(data, q_wpm, q_wfm, q_rwfm; ...)` Julia signature that matches MATLAB's struct fields would give a parity axis the tests could verify. If MATLAB is declared legacy, skip.
- **`SigmaTau.jl:1-13` module docstring** lists 6 of ~48 exports (AUDIT_01 parking lot; still unfixed). Prompt 3 candidate: regenerate.
- **`nstates` asymmetry inside MATLAB.** `kalman_filter.m` supports `nstates ∈ {2,3,5}`; `optimize.m` supports only `{2,3}` (AUDIT_01 parking lot; still unfixed).
- **`scripts/python/plot_kf.py` hard-codes CSV paths.** Chain is `scripts/julia/kf_pipeline.jl` (broken) → CSVs → plot_kf.py (unreachable). Prompt 3 candidate: API redesign of the plotting-data-handoff contract.
- **`Project.toml` compat `julia = "1.8"`.** Predates `[weakdeps]` stability. If the plots-extension claim (README line 12) is ever made true, the compat floor must move to 1.9+.
- **Repo hygiene.** Untracked PDFs in `ml/` (5 papers + a .docx), `notebook.ipynb` at 1.1 MB, `.pytest_cache/` and `.venv/` not gitignored. Parking-lot only.

---

## 11. The one thing I'd push back on

**Section 9's recommendation: extract `ml/` to a sibling repo `SigmaTauML`.**

I expect the pushback: *"`ml/` is active coursework — PH 551 final project, dataset v1 just generated, notebook and real-data validation just stabilized. Extracting it right now would disrupt a working iteration loop, add repo-management overhead I don't have time for, and separate the ML work from the reference `reference/raw/` GMR6000 phase records it needs."*

Preemption: the current shared-repo arrangement is not protecting the iteration loop — it's exposing it to parent-repo rot. Three pieces of evidence:

1. `ml/dataset/generate_dataset.jl:86` and the two `real_data_fit*.jl` files call `optimize_kf_nll`, which was removed in commit `1789413` on 2026-04-16. Any attempt to regenerate the dataset today fails at step one. The coupling isn't protecting reproducibility; it's a liability. AUDIT_01 missed this because it was checking for the shorter name `optimize_kf`, which has a related but distinct rename history. A sibling repo would pin SigmaTau to a specific commit and the dataset-generation script would still work against that pin.

2. The subprocess shell-out in `ml/notebook.py:396-438` is not evidence of integration — it's evidence that `ml/` already treats SigmaTau as external. An integrated design would import SigmaTau-generated features from a Python binding or from a pre-computed HDF5 artifact; instead, `ml/notebook.py` writes a Julia script to a tempfile and spawns a fresh process. That is the code you write when you're consuming a package, not when you're a member of it.

3. `ml/dataset/Project.toml` declares `SigmaTau` as a dependency with a separate manifest from `julia/Project.toml` (the `ml/dataset/Manifest.toml` is 48 KB of locked versions — its own dependency universe). The ML subsystem is already managing its own Julia project explicitly. Extracting it to a sibling repo is formalizing what the `Project.toml` is already saying.

The coursework-disruption worry is real, but it argues for *when* to extract, not *whether*. A reasonable staging: pin `ml/dataset/Project.toml` to the current SigmaTau commit, fix the three broken scripts against `optimize_nll`, then extract after the PH 551 submission deadline. Leaving `ml/` in the repo past that point costs more than the one-time extraction does — because every future SigmaTau refactor has to check whether it breaks the three ML driver scripts, and the audit evidence says this check is not reliably performed.

Second-candidate pushback (not my primary one): **"MATLAB is first-class, not legacy-parity."** I preempt by pointing at the evidence chain in §3 and §4.3: Julia exports `ClockModel{2,3,Diurnal}`, `predict_holdover`, `als_fit`, `innovation_nll`, `mhdev_fit`, `compute_feature_vector`, `generate_composite_noise` — seven significant APIs without MATLAB equivalents. Bringing MATLAB to parity is 10+ new files across 3 packages. Declaring MATLAB legacy-parity is a banner. The half-committed state — README promises parity, code doesn't deliver — is the worst option and has been the state since the 2026-04-16 refactor. I expect disagreement on *which* way to commit, not on *whether* to. My recommendation is declare-legacy because the Julia side has the momentum; but the prompt's "don't treat MATLAB as a lesser citizen" clause tells me you may push the other way, and if you do, §3 is the punch list.
