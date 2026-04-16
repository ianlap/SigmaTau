# Audit 01 — Documentation-vs-Reality and Stale-API Sweep

**Scope:** documentation claims vs. code reality; old-KF-API drift; shape divergence between MATLAB and Julia KF APIs. Architecture, performance, and test-coverage are out of scope (parked for later audits). No file except this one is modified.

---

## 1. Executive summary

The biggest readability hazard is **`README.md` lines 5, 11, 12, 35, 42, 70**. The "twin-language implementations mirror each other and cross-validate to machine precision" claim is half-true (deviations only, with a 2e-10 tolerance, silently skipped when the Julia reference file is missing). The "Q-parameter grid-search optimizer" is Nelder-Mead in both languages, never a grid search. The "In Julia, loaded as a package extension" claim is flatly contradicted by `julia/Project.toml`, which has no `[weakdeps]`/`[extensions]` and lists `Plots` as a hard dep; `julia/ext/` is empty. The MATLAB usage example `result = sigmatau.kf.pipeline(phase, tau0)` is **fictional** — there is no `matlab/+sigmatau/+kf/pipeline.m`. The Julia usage example `kf = kf_pipeline(phase, tau0)` names a script filename as if it were a function; the actual script needs `<dataset>` as a positional arg and is additionally **broken** on the current public API.

The second hazard is **`scripts/julia/kf_pipeline.jl`**. It is advertised by `README.md:70`, `docs/handbook/julia_scripts.md:34-40`, and `docs/handbook/workflows.md:24-25`, and is referenced by every Python plotter. It calls `KalmanConfig` (line 141), `PredictConfig` (151), `kf_predict` (152), `OptimizeConfig` (167, 235), `optimize_kf` (175), and `SigmaTau._kf_nll` (249) — every one of those symbols was removed in the 2026-04-16 refactor. The only repo file that acknowledges this is `ml/STATE.md:434`, which is buried under the ML subproject. The handbook and README still route users into it with no warning.

The third hazard is **reachability of the new Julia KF surface**. `optimize_nll`, `als_fit`, `predict_holdover`, `innovation_nll`, and the `ClockModel{2,3,Diurnal}` / `ClockNoiseParams` types are exported from `SigmaTau` but are **only called from `julia/test/` and the repo-root `scratch_holdover.jl`**. No script, no handbook page, no CLI command, no example uses any of them. A reviewer looking in on this package from the outside sees the MATLAB pipeline working end-to-end and the documented Julia pipeline broken, with no user-facing entry point to the replacement API.

If I had 20 minutes: (1) delete or rewrite `README.md` lines 12, 35, 42 and fix the "grid-search" wording on line 11; (2) stamp a `BROKEN: pending refactor` note across `scripts/julia/kf_pipeline.jl`, `docs/handbook/julia_scripts.md`, and `docs/handbook/workflows.md` §2; (3) either flip `Plots` to a weakdep and populate `julia/ext/` or delete the extension claim. Everything else on the list is smaller.

---

## 2. Recon inventory

**Repo root (`ls /home/ian/SigmaTau/`):**
- Tracked files: `README.md`, `CHANGELOG.md`, `TODO.md`, `.gitignore`
- Tracked dirs: `bin/`, `docs/`, `examples/`, `julia/`, `matlab/`, `ml/`, `reference/`, `scripts/`
- Tracked scratch at root: `scratch_holdover.jl`, `ml_notebook_exec.log`
- `.gitignore`d but present on disk: `CLAUDE.md`, `GEMINI.md`, `.claude/`, `.gemini/`, `devplans/`
- Not in repo: `.venv/`, `.pytest_cache/` (these are not in `.gitignore` — mild risk of accidental commit)
- **`STATE.md` at repo root: does not exist.** One exists at `ml/STATE.md` — scope is the ML sub-project, not the whole repo.

**`julia/` top-level:** `Project.toml`, `Manifest.toml` (gitignored), `cli/SigmaTauCLI/`, `ext/` (empty), `examples/`, `scripts/` (one file: `gen_crossval_data.jl`), `scratch_als.jl`, `src/`, `test/`.

**`julia/src/`:** `SigmaTau.jl`, `types.jl`, `validate.jl`, `noise.jl`, `noise_gen.jl`, `stats.jl`, `engine.jl`, `deviations.jl` + `deviations/{allan,hadamard,total,common}.jl`, `ml_features.jl`, `noise_fit.jl`, `clock_model.jl`, `filter.jl`, `predict.jl`, `optimize.jl`, `als_fit.jl`.

**`matlab/+sigmatau/`:** `+dev/` (10 dev wrappers + `engine.m`), `+noise/` (`noise_id.m`, `generate.m`, `private/`), `+stats/` (5 files), `+util/` (4 files), `+kf/` (`build_Q`, `build_phi`, `predict_covariance`, `predict_state`, `update_pid`, `kalman_filter`, `optimize`). **There is no `+kf/pipeline.m`.**

**`scripts/`:**
- `scripts/julia/`: `basic_usage.jl`, `compute_all_devs.jl`, `generate_comprehensive_report.jl`, `kf_pipeline.jl`, `mhdev_preview.jl`
- `scripts/matlab/`: `basic_usage.m`, `compute_all_devs.m`, `kf_pipeline.m`
- `scripts/python/`: `generate_comprehensive_report.py`, `mhdev_fit_interactive.py`, `parse_stable32.py`, `plot_devs.py`, `plot_kf.py`, `plot_mhdev_preview.py`
- `scripts/README.md` exists but is not linked from the top-level README or the handbook index.

**`docs/handbook/`:** `cli.md`, `index.md`, `julia_scripts.md`, `matlab_scripts.md`, `python_tools.md`, `workflows.md`.

**Julia public surface (`julia/src/SigmaTau.jl:24-42`):** `DeviationResult, DevParams, engine, compute_ci, bias_correction, edf_for_result, noise_id, mhdev_fit, MHDevFitResult, MHDevFitRegion, validate_phase_data, validate_tau0, unpack_result, adev, mdev, tdev, hdev, mhdev, ldev, totdev, mtotdev, htotdev, mhtotdev, KalmanResult, kalman_filter, kf_filter, HoldoverResult, predict_holdover, optimize_nll, innovation_nll, als_fit, ClockNoiseParams, ClockModel2, ClockModel3, ClockModelDiurnal, build_phi, build_Q, build_H, sigma_y_theory, h_to_q, q_to_h, steady_state_covariance, steady_state_gain, nstates, generate_power_law_noise, generate_composite_noise, CANONICAL_TAU_GRID, CANONICAL_M_LIST, FEATURE_NAMES, compute_feature_vector`. The module docstring (lines 7-13) lists only six of these — it is badly stale.

---

## 3. Stale-API reference table (Sweep 1)

Verdicts: **Live** = defined & caller works. **Stale** = caller references a removed symbol; caller is broken. **Fictional** = the symbol never existed. **Archival** = in a doc that explicitly describes pre-refactor state (design doc, STATE.md, old plan).

| Symbol | Location | Verdict | Note |
|--------|----------|---------|------|
| `KalmanConfig` | `scripts/julia/kf_pipeline.jl:141` | **Stale** | script broken |
| `KalmanConfig` | `docs/design/kf_architecture.md:9,32,64,200,255,409,454,470,595,609,662` | Archival | pre-refactor design doc |
| `KalmanConfig` | `ml/STATE.md:72,427,431` | Archival | refactor notes |
| `OptimizeConfig` | `scripts/julia/kf_pipeline.jl:167,235` | **Stale** | script broken |
| `OptimizeConfig` | `docs/design/kf_architecture.md` (7 lines) | Archival | |
| `OptimizeConfig` | `ml/STATE.md:427` | Archival | |
| `OptimizeConfig` | `docs/superpowers/plans/2026-04-14-ml-pipeline.md` (12 lines), `specs/2026-04-14-ml-pipeline-design.md:29` | Archival | historical ML plan |
| `OptimizeResult` | `docs/design/kf_architecture.md:203`; `ml/STATE.md:427`; superpowers plan (4 lines) | Archival | |
| `PredictConfig` | `scripts/julia/kf_pipeline.jl:151` | **Stale** | script broken |
| `PredictConfig` | `docs/design/kf_architecture.md:207`; `ml/STATE.md:73,428` | Archival | |
| `PredictResult` | `docs/design/kf_architecture.md:207`; `ml/STATE.md:428` | Archival | |
| `kf_predict` | `scripts/julia/kf_pipeline.jl:152` | **Stale** | script broken |
| `kf_predict` | `docs/design/kf_architecture.md:207,473`; `ml/STATE.md:73,428,432` | Archival | |
| `optimize_kf` | `scripts/julia/kf_pipeline.jl:12,175` | **Stale** | comment & live call |
| `optimize_kf` | `docs/design/kf_architecture.md` (6 lines); `superpowers/specs/...:26,108`; superpowers plan (5 lines) | Archival | |
| `optimize_kf` | `ml/dataset/real_data_fit.jl`, `real_data_fit_file2.jl`, `generate_dataset.jl` | hits are `optimize_kf_nll`, not `optimize_kf` — no true match |
| `_kf_nll` | `scripts/julia/kf_pipeline.jl:53,249` | **Stale** | `SigmaTau._kf_nll(...)` call on line 249 |
| `_kf_nll` | `docs/design/kf_architecture.md` (6 lines); `ml/STATE.md:146,151`; `superpowers/plans/...` (many) | Archival | |
| `kf_filter` | `julia/src/SigmaTau.jl:34,61`; `julia/test/test_filter.jl:16,32,33` | **Live** | alias `const kf_filter = kalman_filter` |
| `sigmatau.kf.pipeline` | `README.md:35` | **Fictional** | no `matlab/+sigmatau/+kf/pipeline.m` |
| `kf_pipeline` (Julia fn) | `README.md:42` | **Fictional as a function** | `kf_pipeline` is only a script filename; not exported from `SigmaTau`. `README.md:42` writes `kf = kf_pipeline(phase, tau0)` — that call form does not exist. |
| `kf_pipeline` (filename) | `scripts/julia/kf_pipeline.jl`, `scripts/matlab/kf_pipeline.m` | **Live as filenames** | MATLAB version works; Julia version broken |

### New-API caller reachability

`grep -n` for callers outside `julia/src/` and `julia/test/`:

| Symbol | External callers | Verdict |
|--------|------------------|---------|
| `optimize_nll` | `scratch_holdover.jl:29`; `docs/design/kf_architecture.md` (docblock) | No script, example, handbook, or CLI caller |
| `als_fit` | `scratch_holdover.jl:35`; `docs/design/kf_architecture.md` (docblock) | Same |
| `predict_holdover` | `scratch_holdover.jl:44`; `ml/STATE.md` (narrative); `docs/design/kf_architecture.md` | Same |
| `innovation_nll` | `docs/design/kf_architecture.md` (docblock) only | Tests-only reachable |

The new KF API is test-reachable and scratch-reachable; it is not user-facing-reachable.

---

## 4. README-to-reality diff (Sweep 2)

| # | README | Verdict | Evidence |
|---|--------|---------|----------|
| 1 | `README.md:5` — "The MATLAB and Julia implementations mirror each other and cross-validate to machine precision." | **Partially true, misleading wording.** Cross-validation exists for deviations only, at `REL_TOL = 2e-10` (not machine precision). Silently skipped when the Julia reference file is missing. KF is not cross-validated. | `matlab/tests/test_crossval_julia.m:10-14` (skip), `:47` (tol), `:34-45` (dev list only) |
| 2a | `README.md:11` — "Three-state clock model" | **True** but understated. Three models exist: `ClockModel2`, `ClockModel3`, `ClockModelDiurnal`. | `julia/src/clock_model.jl:26,36,46` |
| 2b | `README.md:11` — "Q-parameter grid-search optimizer" | **False.** Both languages use Nelder-Mead, not grid search. The only grid is a 9×9 diagnostic NLL-surface sample in `scripts/julia/kf_pipeline.jl:52-54`, which is not the optimizer. | `julia/src/optimize.jl:1,16,203,227` (`_nelder_mead`); `matlab/+sigmatau/+kf/optimize.m:2,13,68` (`fminsearch`) |
| 3 | `README.md:11` — "end-to-end pipeline (data → deviation → noise fit → KF)" | **Half-true.** MATLAB: `scripts/matlab/kf_pipeline.m` works end-to-end against the current MATLAB public API. Julia: `scripts/julia/kf_pipeline.jl` runs the same four stages but references removed symbols at every stage — see row 7. | `scripts/matlab/kf_pipeline.m:48,62,70`; `scripts/julia/kf_pipeline.jl:141-175` |
| 4 | `README.md:35` — `result = sigmatau.kf.pipeline(phase, tau0);` | **Fictional.** No `matlab/+sigmatau/+kf/pipeline.m` exists. | `ls matlab/+sigmatau/+kf/` (no pipeline.m) |
| 5 | `README.md:42` — `kf = kf_pipeline(phase, tau0)` | **Wrong.** `kf_pipeline` is not exported from `SigmaTau`. It is a script filename; its actual invocation is `julia --project=julia scripts/julia/kf_pipeline.jl <dataset>` (takes a dataset name, reads `reference/<dataset>.txt`, does not take `(phase, tau0)`). The script is additionally **broken**. | `julia/src/SigmaTau.jl:24-42` (no `kf_pipeline` export); `scripts/julia/kf_pipeline.jl:35,37` |
| 6 | `README.md:12` — "In Julia, loaded as a package extension so `using SigmaTau` stays light." | **False.** `Plots = ...` is in `[deps]`, not `[weakdeps]`; no `[extensions]` section exists; `julia/ext/` is empty. `using SigmaTau` loads Plots eagerly. | `julia/Project.toml:10,17-25`; `ls julia/ext/` empty |
| 7 | `README.md:70` — `scripts/julia/kf_pipeline.jl` — "Full noise-fit + KF optimization + filtering pipeline" | **Script exists but is broken.** Uses `KalmanConfig` (141), `PredictConfig` (151), `kf_predict` (152), `OptimizeConfig` (167, 235), `optimize_kf` (175), `SigmaTau._kf_nll` (249) — every one removed from the public API. | `scripts/julia/kf_pipeline.jl` line numbers above |
| 8 | `README.md:67` — `bin/sigmatau` — "Interactive CLI" | **Plausible; not executed.** Launcher resolves symlinks and activates `julia/cli/SigmaTauCLI`. CLI command table in `docs/handbook/cli.md:30-43` matches the actual `HELP_LINES` in `julia/cli/SigmaTauCLI/src/commands/output.jl:63-76`. Flag sets in `docs/handbook/cli.md:16-27` match `BOOLEAN_FLAGS`/`VALUE_FLAGS` in `julia/cli/SigmaTauCLI/src/parser.jl:7-17`. Not executed this pass. | cited files |

---

## 5. CHANGELOG verification (Sweep 3)

CHANGELOG `[Unreleased]` makes three specific claims. Verified:

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | Engine kernel signature refactored to `(x, m, tau0, x_cs)` in **both** `julia/src/engine.jl` and `matlab/+sigmatau/+dev/engine.m` | **Half-false.** Julia: refactored. Docstring declares the 4-arg kernel (`engine.jl:20`), `x_cs` is computed once (`engine.jl:63`), and kernels receive it (`engine.jl:68`). MATLAB: **not refactored.** Docstring still declares `@(x, m, tau0)` (`engine.m:11`), kernels are called with 3 args (`engine.m:72`), and every MATLAB kernel recomputes its own `cumsum([0; x(:)])` internally (e.g. `mdev.m:29,40`). The CHANGELOG's `x_cs` sharing benefit applies to Julia only. | cited lines |
| 2 | TDEV/LDEV as "architectural wrappers" around MDEV/MHDEV | **True in both.** Julia: `julia/src/deviations/allan.jl:124-132` (`tdev`), `julia/src/deviations/hadamard.jl:127-135` (`ldev`) — both ≤9 lines of wrapper + scale. MATLAB: `matlab/+sigmatau/+dev/tdev.m` (22 lines incl. docstring), `ldev.m` (22 lines) — both call the underlying wrapper and rescale. | cited files |
| 3 | `_to_indices` fix for non-contiguous vectors + regression test | **True.** Fix present at `julia/src/noise_fit.jl:183` (`_to_indices(r::AbstractVector{<:Integer})`). Regression test at `julia/test/test_noise_fit.jl:82-90` asserts `mhdev_fit` accepts `[1, 3, 7]`. | cited lines |

---

## 6. Handbook gap table (Sweep 4)

| File | Handbook page | Status |
|------|--------------|--------|
| `scripts/julia/basic_usage.jl` | `julia_scripts.md §basic_usage.jl` | Both |
| `scripts/julia/compute_all_devs.jl` | `julia_scripts.md §compute_all_devs.jl` | Both |
| `scripts/julia/generate_comprehensive_report.jl` | — | **Script without handbook page** |
| `scripts/julia/kf_pipeline.jl` | `julia_scripts.md §kf_pipeline.jl`, `workflows.md` §2 | Both exist; script is **BROKEN** (Sweep 1 row); handbook gives no warning |
| `scripts/julia/mhdev_preview.jl` | `julia_scripts.md §mhdev_preview.jl` | Both |
| `scripts/matlab/basic_usage.m` | `matlab_scripts.md` | Both |
| `scripts/matlab/compute_all_devs.m` | `matlab_scripts.md` | Both |
| `scripts/matlab/kf_pipeline.m` | `matlab_scripts.md` | Both |
| `scripts/python/generate_comprehensive_report.py` | `python_tools.md` | Both |
| `scripts/python/mhdev_fit_interactive.py` | `python_tools.md` | Both |
| `scripts/python/parse_stable32.py` | — | **Script without handbook page** |
| `scripts/python/plot_devs.py` | `python_tools.md` | Both |
| `scripts/python/plot_kf.py` | `python_tools.md` | Both |
| `scripts/python/plot_mhdev_preview.py` | `python_tools.md` | Both |

Handbook pages without an underlying script: none. All `docs/handbook/*.md` pages back onto real scripts or the real CLI.

**CLI drift:** none beyond what `TODO.md:30` already tracks. `parser.jl`'s `BOOLEAN_FLAGS`/`VALUE_FLAGS` and `output.jl:HELP_LINES` match the `docs/handbook/cli.md` command/flag tables on this pass.

**Cross-reference drift noted but not enumerated as CLI:**
- `docs/handbook/workflows.md:24-25` "Run KF optimization + filtering" lands on the broken Julia script.
- `docs/handbook/julia_scripts.md:34-45` describes that script without flagging its broken state.
- `scripts/python/plot_kf.py:46-47,62,273` hard-codes CSV filenames produced by `kf_pipeline.jl`; if users can't run that script, the plotter is unreachable too (unless the user manages to produce equivalent CSVs by hand).

---

## 7. Out-of-repo authority references (Sweep 5)

`.gitignore` excludes `CLAUDE.md`, `GEMINI.md`, `.claude/`, `.gemini/`, `devplans/`. All of those exist on disk. Non-moralized readability-cost table:

| Reference | Context quality for a reader without the external file |
|-----------|--------------------------------------------------------|
| `julia/src/engine.jl:38` "CLAUDE.md §Architecture" (freq→phase convention) | Fine. Next line spells out the rule (`cumsum(y)*tau0 produces phase in seconds`). |
| `matlab/+sigmatau/+dev/engine.m:36` same | Fine. Same inline explanation. |
| `julia/src/deviations.jl:2` "CLAUDE.md §Architecture" | Fine — context obvious from the file's content. |
| `julia/src/deviations/total.jl:184,217` "CLAUDE.md critical rule" | Fine. Comments spell out that htotdev at m=1 falls back to hdev. |
| `matlab/+sigmatau/+dev/htotdev.m:12,31,36` same | Fine. |
| `julia/test/test_total_family.jl:70` "CLAUDE.md critical rule: htotdev at m=1 must equal hdev at m=1" | Fine — full statement on the line. |
| `docs/equations/discrepancies.md:6,7` "CLAUDE.md flags potential off-by-one…", "CLAUDE.md flags: is segment count…" | **Poor.** Reader can see the symptom but not the original "flag" content; has to trust the follow-up verdict. |
| `docs/equations/total.md:58` "this is a documented intentional exception (CLAUDE.md)" | **Poor.** No inline justification; reader has to take it on faith. |
| `TODO.md:11` "to match GEMINI.md mandate §2" | **Poor.** Reader cannot know what §2 says, so cannot judge whether TODO is still valid. |
| `TODO.md:10` "the <100 line mandate" | **Poor.** Unattributed; no file to cross-check. |

---

## 8. `mhdev_fit` vs `optimize_nll` vs `als_fit` clarity check (Sweep 6)

Docstrings side-by-side:

- `mhdev_fit` (`julia/src/noise_fit.jl:75-99`) — clear: σ²-space, successive-subtraction power-law fit on user-declared τ-index regions, with a per-region diagnostic log.
- `optimize_nll` (`julia/src/optimize.jl:180-183`) — one-line signature + "Optimize parameters over NelderMead. Returns optimal `ClockNoiseParams`." No statement of the cost function, no relationship to `mhdev_fit` or `als_fit`, no guidance on when to use which.
- `als_fit` (`julia/src/als_fit.jl:8-21`) — names the ACS method and references Åkesson 2008 / Odelson 2006, but does not position itself relative to `optimize_nll`.

Joint mentions searched for: no README or handbook file mentions any two of these three together. The only in-repo file that uses two simultaneously is `scratch_holdover.jl` (lines 29, 35) — which warm-starts `als_fit` from `optimize_nll` without explaining why.

These three solve **different** problems:
- `mhdev_fit`: σ(τ)-space power-law decomposition from the MHDEV curve (returns `q_wpm, q_wfm, q_rwfm, q_rrfm, sig0_ffm, sig0_fpm`).
- `optimize_nll`: Gaussian innovation likelihood optimum in KF parameter space (returns `ClockNoiseParams`).
- `als_fit`: autocovariance-least-squares match of innovation moments (returns `ClockNoiseParams`).

A user could reasonably read the exports list and think they are three interchangeable fitters. The documentation does not disambiguate.

---

## 9. MATLAB-Julia KF shape diff (Sweep 7)

### Filter signatures

MATLAB (`matlab/+sigmatau/+kf/kalman_filter.m:1`):
```
function result = kalman_filter(data, config)
% config: struct with fields q_wpm, q_wfm, q_rwfm, q_irwfm, q_diurnal, R,
%                            g_p, g_i, g_d, nstates, tau, P0, x0, period
% nstates ∈ {2, 3, 5};  q_diurnal > 0 requires nstates = 5
```

Julia (`julia/src/filter.jl:149`):
```
kalman_filter(data::Vector{Float64}, model;
              x0 = Float64[], P0 = 1e6,
              g_p = 0.1, g_i = 0.01, g_d = 0.05) -> KalmanResult
# model ∈ {ClockModel2, ClockModel3, ClockModelDiurnal} — dispatches on type
```

Same mathematical filter, different software paradigm. MATLAB sits on a global config struct; Julia sits on dispatch over model types that own their `noise::ClockNoiseParams`.

### Optimizer signatures

MATLAB (`matlab/+sigmatau/+kf/optimize.m:1`):
```
[q_opt, results] = sigmatau.kf.optimize(data, cfg)
% cfg fields required: tau, q_wpm, q_wfm, q_rwfm; optional q_irwfm, nstates ∈ {2,3}
% q_wpm (= R) is held fixed; Nelder-Mead via fminsearch on log10-space (q_wfm, q_rwfm[, q_irwfm])
```

Julia (`julia/src/optimize.jl:184`):
```
optimize_nll(data, tau0;
             h_init=nothing, noise_init=nothing,
             optimize_qwpm=true, optimize_irwfm=false,
             verbose=true, max_iter=500, tol=1e-6) -> ClockNoiseParams
# Custom _nelder_mead; can optionally optimize q_wpm and/or q_irwfm
```

Different default semantics. MATLAB holds `q_wpm` fixed (= R). Julia optimizes `q_wpm` by default (`optimize_qwpm=true`). Same algorithm, opposite defaults.

### Julia-only KF-adjacent symbols (no MATLAB equivalent)

| Symbol | MATLAB equivalent? | Documented as Julia-only anywhere? |
|--------|--------------------|-------------------------------------|
| `ClockModel2`, `ClockModel3`, `ClockModelDiurnal` | no | only in `docs/design/kf_architecture.md:13-16` (buried) |
| `ClockNoiseParams` | no | same |
| `predict_holdover`, `HoldoverResult` | no | same |
| `als_fit` | no | same |
| `innovation_nll` | MATLAB has a **private** `kf_nll` inside `optimize.m:126` — not a public API | same |
| `h_to_q`, `q_to_h`, `sigma_y_theory`, `steady_state_covariance`, `steady_state_gain`, `nstates` | no | same |
| `generate_power_law_noise` | `sigmatau.noise.generate(alpha, N, tau0)` — partial equivalent (single-component only) | — |
| `generate_composite_noise` | no | — |
| `CANONICAL_TAU_GRID`, `CANONICAL_M_LIST`, `FEATURE_NAMES`, `compute_feature_vector` | no | — |
| `mhdev_fit`, `MHDevFitResult`, `MHDevFitRegion` | no | — |

`docs/design/kf_architecture.md:13-16` is the **only** in-repo document that states "MATLAB is not migrated as part of this redesign." The README, the handbook, and the module docstring all frame the package as symmetric.

### Verdict

The "twin-language" framing is **accurate for the deviation engine and noise_id only**. Beyond that — the entire KF, predict/holdover surface, ALS, feature extraction, composite-noise generation, and MHDEV fitter — Julia is a superset and MATLAB is frozen at the 2012-style `KalmanConfig` paradigm (even though `KalmanConfig` no longer exists on the Julia side, MATLAB still uses the struct-config shape). README and handbook do not flag this asymmetry.

---

## 10. Parking lot (out of scope; for later prompts)

Architecture, performance, test-coverage, and API redesign observations noted during this pass:

- `docs/design/kf_architecture.md` is a pre-refactor design doc describing a redesign that has since happened. Its filename under `docs/design/` makes it look current. It cites line numbers (e.g. `julia/src/filter.jl:69-85`) that no longer correspond to current code.
- `docs/equations/kalman.md:28,55` cites `julia/src/filter.jl:build_phi!` and `build_Q!`. The current code has `build_phi` / `build_Q` (no `!`) and they live in `julia/src/clock_model.jl`, not `filter.jl`. Documented symbol names and file locations both drifted.
- `SigmaTau.jl` module docstring (`julia/src/SigmaTau.jl:1-13`) lists 6 exports; the module exports ~48. Docstring decayed during the refactor.
- Julia `optimize_nll` optimizes `q_wpm` by default; MATLAB `optimize` holds it fixed. Defaults diverge without a named parity axis.
- MATLAB `kalman_filter.m` supports `nstates ∈ {2, 3, 5}` but MATLAB `optimize.m` supports only `{2, 3}`. Asymmetric internal surface.
- ML-feature code (`julia/src/ml_features.jl`, `julia/src/noise_gen.jl`, exports `CANONICAL_TAU_GRID`, `CANONICAL_M_LIST`, `FEATURE_NAMES`, `compute_feature_vector`) is part of the public surface but unmentioned in README and handbook. Its user-facing caller lives in `ml/notebook.py:396-417`, which shells out to Julia via `IOBuffer`-style subprocess — worth reviewing for the planned notebook workflow.
- `julia/scratch_als.jl` and `scratch_holdover.jl` sit at package and repo root respectively. Neither belongs in a clean checkout.
- `ml_notebook_exec.log` at repo root is a tracked log file.
- `.pytest_cache/` and `.venv/` at repo root are not in `.gitignore`. Mild commit-hygiene risk.
- `scripts/README.md` exists but is not referenced by the top-level `README.md` or the handbook index.
- `ml/STATE.md:434` is the only place that acknowledges `scripts/julia/kf_pipeline.jl` is broken. That's a project-lifecycle ergonomics problem (state captured only in sub-project notes).
- `scripts/python/plot_kf.py` and `plot_mhdev_preview.py` have hard-coded CSV paths tied to the broken Julia script; the coupling chain is longer than the pipeline's documented one.
- `gen_crossval_data.jl` lives at `julia/scripts/gen_crossval_data.jl`, not `scripts/julia/`. Two different `scripts/` directory conventions coexist.
- `julia/src/SigmaTau.jl:72` uses `cumsum(randn(Xoshiro(0), 64))` in the `@compile_workload`. Fine for precompile; just noting it for test-suite reproducibility concerns a later audit may want to check.
- `matlab/tests/test_crossval_julia.m:10-14` silently skips when `crossval_results.txt` is missing. A test that silently skips is a test that doesn't protect the cross-validation claim.
- `julia/Project.toml` compat bound `julia = "1.8"` — predates `[weakdeps]` stability; if you do flip Plots to an extension, the compat floor has to move to `1.9+`.
- Already-tracked TODOs (not novel): `NEFF_RELIABLE` 50→30, oversized MATLAB functions (`engine.m` 163 lines, `kalman_filter.m` 208, `optimize.m` 215), MHTOTDEV EDF approximation, MATLAB KF unit tests, Mtot multi-noise validation.

---

## 11. The one thing I'd push back on

The README-line-12 "package-extension" finding.

I expect you to say: the extension plan is real and tracked (`TODO.md:21` mentions expanding the Julia Plots extension), and the README was written in aspirational present tense so users know the direction of travel.

My preemption: the sentence as written — "In Julia, loaded as a package extension so `using SigmaTau` stays light" — is not forward-looking. It asserts current fact. And the fact is contradicted by three independent pieces of evidence: `Plots` is in `[deps]`, there is no `[weakdeps]` or `[extensions]` section in `Project.toml`, and `julia/ext/` is empty. A user who `instantiates` the project expecting a light install today gets the full Plots tree pulled in at `using SigmaTau`. That's not a roadmap gap; that's a wrong claim about current behavior. The fix is a four-word change ("will be loaded" → tense shift, or strike the sentence), not a roadmap defense.

Second-likely pushback candidate: flagging `scripts/julia/kf_pipeline.jl` as a novel finding when `ml/STATE.md:434` already notes it. My response to that in advance: the finding is not that the script is broken (you knew). The finding is that `README.md:70`, `docs/handbook/julia_scripts.md:34-40`, and `docs/handbook/workflows.md:24-25` — which are the entry points any external reader hits first — still route into it without a warning label. That mismatch between sub-project state and top-level docs is the actual finding, and it is the reason a reviewer would form a bad first impression of the package.
