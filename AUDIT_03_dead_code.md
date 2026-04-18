# Audit 03 — Dead Code, Consolidation, and Directory Structure

**Scope:** every tracked file in the repo except `.claude/`, `.git/`, `.github/`, and `julia/Manifest.toml`. No file except this one is modified. Produces a decision list for a later execution pass; the user will approve per-item before any deletion or archive.

---

## 1. Executive summary

The three earlier audits and four landed fixes have done most of the dead-code work already. The big surprise of this pass is that **two prominent findings from AUDIT_01/02 have been quietly closed and nobody told the audit documents**. `scripts/julia/kf_pipeline.jl` — flagged by AUDIT_01 §3 as broken against every post-refactor KF symbol — has been rewritten; a `grep` for `KalmanConfig|PredictConfig|OptimizeConfig|kf_predict|optimize_kf|_kf_nll` across `scripts/` returns zero hits, and the current script uses `optimize_nll` / `mhdev_fit` as the audits recommended. The three `ml/dataset/*.jl` driver files that AUDIT_02 §7 flagged as broken against `optimize_kf_nll` are likewise fixed — all three now call `optimize_nll` with comments preserving the pre-refactor semantic intent (`optimize_qwpm=false`). Those rows of AUDIT_01/02 are obsolete and should not carry forward.

With those two closed, the live dead-code universe shrinks dramatically. **Six rows** are confidently dead (default: delete): the 11 MB `ml/Ongoing Report on MC products.docx` (zero repo references), the zero-byte `ml/__init__.py`, the two untracked `*notebook_exec.log` files, the orphaned `mandate_rewrite_plan.md` at repo root (no references anywhere including FIX_PARKING_LOT.md), `ml/ml_pipeline_spec_v2.md` (superseded by `ml/STATE.md` with conflicting numbers), and `julia/scratch_als.jl` (scratch, no callers). A further **four rows** are user-decides flags: `docs/validation.md` (orphaned from navigation but content-current), `docs/papers/deviations/2019_riley_pseudo_flicker_floor.pdf` (uncited reference PDF — domain-knowledge call), `scripts/python/parse_stable32.py` (self-contained utility, no external callers), and the two `matlab/tests/test_allantools_*.py` (Python tests in a MATLAB directory with no documented runner — treat as one row). Plus one legacy-preservation row: `docs/design/kf_architecture.md` — which contrary to AUDIT_02 §7 **is** tracked (since commit `a138487`), and contains substantive pre-refactor rationale worth archiving rather than deleting. One scratch file (`scratch_holdover.jl`) is dead-at-its-location but live-by-reference (it's the only working demo of the new KF API outside tests, cited by the kalman-filter-julia skill); the recommendation is promote-to-`examples/`, not delete. One directory correction: AUDIT_02 §7 reported `matlab/+sigmatau/+plot/` and `+steering/` as empty — the truth is stricter, they do not exist as directories at all.

If cleanup were executed today on just the "definitely dead" subset: approximately **~11 MB of binary** plus **~4-6 KB of markdown / jl** removed, net line-change footprint probably 300-500 LOC (most of that is the stale `ml_pipeline_spec_v2.md` and `mandate_rewrite_plan.md`). The high-value wins are `.docx` deletion (repo size) and the small markdown orphans (readability). The one item where the recommendation deliberately deviates from delete is `docs/design/kf_architecture.md` — it's a tracked pre-refactor roadmap with substantive design rationale worth preserving, so the recommendation is "archive to `docs/archive/`" rather than delete.

---

## 2. File inventory

Counts of tracked files in scope by top-level category (total: 207 in scope; `.claude/` and `.github/` excluded from scope per the audit rules):

| Category | Count | Path |
|----------|-------|------|
| Julia source | 18 | `julia/src/` (14 top-level + 4 in `deviations/`) |
| Julia CLI | 13 | `julia/cli/SigmaTauCLI/` |
| Julia tests | 10 | `julia/test/` |
| Julia scripts | 5 | `scripts/julia/` |
| Julia scripts (split location) | 1 | `julia/scripts/gen_crossval_data.jl` |
| Julia examples/scratch | 2 | `scratch_holdover.jl` (root), `julia/scratch_als.jl` |
| MATLAB `+sigmatau/` source | 32 | 11 `+dev/` + 7 `+kf/` + 5 `+noise/` + 5 `+stats/` + 4 `+util/` |
| MATLAB tests | 15 | `matlab/tests/` (8 .m + 2 .py + 4 .txt fixtures + 1 runner) |
| MATLAB scripts | 3 | `scripts/matlab/` |
| Python library | 5 | `ml/src/` (incl. `__init__.py`) |
| Python tests | 5 | `ml/tests/` (incl. `__init__.py`) |
| Python scripts | 7 | `scripts/python/` |
| Python dataset drivers | 11 | `ml/dataset/` |
| Notebook + ML loose | 5 | `ml/notebook.py`, `ml/__init__.py`, `ml/STATE.md`, `ml/README.md`, `ml/requirements.txt`, `ml/.gitignore`, `ml/ml_pipeline_spec_v2.md`, `ml/Ongoing Report on MC products.docx` (plus ml/data/.gitignore) — roughly 9 |
| Docs (handbook) | 6 | `docs/handbook/` |
| Docs (equations) | 6 | `docs/equations/*.md` + `docs/equations.md` landing |
| Docs (design + validation + superpowers) | 6 | `docs/design/kf_architecture.md`, `docs/validation.md`, `docs/superpowers/{plans,specs}/*` (3 files) |
| Docs (papers — PDFs) | 11 | `docs/papers/{deviations,reference}/*.pdf` |
| Examples | 28 | `examples/{6krb25apr, drift_noisy_drift, mixed_noise_validation, noise_id_validation, plots}/` |
| Reference data | 6 | `reference/validation/` |
| Bin | 1 | `bin/sigmatau` |
| Top-level `.md` | 9 | README, CHANGELOG, CLAUDE, GEMINI, TODO, FIX_PARKING_LOT, AUDIT_01, AUDIT_02, mandate_rewrite_plan |
| Top-level misc | 2 | `scratch_holdover.jl`, `ml_notebook_exec.log` |
| Config | ~5 | `.gitignore` files, `Project.toml` files |

---

## 3. Dead-code decision list

### 3.1 Definitely dead (default: delete)

---
**File:** `ml/Ongoing Report on MC products.docx`
**Category:** Definitely dead
**Evidence:** 11 MB binary file in `ml/` root. Grep `"Ongoing Report on MC"` across repo returns no callers, no documentation citations, no `.gitattributes` lfs entry. Not referenced by `ml/STATE.md` or `ml/README.md`.
**Recommendation:** Delete.
**Rationale:** Monte-Carlo-products report artifact unrelated to active pipeline; 11 MB is a non-trivial repo-size burden.
**Related:** AUDIT_01 §10 parking-lot noted untracked PDFs; this is the only tracked one.

---
**File:** `ml/__init__.py`
**Category:** Definitely dead
**Evidence:** 0 bytes. `ml/src/__init__.py` is also 0 bytes. `ml/tests/__init__.py` is 0 bytes. None of them is needed — `ml/tests/` uses `sys.path.insert(0, ...)` for imports; there is no `import ml` or `from ml import X` anywhere in the repo.
**Recommendation:** Delete `ml/__init__.py` (keep `ml/src/__init__.py` + `ml/tests/__init__.py` since pytest auto-discovers).
**Rationale:** The `ml/` root `__init__.py` makes `ml` a Python package, but nothing imports `ml` — the code path is `ml/src/…` and `ml/tests/…` used as sibling directories, not a package.

---
**File:** `ml_notebook_exec.log` (repo root) and `ml/notebook_exec.log`
**Category:** Definitely dead (untracked log artifacts)
**Evidence:** Both are single-line jupyter-exec logs. Neither is in `git ls-files`. Per `git status` at start of this session, both sit untracked alongside the new PDFs and the `notebook.ipynb`. AUDIT_01 §10 parking lot flagged `ml_notebook_exec.log`; the second one (`ml/notebook_exec.log`) is new.
**Recommendation:** Delete from disk; add `*notebook_exec.log` pattern to `.gitignore`.
**Rationale:** Regenerable logs; no purpose serving either.
**Related:** AUDIT_01 §10 parking lot.

---
**File:** `mandate_rewrite_plan.md` (repo root)
**Category:** Definitely dead (or user archives)
**Evidence:** 21 KB markdown file. Grep `mandate_rewrite_plan` or `mandate-rewrite-plan` across repo returns **zero matches**. Not cited by `FIX_PARKING_LOT.md`, `TODO.md`, `CLAUDE.md`, `GEMINI.md`, `CHANGELOG.md`, or any doc.
**Recommendation:** Delete or move to `docs/archive/`.
**Rationale:** Lowercase filename at repo root (breaks the MANDATE-SHOUT convention of CLAUDE/GEMINI/AUDIT files). Per `CHANGELOG.md [Unreleased]` and `FIX_PARKING_LOT.md`, the mandate rewrite has been executed; the plan is spent.

---
**File:** `ml/ml_pipeline_spec_v2.md`
**Category:** Definitely dead (superseded with conflicting numbers)
**Evidence:** Design spec with `N=2¹⁷` and h-α ranges [-26.5, -23.5]. `ml/STATE.md` (active, dated 2026-04-15) describes current ranges [-19, -16] and `N=2¹⁹`. Grep `ml_pipeline_spec_v2` returns no callers.
**Recommendation:** Archive to `docs/archive/` or delete.
**Rationale:** `ml/STATE.md` is the load-bearing state doc now. A v2 spec whose numbers conflict with STATE.md is actively misleading — the spec is worse than "not consulted," it's "consulted and wrong."

---
**File:** `julia/scratch_als.jl`
**Category:** Definitely dead
**Evidence:** Scratch file at `julia/` root. Grep across repo returns zero callers outside AUDIT_01/02 commentary. Contents are Lyapunov-ALS basis-function experiments already folded into `julia/src/als_fit.jl`.
**Recommendation:** Delete.
**Rationale:** The experiment has been productized into `als_fit.jl`; the scratch no longer carries weight. AUDIT_02 §7 already recommended delete-or-promote.
**Related:** AUDIT_02 §7.

---

### 3.2 Probably dead (user decides)

---
**File:** `docs/validation.md`
**Category:** Probably dead (orphaned but content-current)
**Evidence:** Grep `docs/validation\.md|validation\.md` across repo returns zero matches from any README, handbook index, or source file.
**Recommendation:** User decides — link from `README.md` or handbook index if worth keeping; else delete.
**Rationale:** Content describes MATLAB↔Julia parity validation — accurate and current, but orphaned from any navigation entry. The content has value if connected to navigation; orphaned content has none. User's call.

---
**File:** `docs/papers/deviations/2019_riley_pseudo_flicker_floor.pdf`
**Category:** Probably dead (low confidence)
**Evidence:** Grep `2019_riley_pseudo_flicker` and `pseudo_flicker` across repo returns no files. Not cited by any markdown, equation doc, or code comment.
**Recommendation:** User decides. Archive if "keep for future TDEV work"; delete if no planned use.
**Rationale:** Reference-paper PDFs are legitimate even if nobody currently cites them — the user's domain knowledge is the right arbiter. All other 10 PDFs in `docs/papers/` are cited at least once.

---
**File:** `docs/design/kf_architecture.md`
**Category:** Legacy with historical value (archive, do not delete)
**Evidence:** **Correction to AUDIT_02 §7:** this file **is** tracked in git (`git log --follow` shows commits `d28a9b9`, `a138487`). AUDIT_02 was wrong to claim it was untracked. Content is a pre-refactor design plan that references removed symbols (`KalmanConfig`, `PredictConfig`, `build_phi!`/`build_Q!` with bang — current code has no bang). Per AUDIT_01 §3 stale-API table.
**Recommendation:** Move to `docs/archive/kf_architecture_2026-04-16.md` with a tombstone banner. Do not delete.
**Rationale:** Substantive design rationale. Its current filename under `docs/design/` and its stale symbol references make it actively misleading to a reader who stumbles onto it. Archiving preserves the institutional memory without the cost of "is this current?" confusion.
**Related:** AUDIT_01 §3 (stale-API table, multiple rows); AUDIT_02 §7.

---
**File:** `scripts/python/parse_stable32.py`
**Category:** Orphaned / Probably dead
**Evidence:** Defines `parse_stable32_output` used only inside the same file (line 75). No external imports, not mentioned in README or handbook (per AUDIT_01 §6), not called by `generate_comprehensive_report.py` (which calls the Julia script, not this Python one).
**Recommendation:** User decides — delete if the Julia wrapper now handles Stable32 parsing, else promote via handbook entry.
**Rationale:** Self-contained utility with an unclear caller story. May still be used ad-hoc by Ian at the command line; that's a knowledge call the audit can't make.
**Related:** AUDIT_01 §6 handbook gap table, AUDIT_02 §7.

---
**File:** `matlab/tests/test_allantools_adev.py`
**File:** `matlab/tests/test_allantools_stable32.py`
**Category:** Orphaned (unusual placement, unclear invocation)
**Evidence:** Python pytest files inside `matlab/tests/`. `matlab/tests/run_all.m` includes only the 8 `.m` test files, not these two. Grep for `test_allantools_adev` and `test_allantools_stable32` across repo returns only self-references. No CI config, no README entry, no pytest configuration pointing at this path.
**Recommendation:** Move to a new `tests/python/` or document the invocation path.
**Rationale:** Python tests that hardcode relative paths to MATLAB test fixtures, with no documented runner — the file placement suggests someone ran them interactively and committed them. Either they're live and need a documented `pytest matlab/tests/test_allantools*.py` runbook, or they're dead.

---

### 3.3 Orphaned example / Stale data (low-confidence flags)

---
**File:** `reference/validation/stable32gen.DAT`
**File:** `reference/validation/stable32out/*.{md,csv,txt}` (5 files)
**Category:** Live (verified, not flagged — listed here for completeness)
**Evidence:** `stable32gen.DAT` is cited by `docs/validation.md:*`, `docs/handbook/{cli,workflows,matlab_scripts}.md`, `scripts/README.md`, `scripts/python/generate_comprehensive_report.py:34`, `scripts/python/parse_stable32.py:11`, `matlab/tests/test_allantools_stable32.py:8`. The `stable32out/` files are generated by `generate_comprehensive_report.py` and consumed by `test_allantools_stable32.py:35`.
**Recommendation:** Keep all.
**Rationale:** Load-bearing fixtures. No action needed.

---
**File:** `examples/mixed_noise_validation/reference/*.csv` (10 CSV files) + `mixed_noise.txt`
**Category:** Live (verified)
**Evidence:** Reference dataset for the mixed-noise validation example (AUDIT_01 did not cover `examples/`). The spec in `docs/superpowers/specs/2026-04-13-mixed-noise-validation-dataset-design.md` describes the deliverable. `examples/mixed_noise_validation/generate.jl` uses current API (`adev, mdev, hdev, mhdev, totdev, mtotdev, htotdev, mhtotdev, tdev, ldev, noise_id`); no removed symbols.
**Recommendation:** Keep.
**Rationale:** Pre-computed reference data intentionally versioned; the generator is reproducible.

---
**File:** `examples/plots/{adev_wfm.png, sigmatau_adev.csv, wfm_phase.csv}`
**Category:** Live (regression artifacts)
**Evidence:** `examples/plots/generate.m` produces them; `plot_adev.py` overlays allantools.oadev for comparison. Tracked intentionally as regression targets.
**Recommendation:** Keep. (Not flagged.)

---

### 3.4 Resolved since prior audits (close these rows)

The following AUDIT_01/02 rows are obsolete. Verified by grep on current `dev` branch:

---
**Row:** AUDIT_01 §3 — `scripts/julia/kf_pipeline.jl` broken against `KalmanConfig/PredictConfig/OptimizeConfig/kf_predict/optimize_kf/_kf_nll`.
**Status:** **Closed.** `grep "KalmanConfig|PredictConfig|OptimizeConfig|kf_predict|optimize_kf|_kf_nll" scripts/` returns zero matches. Script uses `optimize_nll` and `mhdev_fit` as the audits recommended. The `FIX_PARKING_LOT.md` entry "scripts/julia/kf_pipeline.jl migration" on rolling-origin RMS scaling survives — that's a separate performance concern, unrelated to the API rewrite.

---
**Row:** AUDIT_02 §7 — `ml/dataset/{generate_dataset,real_data_fit,real_data_fit_file2}.jl` broken against `optimize_kf_nll`.
**Status:** **Closed.** All three files now call `optimize_nll`. The name `optimize_kf_nll` appears only in comments preserving historical context for the `optimize_qwpm=false` semantic.

---
**Row:** AUDIT_02 §7 — `matlab/+sigmatau/+plot/` and `+steering/` "empty directories."
**Status:** **Correction.** Those directories do not exist at all (verified via `ls matlab/+sigmatau/`: the children are `+dev`, `+kf`, `+noise`, `+stats`, `+util` — no `+plot`, no `+steering`). AUDIT_02's claim is imprecise, not wrong — the files promised by TODO.md "MATLAB Plotting Package" are absent. Nothing to delete here; the TODO still applies.

---
**Row:** AUDIT_02 §7 — `docs/design/kf_architecture.md` untracked.
**Status:** **Correction.** File is tracked. `git log --follow docs/design/kf_architecture.md` shows commits `d28a9b9` and `a138487`. AUDIT_02's factual claim is wrong. Its substantive recommendation (archive with a tombstone) still stands — see §3.2 above.

---
**Row:** AUDIT_02 §7 — `julia/ext/` empty directory.
**Status:** **Confirmed.** `ls -la julia/ext/` shows only `.` and `..`. Empty. Not in `git ls-files`. The README line 12 claim about "loaded as a package extension" is still false (see AUDIT_01 §4).

---

## 4. Consolidation observations

---
**Group:** Empty `__init__.py` proliferation
**Files:** `ml/__init__.py`, `ml/src/__init__.py`, `ml/tests/__init__.py` — all 0 bytes
**Overlap:** All three are empty package markers.
**Consolidation proposal:** Delete `ml/__init__.py` (nothing imports `ml` as a package). Keep the other two — `ml/src/__init__.py` supports `from ml.src import …` in tests; `ml/tests/__init__.py` supports pytest discovery.
**Cost:** Low. Single-file delete.

---
**Group:** Dual documentation on the equation family
**Files:** `docs/equations.md` (14 lines, landing page) + `docs/equations/index.md` (27 lines, bibliography & sigils)
**Overlap:** Both at `docs/equations*`. Not duplicates — `equations.md` is a top-level index with six outbound links; `equations/index.md` is the source-citation table and audit-status tracker. Different roles.
**Consolidation proposal:** Keep both. Not a consolidation target.
**Cost:** N/A.

---
**Group:** Parallel real-data fit diagnostics
**Files:** `ml/dataset/real_data_fit.jl`, `ml/dataset/real_data_fit_file2.jl`
**Overlap:** Near-identical structure — each loads one dataset, runs `mhdev_fit` + `optimize_nll`, writes CSV+PNG overlays. File-2 adds a combined-h-ranges CSV.
**Consolidation proposal:** Accept the duplication or parameterize as `real_data_fit.jl <dataset>`. Low value since these are one-off diagnostic scripts run by hand; consolidation would add cognitive overhead without real ergonomic gain.
**Cost:** Low-to-medium.

---
**Group:** Scripts-hierarchy split (well-known; AUDIT_02 flagged)
**Files:** `scripts/julia/*.jl` (5 files) + `julia/scripts/gen_crossval_data.jl` (1 file)
**Overlap:** Both hold Julia-language scripts; conventions diverge.
**Consolidation proposal:** Move `julia/scripts/gen_crossval_data.jl` → `scripts/julia/` (and update `CLAUDE.md` reference). **OR** accept the split as "tests fixtures live inside the Julia package, user-facing scripts live at the top-level `scripts/`." Either is defensible; the split as-is is a coin-toss for a reviewer.
**Cost:** Low (move + one `CLAUDE.md` edit).

---
**Group:** Two `__init__.py`-style stubs across MATLAB test runners
**Files:** `matlab/tests/test_allantools_adev.py`, `matlab/tests/test_allantools_stable32.py`
**Overlap:** Python pytest tests sitting inside a MATLAB test directory, invoked by nothing in `run_all.m`.
**Consolidation proposal:** Move both to a new `tests/python/` or integrate into `ml/tests/`. The more honest option: delete if not used.
**Cost:** Low.

---
**Group:** Top-level documentation sprawl
**Files:** `README.md`, `CHANGELOG.md`, `TODO.md`, `CLAUDE.md`, `GEMINI.md`, `FIX_PARKING_LOT.md`, `AUDIT_01_docs_and_drift.md`, `AUDIT_02_architecture.md`, `mandate_rewrite_plan.md`, `AUDIT_03_dead_code.md` (pending)
**Overlap:** 10 markdown files at repo root is unusual. Most are load-bearing (README, CHANGELOG, CLAUDE, GEMINI, TODO, PARKING_LOT, the three audits). `mandate_rewrite_plan.md` is the odd one.
**Consolidation proposal:** Delete `mandate_rewrite_plan.md` (see §3.1). Consider a `docs/audit/` subdirectory for the three AUDIT_*.md files once they are "finished." For now, leave them at root since they're active institutional memory.
**Cost:** Low.

---

## 5. Directory structure observations

1. **`scripts/` vs `julia/scripts/` vs `ml/dataset/`.** Three script homes. `scripts/{julia,matlab,python}/` holds user-facing command-line tools. `julia/scripts/` holds one file (`gen_crossval_data.jl`) intended for Julia-package-internal use. `ml/dataset/` is an 11-file Julia project with its own `Project.toml`. AUDIT_02 §9 recommended extracting `ml/dataset/` as a sibling package; I agree. The `julia/scripts/` single-file directory is a coin-toss (see §4).

2. **`docs/` four-way split.** `docs/handbook/` (user manual, 6 files), `docs/equations/` (math reference, 6 files), `docs/papers/` (11 reference PDFs), `docs/superpowers/{plans,specs}/` (3 workflow-docs files), plus loose `docs/design/kf_architecture.md`, `docs/validation.md`, and `docs/equations.md`. The handbook/equations/papers triple is principled. The `superpowers/` and `design/` and loose files are accretional — no `docs/archive/` exists to park the executed ones.

3. **Top-level markdown count of 10.** README/CHANGELOG/TODO/CLAUDE/GEMINI/PARKING_LOT are inherent and fine. Three AUDIT files are acceptable as active institutional memory. `mandate_rewrite_plan.md` is the sore thumb.

4. **`matlab/tests/*.py`.** Python tests in a MATLAB test directory. Unique in the repo to this location. Either `tests/python/` exists as a new convention (it doesn't), or the two `.py` files move into `ml/tests/`, or they leave.

5. **Two scratch files at package roots.** `/home/ian/SigmaTau/scratch_holdover.jl` (repo root) and `/home/ian/SigmaTau/julia/scratch_als.jl` (Julia package root). Both are the only demos of the new KF API outside tests — one of them at least deserves promotion to `examples/kf_holdover.jl`. The naming convention says "temporary"; the reality is "load-bearing demo." Mismatch.

6. **No `docs/archive/` directory.** A common destination for "executed plans, superseded specs, pre-refactor design docs." Would tidy `docs/superpowers/specs/2026-04-13-mixed-noise-validation-dataset-design.md` (executed) and `docs/design/kf_architecture.md` (superseded) in one move. This is the most productive single structural change available from this audit.

---

## 6. Cross-reference to known debt

### 6.1 Items this audit would close

| Source | Item | Status after this audit |
|--------|------|-------------------------|
| AUDIT_01 §3 row 1 | `scripts/julia/kf_pipeline.jl` calls `KalmanConfig/OptimizeConfig/PredictConfig/kf_predict/optimize_kf/_kf_nll` | **Closed** — script rewritten; grep proves no references to any of those symbols in `scripts/`. |
| AUDIT_02 §7 row | `ml/dataset/generate_dataset.jl` line 86 `optimize_kf_nll` broken | **Closed** — now calls `optimize_nll`. |
| AUDIT_02 §7 row | `ml/dataset/real_data_fit.jl` line 109 `optimize_kf_nll` broken | **Closed.** |
| AUDIT_02 §7 row | `ml/dataset/real_data_fit_file2.jl` line 116 `optimize_kf_nll` broken | **Closed.** |
| AUDIT_02 §7 row | `matlab/+sigmatau/+plot/` empty directory | **Correction** — does not exist as a directory; no cleanup needed. TODO.md "MATLAB Plotting Package" item stands. |
| AUDIT_02 §7 row | `matlab/+sigmatau/+steering/` empty directory | **Correction** — does not exist as a directory. |
| AUDIT_02 §7 row | `docs/design/kf_architecture.md` untracked | **Correction** — is tracked; recommendation (archive) stands. |

### 6.2 Items this audit would leave open for execution

| Source | Item | Notes from this audit |
|--------|------|-----------------------|
| FIX_PARKING_LOT.md | `scripts/julia/kf_pipeline.jl` rolling-origin RMS scaling | Still relevant. Unrelated to the API rewrite; performance concern. |
| FIX_PARKING_LOT.md | `scripts/python/plot_kf.py` CSV dir mismatch | Still relevant. |
| FIX_PARKING_LOT.md | `ml/notebook.py:463` pre-Wu h-derivation | Still relevant. |
| FIX_PARKING_LOT.md | `GEMINI.md §2.3` + Goal G2 MATLAB engine refactor stale | Still relevant; mandate-pass only. |
| FIX_PARKING_LOT.md | `matlab/tests/test_filter.m` Test 2 bias | Still relevant (xfail'd). |
| AUDIT_01 §4 row 6 | README line 12 package-extension claim | Still relevant (not in dead-code scope). |
| AUDIT_01 §3 rows | `docs/design/kf_architecture.md` archival symbol mentions | Addressed above — recommend archive. |
| AUDIT_01 parking lot | `scripts/README.md` not linked from top-level README | Still relevant; handbook-structure concern. |
| AUDIT_02 §7 | `scratch_holdover.jl` promotion to `examples/` | Still relevant (not deletion). |
| AUDIT_02 §9 | Extract `ml/` to sibling repo | Still relevant. |
| TODO.md | NEFF_RELIABLE 50→30 | Unchanged. |
| TODO.md | Oversized MATLAB functions | Unchanged. |
| TODO.md | MHTOTDEV EDF | Unchanged. |

### 6.3 Net new items surfaced by this audit

- `mandate_rewrite_plan.md` at repo root is orphaned (no references) and the mandate rewrite has landed per CHANGELOG `[Unreleased]` + FIX_PARKING_LOT. Candidate for deletion.
- `docs/validation.md` is orphaned from any navigation. Either link it (README §Validation or handbook) or delete.
- `matlab/tests/test_allantools_{adev,stable32}.py` are invoked by no runner; placement and provenance unclear.
- `docs/papers/deviations/2019_riley_pseudo_flicker_floor.pdf` has no citations. Low-confidence flag.
- `ml/__init__.py` is an empty package marker that nothing imports.
- `ml/Ongoing Report on MC products.docx` is an 11 MB binary with no references.

---

## 7. The one thing I'd push back on

The recommendation to delete `mandate_rewrite_plan.md`.

I expect the user to say: "it's institutional memory of *how* the mandate was rewritten, and we've preserved every other audit document at repo root for the same reason."

Preemption: the file is not institutional memory of how; it is the plan itself — the pre-execution design. Per CHANGELOG `[Unreleased]` and `FIX_PARKING_LOT.md:8`, the plan has been executed (MATLAB engine 4-arg migration completed 2026-04-16/17; skills written; mandates updated). A plan that has been executed without being referenced anywhere in the repo has zero informational content that isn't also in the resulting commits and updated mandates. Keeping it at repo root creates three costs: (1) lowercase filename breaks the MANDATE-SHOUT repo-root convention that CLAUDE/GEMINI/AUDIT files share; (2) it adds to the top-level-markdown sprawl without earning it; (3) a new reader sees "mandate_rewrite_plan.md" and wastes attention determining whether it's a live plan or a done one. The test is: does `grep -r mandate_rewrite_plan` find any reader? No. That's the definition of "institutional memory nobody reads," which is not memory at all.

Second-candidate pushback: deleting the 11 MB `.docx` file. If the user says "it's MC products analysis notes from coursework," my response is: the repo is not an archive for binary coursework artifacts — institutional value for that goes in git lfs or a shared drive, not here. But since the domain-knowledge test is the user's and not mine, I flag as "Definitely dead" with default "delete" rather than hard "must delete." Happy to downgrade to "archive" if the user wants a `docs/archive/` directory.

Third-candidate pushback: the `docs/design/kf_architecture.md` recommendation (archive vs delete). I'm already recommending archive-not-delete specifically to avoid this pushback.
