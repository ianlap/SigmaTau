---
name: git-workflow
description: >
  Use when committing, pushing, merging, fast-forwarding main, weakening
  tests, or structuring commit messages for this repo. Trigger on any
  git commit/push/merge action.
---

# Git Workflow

Codifies the dev-then-main workflow, commit-message conventions, and
test-discipline gates used on this repo.

## Dev-then-main

- **Commits land on `dev`.** Never commit directly to `main`.
- **Push to `origin/dev` after every commit.** No long-lived unpushed work.
  If you have multiple logical commits in flight, push each as it lands; the
  remote `dev` is the integration surface for anyone else watching.
- **`main` advances via fast-forward at milestones** when tests pass. No
  merge commits into main — the dev branch IS main with extra commits.
- **Never force-push to `main`**, even "just this once". A force-push to
  main rewrites shared history.

Current branch: `dev`. Main: `main`. Both track to `origin`.

## Conventional Commits

Every commit message uses the `type(scope): subject` form. Subject ≤ 70
characters; body wraps at ~72.

### Types

- `feat` — new user-facing capability
- `fix` — bug fix
- `refactor` — internal restructuring with no behavior change
- `test` — test file changes only
- `docs` — documentation changes only (includes mandates, CLAUDE.md, GEMINI.md)
- `chore` — tooling, git, build, dependencies

### Scopes (observed in this repo's log)

Engineering: `q↔h`, `matlab/dev`, `matlab/engine`, `matlab/filter`, `crossval`,
`noise_id`, `als_fit`, `optimize`, `kf`, `scripts`, `ml/dataset`, `ml/loader`,
`ml`. Documentation/meta: `mandates`, `changelog`, `parking-lot`. Housekeeping:
`gitignore`, `claude`.

Add a new scope when working in a subsystem not covered by the list above;
don't invent scopes that overlap with existing ones.

### Commit body structure

```text
<type>(<scope>): <subject, imperative, lowercase-ish>

Lead with *why* — what motivated this commit. What constraint, decision,
or incident made it necessary. One paragraph.

Then *what* — the specific changes. Bullet list if >2 things.

Then *result* — what's true after this commit that wasn't before (tests
passing, build green, invariant restored).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

The Co-Authored-By footer is standard for AI-assisted commits.

## Test discipline

### Hard gates

- **Julia test suite** (`cd julia && julia --project=. -e 'using Pkg; Pkg.test()'`) — must pass before `main` advances. Fast, reliable.
- **Python test suite** (`cd ml && python -m pytest`) — must pass before
  `main` advances for any `ml/` change. Fast, reliable.

### Aspirational gate (MATLAB)

The MATLAB suite (`cd matlab && matlab -batch "addpath(genpath('.')); run('tests/run_all.m')"`) is
aspirational, not a hard gate. It's slow (~1 minute), occasionally produces
transient `malloc unaligned tcache` crashes unrelated to code, and several
tests are seed-sensitive. Run it for MATLAB-side changes; do not block
milestone pushes on MATLAB CI flakiness alone. MATLAB-side correctness is
partially covered by `test_crossval_julia.m` against a reference fixture.

### Three-part justification for any test weakening

If a test's tolerance is loosened, threshold adjusted, or check replaced
with a weaker check, the commit message must include:

1. **Why the prior check was wrong** — what specifically about it was
   unsound (statistical assumption violated, parameter-dependent, etc.)
2. **What the new check verifies** — the weaker property, in precise terms
3. **What it no longer verifies** — the stronger property we've explicitly
   given up on (with a pointer to where/when it will be re-checked if at all)

Example: Test 3 in `matlab/tests/test_filter.m` (commit `52d12db`) went from
`P_final < P0/1000` (anchored to initial guess) to `|ΔP/P| < 1e-3` over the
second half (parameter-independent convergence). (1) prior check punished a
good P0; (2) new check verifies sequence convergence; (3) we no longer
verify an absolute magnitude bound on the steady-state value — if the math
were wrong such that steady-state doubled, the test would still pass.

### XFAIL pattern for known-bad tests

For tests with diagnosed-but-unfixed bugs, wrap the assert in `try/catch`
(MATLAB script-style) or use the framework's `assumeFail` / `xfail`
equivalent. Print `XFAIL (expected)` on catch and `XPASS (unexpected)` on
unexpected pass. Include a pointer to `FIX_PARKING_LOT.md` in the test
comment. Precedent: `test_filter.m` Test 2 xfail (commit `52d12db`).

## Pre-push checklist

1. Relevant test suite passes locally. Never `--no-verify`.
2. `git status` clean except for intentional changes.
3. Commit message follows Conventional-Commits + body structure.
4. No mandate or code file changes outside the stated scope of the fix/feat.

## Push cadence

- Push every commit in the same session. Don't queue up 5 commits on `dev`
  locally before pushing.
- If a commit breaks the build, push the fix immediately — don't leave
  `origin/dev` red waiting for "one more polish commit".

## Commit granularity

- **One logical change per commit.** A commit that touches deviation-engine
  math AND fixes a ML-pipeline bug is two commits.
- **Fixes and features don't mix.** If a refactor surfaces a latent bug, the
  bug fix is its own commit (first) followed by the refactor.
- **Atomic editorial passes are one commit.** Writing 7 new skills in one
  editorial pass is one commit, not 7 — the atomicity is the editorial
  discipline, not the skill count.

## When in doubt — escalate, don't guess

The four escalation triggers (from prior fix-series workflows):

1. The change is hard to reverse (force-push, history rewrite, amending a
   pushed commit, deleting a branch with unmerged work).
2. The change's blast radius extends beyond the local workspace (shared
   branch, CI config, external service).
3. A finding surfaces that belongs outside the current scope — log to
   `FIX_PARKING_LOT.md`, don't code-fix.
4. A mandate or audit document contradicts what you're about to write —
   surface the contradiction, don't silently override. The mandate might be
   stale (see the GEMINI.md §2.3 MATLAB-engine-4-arg finding), in which
   case log to parking-lot and proceed with current code reality.
