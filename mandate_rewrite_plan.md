I have enough evidence for the plan. Let me produce the five-part rewrite plan
  now. No edits yet — awaiting acknowledgment.

  ---
  Rewrite Plan — CLAUDE.md and GEMINI.md Reconciliation

  Plan Part A — Classification of every current GEMINI.md mandate

  Every row below cites a concrete check. Status legend: T = true, F = false, P =
  partial/scope-confused, S = stale, A = aspirational, SC = scope-confused (one
  language yes, other no).

  #: M1.1
  Mandate (abridged): §1 "Shared Engine" — all 10 devs use engine.m / engine.jl
  Status: T
  Evidence: engine.jl:30, engine.m:1; all 10 wrappers in julia/src/deviations/*.jl

    and matlab/+sigmatau/+dev/
  Disposition: Keep, scope to stability half
  ────────────────────────────────────────
  #: M1.2
  Mandate (abridged): "Kernel Interface" = @(x, m, tau0)
  Status: SC
  Evidence: Julia is 4-arg (x, m, tau0, x_cs) per engine.jl:20. MATLAB is 3-arg
  per
    engine.m:11. AUDIT_02 §5 calls this pending-work divergence (CHANGELOG claim
  is
     half-false).
  Disposition: Split per-language. Julia mandate = 4-arg; MATLAB mandate = 3-arg
    with migration flagged as Goal
  ────────────────────────────────────────
  #: M1.3
  Mandate (abridged): "Variance return" — kernels return σ², engine sqrts
  Status: T
  Evidence: engine.jl:68,87 (var_val, n = kernel(...), sqrt); engine.m:72,87 same
  Disposition: Keep, stability scope
  ────────────────────────────────────────
  #: M1.4
  Mandate (abridged): "Data Types" — data_type kw; freq→phase lives in engine
  Status: T
  Evidence: engine.jl:40-44; engine.m:38-42
  Disposition: Keep, stability scope
  ────────────────────────────────────────
  #: M1.5
  Mandate (abridged): "O(N) Complexity" — mdev/mhdev use cumsum prefix sums
  Status: T both
  Evidence: Julia: engine precomputes x_cs (engine.jl:63). MATLAB: each kernel
  does
    own cumsum([0; x(:)]) (5 of 10 kernels per AUDIT_02 §5) — still O(N) but
    redundant. Mandate's literal claim ("use cumsum prefix sums for O(N)") holds
  in
     both.
  Disposition: Keep, stability scope. Note: engine-level sharing is Julia-only
    (Goal: finish MATLAB refactor)
  ────────────────────────────────────────
  #: M1.6
  Mandate (abridged): "Input/Output" — KF uses config struct in / results struct
    out; no positional args
  Status: SC
  Evidence: MATLAB: kalman_filter.m:1 takes (data, config) struct ✓. Julia:
    filter.jl:149 takes (data::Vector, model::ClockModel; kwargs) — type-dispatch,

    not struct. Per user decision §2: this is legitimate per-language divergence.
  Disposition: Split per-language. MATLAB keeps struct pattern (verified); Julia
    rewrites to ClockModel{2,3,Diurnal} dispatch (verified filter.jl:149,
    clock_model.jl:26,36,46)
  ────────────────────────────────────────
  #: M1.7
  Mandate (abridged): "Composition" — KF pipeline load → engine → noise_fit →
    filter → optimize
  Status: P / SC
  Evidence: Order is backwards (for Julia you optimize → filter). Julia side:
    scripts/julia/kf_pipeline.jl is broken (AUDIT_01 §3). MATLAB
    scripts/matlab/kf_pipeline.m follows a variant. Vague as a mandate.
  Disposition: ESCALATE — delete as too vague, or rewrite as "KF filters MUST
    consume a fit-derived noise-params struct/ClockModel, not inline magic
    numbers"?
  ────────────────────────────────────────
  #: M2.1
  Mandate (abridged): "htotdev m=1" — overlapping HDEV, not totdev reflection
  Status: T
  Evidence: docs/equations/total.md:58; julia/src/deviations/total.jl:184,217;
    matlab/+sigmatau/+dev/htotdev.m:12,31,36
  Disposition: Keep, stability scope
  ────────────────────────────────────────
  #: M2.2
  Mandate (abridged): "mhtotdev EDF" — approximate coefficients from mc fitting
  Status: A (description, not mandate)
  Evidence: Coefs are approximate; no published model. TODO.md:13 lists refinement

    as task.
  Disposition: ESCALATE — demote to "Known-state" note or delete (it's
  descriptive,
     not a binding rule)
  ────────────────────────────────────────
  #: M2.3
  Mandate (abridged): "Bias Correction" tables match SP1065 exactly
  Status: T
  Evidence: Confirmed against docs/papers/reference/sp1065.pdf (per user memory);
    tables in julia/src/stats.jl and MATLAB equivalents
  Disposition: Keep, stability scope
  ────────────────────────────────────────
  #: M2.4
  Mandate (abridged): "Noise ID" dual-path threshold = 30
  Status: F in code
  Evidence: julia/src/noise.jl:24: NEFF_RELIABLE = 50.
    matlab/+sigmatau/+noise/noise_id.m:28: same. TODO.md:11 tracks migration to
  30.
  Disposition: ESCALATE — does mandate stay at 30 with "code pending"? Or does
    mandate update to match current 50? User's hard rule says "no aspirational as
    current fact."
  ────────────────────────────────────────
  #: M2.5
  Mandate (abridged): "Totdev Denominator" = 2(N-2)(mτ₀)², not N-1
  Status: T
  Evidence: SP1065 Eq 25, verified 2026-04-14 (CLAUDE.md note is primary record)
  Disposition: Keep verbatim — this is institutional memory
  ────────────────────────────────────────
  #: M2.6
  Mandate (abridged): "PID Convention" — integral accumulates phase error sumx +=
    x(1)
  Status: T
  Evidence: matlab/+sigmatau/+kf/update_pid.m:5 literally matches;
    kalman_filter.m:40 declares pid_state = [sumx; last_steer]
  Disposition: Keep, scope to KF steering (per-language or shared?)
  ────────────────────────────────────────
  #: M2.7
  Mandate (abridged): "Q-Matrix" exact τ powers (τ, τ³/3, τ⁵/20)
  Status: T
  Evidence: clock_model.jl:76-79 (build_Q ClockModel2); optimize.jl:103-110
    (3-state with IRWFM); matlab/+sigmatau/+kf/build_Q.m
  Disposition: Keep, KF scope
  ────────────────────────────────────────
  #: M2.8
  Mandate (abridged): "Covariance" P = (I - K*H)*P; safe_sqrt on diagonals
  Status: T
  Evidence: Julia: optimize.jl:174 has P = (I - K *H)* P; MATLAB:
    kalman_filter.m:85,126 uses safe_sqrt
  Disposition: Keep, KF scope
  ────────────────────────────────────────
  #: M2.9
  Mandate (abridged): "Internal Consistency" MATLAB vs Julia < 10⁻¹² rel err
  Status: F
  Evidence: Real tolerance: 2e-10 at matlab/tests/test_crossval_julia.m:47. Not
    enforced anywhere else.
  Disposition: DELETE (per user hard rule) and replace with accurate scoped
    tolerance
  ────────────────────────────────────────
  #: M2.10
  Mandate (abridged): "Cross-Validation" < 10⁻¹⁰ across tau, dev, edf, ci
  Status: F
  Evidence: Test checks deviation only (test_crossval_julia.m:34-45,62-80); not
    edf, not ci; not KF. Silently skips if file missing (:10-14).
  Disposition: DELETE and replace with: "Stability deviations cross-validate at
    REL_TOL = 2e-10 (matlab/tests/test_crossval_julia.m:47). Covers point
  estimates
     only; not EDF, CI, or KF."
  ────────────────────────────────────────
  #: M3.1
  Mandate (abridged): "Function Length" < 100 lines
  Status: F with documented violations
  Evidence: wc -l: engine.m=163, kalman_filter.m=208, optimize.m=215. TODO.md High

    Priority tracks refactor.
  Disposition: ESCALATE — user offered two options: "except [list]" or move to
    Goals. Recommend: keep mandate as "≤100 lines except the three documented
    legacy-size MATLAB files (engine.m:163, kalman_filter.m:208, optimize.m:215) —

    tracked in TODO.md for split."
  ────────────────────────────────────────
  #: M3.2
  Mandate (abridged): "Documentation" — cite equations in non-trivial blocks
  Status: P
  Evidence: Many functions cite (totdev.m, noise.jl, etc.); some don't
    (optimize.jl:180-183 is 1-line docstring per AUDIT_01 §8).
  Disposition: Keep as style standard, mark "spot-check verified; not exhaustively

    enforced."
  ────────────────────────────────────────
  #: M3.3
  Mandate (abridged): "Magic Numbers" — named constants
  Status: P
  Evidence: CONFIDENCE_DEFAULT in engine.jl:5,_P0_SCALE, _NM_* in optimize.jl,
    etc. Some magic numbers likely remain (unchecked).
  Disposition: Keep as style standard, same hedge
  ────────────────────────────────────────
  #: M3.4
  Mandate (abridged): "Dependencies" — Plots.jl MUST be package extension
  Status: F
  Evidence: julia/Project.toml:10 lists Plots in [deps]; no [weakdeps], no
    [extensions] section, julia/ext/ is empty (AUDIT_01 §4 row 6)
  Disposition: DEMOTE TO GOAL (per user hard rule). TODO.md:21 already tracks.
  ────────────────────────────────────────
  #: M3.5
  Mandate (abridged): "Namespace" — MATLAB in +sigmatau
  Status: T
  Evidence: ls matlab/+sigmatau/ shows +dev, +kf, +noise, +stats, +util, +plot
    (empty), +steering (empty)
  Disposition: Keep, MATLAB scope

  Summary

- Keep as-is / tighten wording: 10 mandates (M1.1, M1.3, M1.4, M1.5, M2.1, M2.3,
   M2.5, M2.6, M2.7, M2.8, M3.5)
- Split per-language / per-subsystem: 2 mandates (M1.2, M1.6)
- Delete and replace with accurate replacement: 2 mandates (M2.9, M2.10)
- Demote to Goal: 1 mandate (M3.4)
- Keep with documented exceptions: 1 mandate (M3.1) — pending escalation choice
- Escalation required: 3 mandates (M1.7, M2.2, M2.4) — plus confirming M3.1
  option

  ---
  Plan Part B — New GEMINI.md outline

  Section-by-section; every mandate in the new file carries a verification tag
  (format in Part D).

  1. Header & scope note — one short paragraph: what this file binds, how scope
  tags work, how the verification tag works, what overrides what (per superpowers
  priority rules).
  2. §1 Project Goals (new section, per user decision §5)
    - 2.1 ML pipeline purpose (PH 551 final: predict KF params from σ(τ))
    - 2.2 Convention: Wu 2023 q↔h parameterization is canonical
  (julia/src/clock_model.jl:138,151). Legacy formulas at test_filter.jl:186 and
  scripts/python/plot_kf.py:54-59 are debt, not alternative conventions — tracked
  in FIX_PARKING_LOT.md.
    - 2.3 Clock model is 3-state (WPM=R, WFM=σ₁², RWFM=σ₂²). Drift RW (σ₃²) is
  unobservable on available datasets — do not propose 4-parameter KF without
  explicit user sign-off.
    - 2.4 Packaging: one package, not two, pending post-PH-551 revisit. AUDIT_02
  documented the zero-coupling structure but decision is ⓒ-status-quo for now.
    - 2.5 KF paradigm asymmetry: Julia has migrated to ClockModel/ClockNoiseParams
   dispatch; MATLAB +kf/ is frozen at struct-config paradigm. Parity across
  languages is not a goal.
  3. §2 Deviation Engine mandates (scope: stability half — Julia engine.jl, MATLAB
   engine.m, and the 10 deviation wrappers)
    - M1.1 Shared engine
    - M1.2-Julia 4-arg kernel contract (x, m, tau0, x_cs)
    - M1.2-MATLAB 3-arg kernel contract (x, m, tau0) + Goal: migrate to 4-arg
    - M1.3 Variance return
    - M1.4 Data types (phase/freq, conversion in engine)
    - M1.5 cumsum prefix sums for O(N) — mdev/mhdev
    - M2.1 htotdev m=1 overlapping HDEV exception
    - M2.3 Bias correction tables (totvar, mtot, htot) exactly match SP1065
    - M2.5 Totdev denominator 2(N-2)(mτ₀)² — with the CLAUDE.md-verified
  2026-04-14 note preserved and cross-referenced
  4. §3 Noise Identification mandates (scope: stability half — julia/src/noise.jl,
   matlab/+sigmatau/+noise/noise_id.m)
    - M2.4 Dual-path (lag-1 ACF for long, B₁+R(n) for short) — threshold value
  pending escalation (see Part A row M2.4)
  5. §4 Kalman Filter mandates (scope per-language)
    - §4a Julia KF (type-dispatch pattern)
        - ClockModel dispatch is the primary API (filter.jl:149;
  clock_model.jl:26,36,46)
      - ClockNoiseParams is the noise-container type (clock_model.jl:13)
      - M2.7 Q-matrix exact τ powers
      - M2.8 Standard covariance update
      - M2.6 PID convention
    - §4b MATLAB KF (struct-config pattern, pre-refactor frozen)
        - Struct in / struct out (kalman_filter.m:1, optimize.m:1)
      - M2.7 / M2.8 / M2.6 same
      - Explicit note: MATLAB KF is not migrated to match Julia; no parity test
  exists
  6. §5 Cross-validation & accuracy (scope: stability-engine only)
    - Replaced M2.9/M2.10: Stability deviations cross-validate at REL_TOL = 2e-10
  (matlab/tests/test_crossval_julia.m:47). Covers the 10 deviation point estimates
   only — not EDF, CI, or KF. Test skips silently when crossval_results.txt
  missing (known gap, AUDIT_01 parking lot).
    - No KF cross-validation exists across languages.
  7. §6 Code Style & Standards (scope: repo-wide unless noted)
    - M3.1 ≤100 lines except three documented MATLAB legacy-size files
  (engine.m=163, kalman_filter.m=208, optimize.m=215) — tracked in TODO.md High
  Priority (pending Part A M3.1 escalation)
    - M3.2 Equation citations in non-trivial blocks (style; spot-check verified)
    - M3.3 Named constants over magic numbers (style; spot-check verified)
    - M3.5 MATLAB namespace +sigmatau
  8. §7 Goals (not mandates) — moved from current file where verification failed
    - G1 Plots.jl → package extension: move from [deps] to [weakdeps], populate
  julia/ext/, bump compat julia = "1.9"+
    - G2 MATLAB engine kernels → 4-arg contract (x, m, tau0, x_cs) (finish
  CHANGELOG claim)
    - G3 KF cross-validation (MATLAB vs Julia)
    - G4 Oversized MATLAB functions split (engine.m, kalman_filter.m, optimize.m)
    - G5 Noise-ID threshold alignment with mandate (contingent on M2.4 escalation
  choice)
  9. §8 References — NIST SP1065 (path), Greenhall & Riley PTTI 2003, IEEE
  1139-2022, Matsakis & Banerjee (path per user memory), Wu 2023 (docs/papers/stat
  e_estimation/2023_wu_kf_performance_lti_atomic_clock_ieee_taes.pdf)

  ---
  Plan Part C — New CLAUDE.md outline

  CLAUDE.md stays developer-workflow focused per user decision §5.

  1. Build & Test — preserve current cd matlab && ... and cd julia && ... commands
   (verified they work)
  2. Verification — rewrite the "< 1e-12 relative error" claim
    - Test slopes (White PM: τ⁻¹; White FM: τ⁻½; RWFM: τ^(+½)) — keep but cite
  which tests cover this (matlab/tests/test_noise_slopes.m,
  julia/test/test_allan_family.jl)
    - Cross-validation: MATLAB ↔ Julia stability deviations at REL_TOL = 2e-10
  (matlab/tests/test_crossval_julia.m:47). Covers point estimates only; silently
  skips if Julia ref file missing.
  3. Resources — update references (add Matsakis/Banerjee PDF pointer per user
  memory); DELETE stale "Read from matlab/legacy/ and julia/legacy_stablab/"
  (directories don't exist)
  4. Institutional memory / known-good rules — new section heading (rename from
  "Known Bugs")
    - Preserve verbatim: totdev denominator note (with verification tag added)
    - Preserve: htotdev bias direction / EDF loop indexing / mhtotdev Neff formula
   — still flagged for Riley (2001) verification
    - Add: Int64 overflow in MDEV/MHDEV denominators — fixed commit ead11ea
  (ml/STATE.md:171-181); don't regress — promote 2→2.0 and m→Float64(m) in
  allan.jl, hadamard.jl, noise.jl, total.jl
  5. Maintenance Tasks — DELETE the three current items; all files (mhdev_noID.m,
  mhtotdev_par.m, compute_devs_from_file.m) are absent from the repo (verified via
   Glob). Replace with a pointer: "Current maintenance queue lives in TODO.md
  (engineering work) and FIX_PARKING_LOT.md (in-flight fix debt)."

  ---
  Plan Part D — Verification timestamp format

  Pick: Option 3 variant — trailing italicized sentence on a new line after each
  mandate.

  Format:

- **Mandate statement.** Narrative body.
    _[Verified 2026-04-16 — check: <specific named action and expected result>.]_

  Why this over the three examples in the prompt:

- Option 1 (inline one-line suffix): becomes unwieldy when the check name is >3
  words — many of ours need ~10 words
- Option 2 (separate verification table): bifurcates the document; future edits
  will desync mandate text from its check, which is exactly the drift we're trying
   to prevent
- Option 3 as given ([v:2026-04-16 · <check>]): too cryptic for institutional
  memory; future readers lose the audit trail

  Trailing italicized tag on its own line keeps the check beside the mandate (no
  sync burden) while staying visually distinct from the mandate body. Italics
  signal "metadata, not normative text."

  Three example mandates in the chosen format

- **Shared deviation engine.** All 10 deviation functions dispatch through
    `matlab/+sigmatau/+dev/engine.m` or `julia/src/engine.jl`. Do not duplicate
    engine boilerplate in wrappers.
    _[Verified 2026-04-16 — check: read engine.jl:30 (`function engine(...)`) and
    engine.m:1 (`function result = engine(...)`); confirmed all 10 wrappers in
    `julia/src/deviations/*.jl` and `matlab/+sigmatau/+dev/{adev,mdev,hdev,mhdev,t
  dev,ldev,totdev,mtotdev,htotdev,mhtotdev}.m` call their language's engine.]_

- **Totdev denominator.** Use `2(N-2)(mτ₀)²` per SP1065 Eq 25 for phase form
    (equivalently `2(M-1)` for frequency form, `M = N-1`). Do not change to `N-1`.
    _[Verified 2026-04-14 — check: SP1065 Eq 25 reviewed against current
    `julia/src/deviations/total.jl` and `matlab/+sigmatau/+dev/totdev.m`;
  denominator
    matches 2τ²(N-2) in phase form. Preserved from prior CLAUDE.md note.]_

- **Cross-validation tolerance (stability deviations only).** MATLAB ↔ Julia
    deviation point estimates must agree within `REL_TOL = 2e-10`. Covers the 10
    deviation values only — **not** EDF, CI, or KF outputs. Test silently skips
    when the Julia reference file is missing.
    _[Verified 2026-04-16 — check: `matlab/tests/test_crossval_julia.m:47`
    (`REL_TOL = 2e-10`), `:10-14` (skip-on-missing), `:34-45,62-80` (deviation
    fields only; no edf/ci/KF comparison).]_

  ---
  Plan Part E — What is NOT changing (deliberate preservation)

  1. CLAUDE.md — totdev denominator paragraph — verbatim preservation, just adds
  the verification tag. User explicitly named this as archetypal.
  2. CLAUDE.md — Build & Test commands — verified both work (cd matlab && matlab
  -batch ... and cd julia && julia --project=. -e ...); no drift.
  3. CLAUDE.md — References list — NIST SP1065, Greenhall & Riley, IEEE 1139-2022
  preserved; Matsakis & Banerjee added from user memory; matlab/legacy/ +
  julia/legacy_stablab/ lines deleted (stale).
  4. GEMINI.md — these mandate bodies unchanged (only verification tag added):
  M1.1, M1.3, M1.4, M1.5, M2.1, M2.3, M2.5, M2.6, M2.7, M2.8, M3.5.

  ---
  Escalations — awaiting your call

  Four points need your sign-off before rewrite:

  1. M2.4 (NEFF_RELIABLE): Mandate says 30; code has 50 in both noise.jl:24 and
  noise_id.m:28; TODO.md:11 tracks updating code to 30. Options:
   (a) as stated — proceed: mandate=30, Goal G5="move code from 50 → 30 per
  SP1065 §5.5.6"
  2. M1.7 (KF Composition): Vague-as-written, half-broken on Julia side. Options:

  (b) Rewrite as: "KF filters MUST consume a fit-derived noise-params struct /
  ClockModel, not inline magic numbers" (my suggested rewrite)

  1. M2.2 (mhtotdev EDF approximate coefs): This is a description of current
  state, not a binding rule. Options:
  (a) Delete from mandates (TODO.md already tracks refinement)

  2. M3.1 (Function Length): Confirm approach:
  (a) "≤100 lines except three documented legacy-size MATLAB files (engine.m=163,
  kalman_filter.m=208, optimize.m=215); tracked in TODO.md" (my recommendation —
  keeps the standard visible, acknowledges reality)
