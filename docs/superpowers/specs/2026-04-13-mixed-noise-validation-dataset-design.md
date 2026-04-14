# Mixed-Noise Validation Dataset — Design

**Date:** 2026-04-13
**Purpose:** Create a reproducible 2^15-point mixed-noise phase dataset for cross-validating SigmaTau against Stable32 and allantools, with particular emphasis on exercising noise-identification dispatch and bias-correction branches across multiple τ regimes.

---

## Motivation

Existing validation assets in the repo (`examples/noise_id_validation/`) contain five single-α datasets at 2^16 points each. These are useful for verifying slope behavior of individual noise types, but they do not exercise:

- **Noise-ID dispatch across τ** — the algorithm must return different α values at different averaging factors when the dominant process changes.
- **Bias-correction branches** — `totvar`, `mtot`, and `htot` each use different bias-correction formulas per noise type; a mixed dataset hits several branches in one file.

A single mixed dataset with well-separated crossovers lets one validation file test all of these paths, with Stable32 and allantools as independent oracles.

## Scope

- **In scope:** Generate a single 2^15-point phase-data file, a Julia generator script, a README, and SigmaTau-computed σ(τ) reference tables for all 10 deviations plus noise ID.
- **Out of scope:** Running Stable32 or allantools; comparison diffing; MATLAB-side regenerator (Julia is the single source of truth for this dataset).

## Deliverables

Folder: `examples/mixed_noise_validation/`

```
mixed_noise.txt          — 2^15 single-column phase data, %.17g
generate.jl              — reproducible Julia generator
README.md                — composition, seeds, loading instructions
reference/
  adev.csv mdev.csv hdev.csv mhdev.csv
  totdev.csv mtotdev.csv htotdev.csv mhtotdev.csv
  tdev.csv ldev.csv
    cols: m, tau, sigma, edf, ci_lo, ci_hi, alpha_id
  noise_id.csv
    cols: m, tau, alpha_id
```

## Dataset Specification

| Parameter       | Value                                 |
|-----------------|---------------------------------------|
| N               | 32768 (2^15)                          |
| tau0            | 1.0                                   |
| Data type       | Phase                                 |
| Components      | WPM (α=+2), WFM (α=0), RWFM (α=−2)    |
| RNG seeds       | 1001 (WPM), 1002 (WFM), 1003 (RWFM)   |
| Target crossover 1 | τ ≈ 32  (WPM → WFM)                |
| Target crossover 2 | τ ≈ 512 (WFM → RWFM)               |
| File format     | Plain single-column text, `%.17g`     |
| Line count      | 32768                                 |

### Expected behavior

- **ADEV slope:** −1 at short τ, −1/2 at mid τ, +1/2 at long τ.
- **Noise ID:** returns +2 in short-τ band, ~0 in mid-τ band, ~−2 in long-τ band.

## Composition and Amplitude Calibration

Three independent realizations are summed:

```
x_mixed = c_WPM · x_WPM + c_WFM · x_WFM + c_RWFM · x_RWFM
```

Because the three realizations are independent, the component ADEVs combine in quadrature:

```
σ²_ADEV,mix(τ) = c_WPM² · A_WPM²(τ) + c_WFM² · A_WFM²(τ) + c_RWFM² · A_RWFM²(τ)
```

Each pure-noise component after `SigmaTau.noise_generate` has the canonical ADEV slope:

| Component | A(τ) slope  |
|-----------|-------------|
| WPM       | τ^(−1)      |
| WFM       | τ^(−1/2)    |
| RWFM      | τ^(+1/2)    |

**Crossover equations** (solve for the coefficient ratios that make two components equal at the target τ):

```
c_WPM  · A_WPM(τ=32)  = c_WFM · A_WFM(τ=32)           → fixes c_WPM / c_WFM
c_WFM  · A_WFM(τ=512) = c_RWFM · A_RWFM(τ=512)        → fixes c_RWFM / c_WFM
```

Fix `c_WFM = 1`, solve the other two. `A_i(τ=1)` is measured empirically from each generated realization (since Kasdin normalization for different α produces different absolute scales). The closed-form slopes let us extrapolate to the target τ without needing a fit.

## Algorithm (generate.jl)

1. `using SigmaTau, Random, Statistics, DelimitedFiles, Printf`
2. For each α ∈ {+2, 0, −2}:
   - `Random.seed!(seed_α)`
   - `x_α = SigmaTau.noise_generate(α, 2^15, 1.0)` (or equivalent — verify actual exported name during implementation)
   - Measure `A_α(τ=1)` by calling `SigmaTau.adev(x_α, 1.0)` and taking the first entry.
3. Solve for `c_WPM` and `c_RWFM` using the crossover equations above (`c_WFM := 1`).
4. Form `x_mixed = c_WPM*x_WPM + x_WFM + c_RWFM*x_RWFM`.
5. Write `mixed_noise.txt` with one value per line, `@sprintf("%.17g\n", v)`.
6. For each deviation function in SigmaTau, compute σ(τ) over an octave m-list (`m = 1, 2, 4, 8, …` up to `⌊N/4⌋` — the convention used in existing examples), capture tau / sigma / edf / ci_lo / ci_hi / alpha_id, and write to `reference/<name>.csv`.
7. Call `SigmaTau.noise_id` once and write `reference/noise_id.csv`.
8. Print a summary: achieved crossover τ values (for cross-check against the target).

## README Content

- Description: mixed-noise validation dataset for SigmaTau / Stable32 / allantools cross-checks.
- Composition, seeds, target crossovers.
- Size: 2^15 samples, tau0 = 1.0, phase data.
- Loading:
  - **Stable32**: Open file → select "Phase" → τ0 = 1.
  - **allantools**: `x = np.loadtxt('mixed_noise.txt'); at.oadev(x, rate=1.0, data_type='phase')`.
- Expected σ(τ) slopes and noise-ID behavior (as above).
- Regenerate: `julia --project=. examples/mixed_noise_validation/generate.jl` (from `julia/` directory).

## Data Format Rationale

- **Plain single-column text** matches the existing `noise_id_validation/*.csv` convention and is the simplest format both Stable32 and allantools accept without configuration.
- **`%.17g`** is the shortest printf format that round-trips IEEE 754 doubles exactly. This prevents precision loss from becoming a cross-tool discrepancy source.
- **Phase data** (not frequency) matches SigmaTau's default and both external tools' defaults, minimizing load-time conversion surface.

## Risks and Mitigations

- **Risk:** Generated crossovers do not land exactly at τ=32 / τ=512. **Mitigation:** The target is "≈" — small offsets are fine. The generator prints achieved values; if off by >2×, investigate.
- **Risk:** `SigmaTau.noise_generate` API name differs from what this spec assumes. **Mitigation:** Implementation step 1 is to look up the actual exported name and use it.
- **Risk:** Stable32 rejects `%.17g` scientific notation. **Mitigation:** Stable32 accepts standard scientific notation; the existing `noise_id_validation/` files use it. If there's an issue, fall back to `%.15e`.

## Success Criteria

1. `mixed_noise.txt` exists with exactly 32768 lines, each a parseable float.
2. `generate.jl` runs from `julia/` in under 60 s and produces byte-identical output on re-run (seeded RNG).
3. SigmaTau's noise-ID on the dataset returns α consistent with +2 → 0 → −2 across τ.
4. `reference/*.csv` exist for all 10 deviations plus noise_id.
5. README documents the loading commands for both Stable32 and allantools.
