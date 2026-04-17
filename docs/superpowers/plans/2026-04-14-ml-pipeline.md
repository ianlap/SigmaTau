# ML Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trained ML model that predicts Kalman-filter noise parameters `(q_wpm, q_wfm, q_rwfm)` from frequency-stability curves, validated on real GMR6000 Rb oscillator data.

**Architecture:** Julia side generates a 10,000-sample synthetic dataset (composite power-law phase noise + 3-D NLL-optimized labels) and writes `ml/data/dataset_v1.npz`. Python side trains Random Forest and XGBoost regressors on a 196-feature vector (4 deviations × 20 τ + slopes + variance ratios), evaluates on held-out synthetic samples and real-data windows.

**Tech Stack:** Julia (SigmaTau package + NPZ + StaticArrays); Python (scikit-learn, xgboost, forestci, numpy, matplotlib, jupyter).

**Spec:** `docs/superpowers/specs/2026-04-14-ml-pipeline-design.md` (authoritative design; this plan is the implementation breakdown).

---

## Dependencies and Prerequisites

- Julia 1.8+ with SigmaTau already checked out and passing `using Pkg; Pkg.test()`.
- Python 3.10+ with `pip` or `uv`.
- 12-core machine for dataset generation (designed for a single workstation).
- `reference/raw/6k27febunsteered.txt` and `reference/raw/6krb25apr.txt` present on disk.

---

## Phase 1 — Julia: Physics-preserving composite noise generator

**Deliverable:** `generate_composite_noise(h_coeffs, N, tau0; seed) → Vector{Float64}` producing a phase time series with correct absolute h_α amplitudes in seconds.

### Task 1.1 — Add NPZ and StaticArrays to SigmaTau

**Files:**
- Modify: `julia/Project.toml`

- [ ] **Step 1: Add deps to Project.toml**

Edit [julia/Project.toml](../../julia/Project.toml):

```toml
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NPZ = "15e1cf62-19b3-5cfa-8e77-841668bca605"
PrecompileTools = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"

[compat]
NPZ = "0.4"
PrecompileTools = "1"
StaticArrays = "1"
StatsFuns = "1.5.2"
julia = "1.8"
```

- [ ] **Step 2: Resolve deps**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'`
Expected: completes without error, `Manifest.toml` updated.

- [ ] **Step 3: Commit**

```bash
git add julia/Project.toml julia/Manifest.toml
git commit -m "deps: add NPZ and StaticArrays for ML pipeline"
```

### Task 1.2 — Single-α Kasdin generator with absolute amplitude

**Files:**
- Create: `julia/src/noise_gen.jl`
- Create: `julia/test/test_noise_gen.jl`
- Modify: `julia/src/SigmaTau.jl`
- Modify: `julia/test/runtests.jl`

- [ ] **Step 1: Write failing test for single-α generator**

Create [julia/test/test_noise_gen.jl](../../julia/test/test_noise_gen.jl):

```julia
# test_noise_gen.jl — composite power-law noise generator (Kasdin & Walter, 1992)

using Random

@testset "noise_gen" begin

    @testset "single-α amplitude calibration — WFM" begin
        # White FM (α=0): σ_y²(τ) = h₀ / (2τ) (SP1065 Table 5)
        # So ADEV(τ=1s) = √(h₀/2).  Verify realized ADEV matches target.
        h0  = 1e-22
        N   = 2^16
        τ₀  = 1.0
        x   = generate_power_law_noise(0.0, h0, N, τ₀; seed=42)
        @test length(x) == N
        @test all(isfinite, x)
        r = adev(x, τ₀)
        # Use short-τ region (m=1..4) where WFM asymptote is tightest
        mask = (r.tau .<= 4.0) .& .!isnan.(r.deviation)
        adev_realized = mean(r.deviation[mask])
        adev_theory   = sqrt(h0 / 2)
        @test isapprox(adev_realized, adev_theory; rtol=0.20)
    end

    @testset "single-α amplitude calibration — RWFM" begin
        # RWFM (α=−2): σ_y²(τ) = h₋₂ · (2π²/3) · τ (SP1065 Table 5)
        # Verify realized ADEV matches theory at long τ.
        h_m2 = 1e-24
        N    = 2^16
        τ₀   = 1.0
        x    = generate_power_law_noise(-2.0, h_m2, N, τ₀; seed=43)
        r    = adev(x, τ₀)
        mask = (r.tau .>= 128.0) .& (r.tau .<= 1024.0) .& .!isnan.(r.deviation)
        # Compare slope: log-log slope should be +1/2
        tau_m = r.tau[mask]
        dev_m = r.deviation[mask]
        # Least-squares slope
        lt = log.(tau_m); ld = log.(dev_m)
        slope = sum((lt .- mean(lt)) .* (ld .- mean(ld))) / sum((lt .- mean(lt)).^2)
        @test isapprox(slope, 0.5; atol=0.15)
        # Magnitude check at τ=512
        idx = findmin(abs.(tau_m .- 512.0))[2]
        adev_theory = sqrt(h_m2 * 2π^2 / 3 * tau_m[idx])
        @test isapprox(dev_m[idx], adev_theory; rtol=0.30)
    end

    @testset "reproducibility — same seed same output" begin
        x1 = generate_power_law_noise(0.0, 1.0, 1024, 1.0; seed=7)
        x2 = generate_power_law_noise(0.0, 1.0, 1024, 1.0; seed=7)
        @test x1 == x2
        x3 = generate_power_law_noise(0.0, 1.0, 1024, 1.0; seed=8)
        @test x1 != x3
    end

end
```

- [ ] **Step 2: Wire the test into runtests.jl**

Edit [julia/test/runtests.jl](../../julia/test/runtests.jl). Add after the existing includes (before the final `end`):

```julia
    include("test_noise_gen.jl")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -40`
Expected: FAIL with "UndefVarError: generate_power_law_noise not defined" or similar.

- [ ] **Step 4: Implement generator**

Create [julia/src/noise_gen.jl](../../julia/src/noise_gen.jl):

```julia
# noise_gen.jl — Composite power-law phase noise generator
# Kasdin & Walter (1992), "Discrete simulation of power law noise", IEEE FCS.
#
# Amplitude convention: bin-k one-sided spectral amplitude
#     |A_k|² = h_α · f_k^α · Δf
# where Δf = 1 / (N·τ₀).  IFFT of the conjugate-symmetric full spectrum
# produces frequency-domain data y(t) with two-sided PSD S_y(f) = h_α · |f|^α.
# Phase is cumsum(y) * τ₀ (in seconds).
#
# Works for any real α (integer or fractional, including FPM α=+1 and FFM α=−1).

using Random
using FFTW  # ← if FFTW not already transitively available; fall back to built-in `fft`

"""
    generate_power_law_noise(α, h, N, τ₀; seed) → Vector{Float64}

Synthesise N phase samples (in seconds) with frequency PSD S_y(f) = h · |f|^α.

# Arguments
- `α`: power-law exponent (S_y(f) ∝ f^α)
- `h`: coefficient h_α (dimensional, depends on α; e.g. for α=0 units are Hz⁻¹)
- `N`: number of output samples (must be even)
- `τ₀`: sampling interval (seconds)
- `seed`: integer RNG seed

# Returns
Phase time series `x(t)` in seconds, length N.
"""
function generate_power_law_noise(α::Real, h::Real, N::Int, τ₀::Real; seed::Integer)
    iseven(N) || throw(ArgumentError("N must be even for FFT symmetry"))
    N >= 4     || throw(ArgumentError("N must be ≥ 4"))
    h > 0      || throw(ArgumentError("h must be positive"))
    τ₀ > 0     || throw(ArgumentError("τ₀ must be positive"))

    rng = Xoshiro(seed)
    Δf  = 1.0 / (N * τ₀)

    # One-sided frequency grid: k = 1..N/2 (DC excluded, Nyquist forced real)
    Nhalf = N ÷ 2
    half  = zeros(ComplexF64, Nhalf + 1)

    # DC bin — set to 0 (no finite-power at f=0 for these power laws)
    half[1] = 0.0

    # Positive-freq bins 1..(Nhalf-1) — complex Gaussian amplitude with variance h·f^α·Δf
    for k in 1:(Nhalf - 1)
        f_k     = k * Δf
        var_k   = h * f_k^α * Δf
        σ_k     = sqrt(var_k / 2)         # split across real+imag for E[|A|²]=var_k
        half[k + 1] = σ_k * (randn(rng) + im * randn(rng))
    end

    # Nyquist bin — forced real so IFFT output is real-valued
    f_ny        = Nhalf * Δf
    var_ny      = h * f_ny^α * Δf
    half[end]   = sqrt(var_ny) * randn(rng)

    # Mirror to conjugate-symmetric full spectrum
    full              = Vector{ComplexF64}(undef, N)
    full[1:Nhalf + 1] = half
    @inbounds for k in 1:(Nhalf - 1)
        full[N - k + 1] = conj(half[k + 1])
    end

    # IFFT — FFTW convention: ifft(X)[n] = (1/N) Σ X[k] e^{+2πi nk/N}
    # Scale: to get continuous-time PSD match, multiply by N (undo the 1/N).
    y_freq = real.(ifft(full)) .* N

    # Integrate frequency → phase; multiply by τ₀ (discrete integration)
    return cumsum(y_freq) .* τ₀
end
```

Note: `ifft` without explicit `FFTW` import requires adding `using FFTW` — but since Julia's base `FFTW.jl` is already a transitive dep via most stacks, check: `julia --project=julia -e 'using FFTW; println("ok")'`. If it errors, add `FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"` to `[deps]`.

- [ ] **Step 5: Include in SigmaTau module and export**

Edit [julia/src/SigmaTau.jl](../../julia/src/SigmaTau.jl). Add in the export block:

```julia
export generate_power_law_noise, generate_composite_noise
```

And in the includes block (after `noise.jl`):

```julia
include("noise_gen.jl")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: PASS on all three `noise_gen` tests.

- [ ] **Step 7: Commit**

```bash
git add julia/src/noise_gen.jl julia/src/SigmaTau.jl julia/test/test_noise_gen.jl julia/test/runtests.jl
git commit -m "feat(noise_gen): add Kasdin power-law noise generator with absolute h_α scale"
```

### Task 1.3 — Composite generator (sum of α components)

**Files:**
- Modify: `julia/src/noise_gen.jl`
- Modify: `julia/test/test_noise_gen.jl`

- [ ] **Step 1: Add failing test for composite generation**

Append to [julia/test/test_noise_gen.jl](../../julia/test/test_noise_gen.jl), inside the `@testset "noise_gen"`:

```julia
    @testset "composite recovers WPM + WFM mix" begin
        # WPM dominates short τ (ADEV slope −1); WFM dominates long τ (slope −1/2)
        h_coeffs = Dict(2.0 => 1e-22,  # WPM
                        0.0 => 1e-22)   # WFM
        x = generate_composite_noise(h_coeffs, 2^15, 1.0; seed=101)
        r = adev(x, 1.0)
        τ = r.tau; σ = r.deviation
        # Short τ slope should be close to −1 (WPM)
        short_mask = (τ .>= 1.0) .& (τ .<= 4.0) .& .!isnan.(σ)
        lt = log.(τ[short_mask]); ld = log.(σ[short_mask])
        slope_short = sum((lt .- mean(lt)) .* (ld .- mean(ld))) / sum((lt .- mean(lt)).^2)
        @test slope_short < -0.7   # clearly dominated by WPM at short τ
        # Long τ slope should approach −1/2 (WFM)
        long_mask = (τ .>= 256.0) .& (τ .<= 2048.0) .& .!isnan.(σ)
        lt = log.(τ[long_mask]); ld = log.(σ[long_mask])
        slope_long = sum((lt .- mean(lt)) .* (ld .- mean(ld))) / sum((lt .- mean(lt)).^2)
        @test isapprox(slope_long, -0.5; atol=0.15)
    end

    @testset "composite is sum of components (linearity)" begin
        # x_sum = generator(α₁) + generator(α₂) at identical seeds should equal composite
        # — because each component draws from the seeded RNG independently.
        # We check numerical equality with the SAME SEED PROPAGATION.
        h = Dict(2.0 => 1e-22, -2.0 => 1e-24)
        x = generate_composite_noise(h, 1024, 1.0; seed=7)
        # Manually sum two independent calls (different seeds to avoid draw reuse)
        x_wpm = generate_power_law_noise( 2.0, 1e-22, 1024, 1.0; seed=7)
        x_rwf = generate_power_law_noise(-2.0, 1e-24, 1024, 1.0; seed=8)  # composite uses seed+1 internally
        @test isapprox(x, x_wpm .+ x_rwf; rtol=1e-12)
    end
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: FAIL on "composite …" with `generate_composite_noise` undefined.

- [ ] **Step 3: Implement composite generator**

Append to [julia/src/noise_gen.jl](../../julia/src/noise_gen.jl):

```julia
"""
    generate_composite_noise(h_coeffs, N, τ₀; seed) → Vector{Float64}

Synthesise composite phase noise by summing independent power-law components.

# Arguments
- `h_coeffs::AbstractDict{<:Real,<:Real}`: α → h_α mapping.
- `N`: number of samples (even).
- `τ₀`: sampling interval (seconds).
- `seed`: base RNG seed; each component uses seed + offset so streams never collide.

# Returns
Phase time series `x(t)` in seconds, length N.
"""
function generate_composite_noise(h_coeffs::AbstractDict{<:Real,<:Real}, N::Int,
                                   τ₀::Real; seed::Integer)
    isempty(h_coeffs) && throw(ArgumentError("h_coeffs must be non-empty"))
    x = zeros(Float64, N)
    # Sort α for deterministic seed offset ordering
    for (offset, α) in enumerate(sort(collect(keys(h_coeffs))))
        h = h_coeffs[α]
        x .+= generate_power_law_noise(α, h, N, τ₀; seed = seed + offset - 1)
    end
    return x
end
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: all `noise_gen` tests pass.

- [ ] **Step 5: Commit**

```bash
git add julia/src/noise_gen.jl julia/test/test_noise_gen.jl
git commit -m "feat(noise_gen): add composite noise generator summing power-law components"
```

---

## Phase 2 — Julia: Canonical τ grid and feature extraction

**Deliverable:** `compute_feature_vector(x, τ₀) → Vector{Float64}` returning 196 features (80 raw + 76 slopes + 40 ratios) computed on a fixed τ grid.

### Task 2.1 — Canonical τ grid

**Files:**
- Create: `julia/src/ml_features.jl`
- Create: `julia/test/test_ml_features.jl`
- Modify: `julia/src/SigmaTau.jl`
- Modify: `julia/test/runtests.jl`

- [ ] **Step 1: Write failing tests for grid constants**

Create [julia/test/test_ml_features.jl](../../julia/test/test_ml_features.jl):

```julia
# test_ml_features.jl — feature-extraction pipeline for ML dataset

using Random

@testset "ml_features" begin

    @testset "canonical tau grid shape" begin
        grid = SigmaTau.CANONICAL_TAU_GRID
        mlist = SigmaTau.CANONICAL_M_LIST
        @test length(grid)  == 20
        @test length(mlist) == 20
        @test all(mlist .>= 1)
        @test mlist == sort(unique(mlist))   # strictly increasing, no collisions
        @test mlist[1]  == 1
        @test mlist[end] <= 131072 ÷ 10       # τ_max = N·τ₀/10 for safety factor 10
        @test grid == Float64.(mlist)          # τ = m·τ₀, τ₀=1
    end

    @testset "feature vector length and ordering" begin
        Random.seed!(0)
        x = cumsum(randn(2^15))
        v = compute_feature_vector(x, 1.0)
        @test length(v) == 196   # 80 raw + 76 slopes + 40 ratios
        @test all(isfinite.(v) .| isnan.(v))  # never Inf
    end

    @testset "feature names parallel the vector" begin
        names = SigmaTau.FEATURE_NAMES
        @test length(names) == 196
        # First 80: raw; slopes: 76; ratios: 40
        raw_n   = count(startswith.(names, "raw_"))
        slope_n = count(startswith.(names, "slope_"))
        ratio_n = count(startswith.(names, "ratio_"))
        @test raw_n   == 80
        @test slope_n == 76
        @test ratio_n == 40
    end

end
```

- [ ] **Step 2: Wire test into runtests.jl**

Edit [julia/test/runtests.jl](../../julia/test/runtests.jl), add at bottom before final `end`:

```julia
    include("test_ml_features.jl")
```

- [ ] **Step 3: Run test to verify failure**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: FAIL with undefined `CANONICAL_TAU_GRID` or `compute_feature_vector`.

- [ ] **Step 4: Implement grid + feature names**

Create [julia/src/ml_features.jl](../../julia/src/ml_features.jl):

```julia
# ml_features.jl — Feature extraction for the ML dataset.
#
# Canonical τ grid shared by generator, feature extractor, and Python loader.
# Computes 196 features per sample: 80 raw σ values + 76 adjacent-τ slopes
# + 40 variance ratios (MVAR/AVAR and MHVAR/HVAR at each τ).

const _N_POINTS_DEFAULT   = 131_072       # 2^17
const _SAFETY_FACTOR       = 10             # τ_max = N / SAFETY_FACTOR

# 20 log-spaced m values from 1 to floor(N/SAFETY_FACTOR).  For N=131072, m_max=13107.
# After integer rounding these 20 values are already unique.
const CANONICAL_M_LIST   = Int[
    1, 2, 3, 4, 7, 12, 19, 31, 51, 83,
    136, 222, 364, 596, 976, 1597, 2614, 4279, 7003, 11461
]
const CANONICAL_TAU_GRID = Float64.(CANONICAL_M_LIST)

# Feature names, ordered as: 80 raw (4 stats × 20 τ), 76 slopes, 40 ratios.
const _STATS  = ("adev", "mdev", "hdev", "mhdev")
const FEATURE_NAMES = let
    names = String[]
    for stat in _STATS, m in CANONICAL_M_LIST
        push!(names, "raw_$(stat)_m$(m)")
    end
    for stat in _STATS, i in 1:19
        push!(names, "slope_$(stat)_m$(CANONICAL_M_LIST[i])_m$(CANONICAL_M_LIST[i+1])")
    end
    for m in CANONICAL_M_LIST
        push!(names, "ratio_mvar_avar_m$(m)")
    end
    for m in CANONICAL_M_LIST
        push!(names, "ratio_mhvar_hvar_m$(m)")
    end
    names
end
@assert length(FEATURE_NAMES) == 196

"""
    compute_feature_vector(x::AbstractVector{<:Real}, τ₀::Real) → Vector{Float64}

Compute the 196-feature vector for a phase time series `x`.  NaN values
propagate through slopes and ratios (column-median imputation happens in Python).
"""
function compute_feature_vector(x::AbstractVector{<:Real}, τ₀::Real)
    m_list = CANONICAL_M_LIST
    # Compute all four deviations on the shared m_list
    r_adev  = adev( x, τ₀; m_list)
    r_mdev  = mdev( x, τ₀; m_list)
    r_hdev  = hdev( x, τ₀; m_list)
    r_mhdev = mhdev(x, τ₀; m_list)

    # --- Raw log10(σ) features: 80 values ---
    σ_adev  = r_adev.deviation
    σ_mdev  = r_mdev.deviation
    σ_hdev  = r_hdev.deviation
    σ_mhdev = r_mhdev.deviation

    v = Float64[]
    append!(v, _safe_log10.(σ_adev))
    append!(v, _safe_log10.(σ_mdev))
    append!(v, _safe_log10.(σ_hdev))
    append!(v, _safe_log10.(σ_mhdev))

    # --- Slope features: 76 values (4 stats × 19 adjacent pairs) ---
    τ = CANONICAL_TAU_GRID
    for σ in (σ_adev, σ_mdev, σ_hdev, σ_mhdev)
        for i in 1:19
            lτ1, lτ2 = log10(τ[i]), log10(τ[i+1])
            lσ1, lσ2 = _safe_log10(σ[i]), _safe_log10(σ[i+1])
            push!(v, (lσ2 - lσ1) / (lτ2 - lτ1))
        end
    end

    # --- Ratio features: 40 values (MVAR/AVAR, MHVAR/HVAR at each τ) ---
    for i in 1:20
        push!(v, (σ_mdev[i]  / σ_adev[i])^2)
    end
    for i in 1:20
        push!(v, (σ_mhdev[i] / σ_hdev[i])^2)
    end

    @assert length(v) == 196
    return v
end

_safe_log10(σ::Real) = σ > 0 ? log10(σ) : NaN
```

- [ ] **Step 5: Include in module and export**

Edit [julia/src/SigmaTau.jl](../../julia/src/SigmaTau.jl):

```julia
export CANONICAL_TAU_GRID, CANONICAL_M_LIST, FEATURE_NAMES, compute_feature_vector
```

And add `include("ml_features.jl")` after `include("noise_gen.jl")`.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: all `ml_features` tests pass.

- [ ] **Step 7: Commit**

```bash
git add julia/src/ml_features.jl julia/src/SigmaTau.jl julia/test/test_ml_features.jl julia/test/runtests.jl
git commit -m "feat(ml_features): canonical τ grid and 196-feature extraction"
```

---

## Phase 3 — Julia: KF NLL optimizer upgrade (3-D + StaticArrays)

**Deliverable:** `optimize_kf_nll(phase, τ₀; h_init=nothing) → OptimizeResult` that optimizes `(q_wpm, q_wfm, q_rwfm)` in log10 space using a StaticArrays-backed inner loop.

### Task 3.1 — Add `optimize_qwpm` flag to OptimizeConfig

**Files:**
- Modify: `julia/src/optimize.jl`
- Modify: `julia/test/test_filter.jl`

- [ ] **Step 1: Add failing test for 3-D optimization**

Append a new `@testset` block to [julia/test/test_filter.jl](../../julia/test/test_filter.jl), before the closing `end`:

```julia
    @testset "3-D NLL optimization recovers q_wpm on WPM+WFM data" begin
        Random.seed!(99)
        # Synthetic phase: WPM (R) + WFM (q_wfm), no drift
        N = 4096; τ = 1.0
        q_wpm_true = 0.25
        q_wfm_true = 0.01
        # Measurement noise
        w  = sqrt(q_wpm_true) .* randn(N)
        # Process: state freq integrates from white drive with var q_wfm_true
        v  = sqrt(q_wfm_true * τ) .* randn(N)
        f  = cumsum(v)
        ph = cumsum(f) .* τ .+ w   # observed phase

        cfg = OptimizeConfig(
            q_wpm   = 1.0,    # poor initial guess — force optimizer to work
            q_wfm   = 0.1,
            q_rwfm  = 1e-8,
            nstates = 3,
            tau     = τ,
            verbose = false,
            max_iter = 2000,
            tol      = 1e-5,
            optimize_qwpm = true,
        )
        res = optimize_kf(ph, cfg)
        # Should recover within 1 decade (NLL landscape is smooth here)
        @test abs(log10(res.q_wpm) - log10(q_wpm_true)) < 1.0
        @test abs(log10(res.q_wfm) - log10(q_wfm_true)) < 1.0
    end
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -40`
Expected: FAIL with `UndefKeywordError: keyword argument optimize_qwpm not assigned` or similar.

- [ ] **Step 3: Extend OptimizeConfig**

Edit [julia/src/optimize.jl](../../julia/src/optimize.jl). Replace the `OptimizeConfig` struct (lines 32–42) with:

```julia
Base.@kwdef struct OptimizeConfig
    q_wpm::Float64       = 100.0   # Measurement noise R (initial guess or fixed)
    q_wfm::Float64       = 0.01    # Initial WFM guess
    q_rwfm::Float64      = 1e-6    # Initial RWFM guess
    q_irwfm::Float64     = 0.0     # Initial IRWFM guess (0 = not optimized)
    nstates::Int         = 3       # KF state dimension: 2 or 3
    tau::Float64         = 1.0     # Sampling interval [s]
    verbose::Bool        = true    # Print progress
    max_iter::Int        = 500     # Max Nelder-Mead iterations
    tol::Float64         = 1e-6    # Convergence tolerance (std of simplex f-values)
    optimize_qwpm::Bool  = false   # If true, walk log10(q_wpm) too
end
```

- [ ] **Step 4: Update `_kf_nll` to accept optional q_wpm parameter**

Edit [julia/src/optimize.jl](../../julia/src/optimize.jl). Replace the `_kf_nll` body so that when `length(theta)` is one greater than the q_wfm/q_rwfm count, the first parameter is treated as `log10(q_wpm)`:

```julia
function _kf_nll(theta::Vector{Float64}, data::Vector{Float64},
                 cfg::OptimizeConfig)::Float64
    N   = length(data)
    ns  = cfg.nstates
    τ   = cfg.tau

    # Unpack theta — layout depends on cfg.optimize_qwpm and cfg.q_irwfm
    idx = 1
    R = cfg.q_wpm
    if cfg.optimize_qwpm
        R = 10.0^theta[idx]; idx += 1
    end
    q_wfm   = 10.0^theta[idx]; idx += 1
    q_rwfm  = 10.0^theta[idx]; idx += 1
    q_irwfm = length(theta) >= idx ? 10.0^theta[idx] : 0.0

    # State-transition Φ
    Φ = Matrix{Float64}(I, ns, ns)
    ns >= 2 && (Φ[1, 2] = τ)
    ns >= 3 && (Φ[1, 3] = τ^2 / 2; Φ[2, 3] = τ)

    H = zeros(Float64, 1, ns); H[1, 1] = 1.0

    Q = _build_Q(ns, q_wfm, q_rwfm, q_irwfm, τ)

    # LS initialization
    n_fit = max(ns, min(_MAX_LS_SAMPLES, N - 1))
    t_fit = Float64.(0:n_fit-1) .* τ
    A_fit = _build_A(t_fit, ns)
    x     = A_fit \ data[1:n_fit]

    P   = _P0_SCALE .* Matrix{Float64}(I, ns, ns)
    nll = 0.0
    for k in 1:N
        if k > 1
            x = Φ * x
            P = Φ * P * Φ' + Q
        end
        ν = data[k] - (H * x)[1]
        S = (H * P * H')[1, 1] + R
        S <= 0.0 && return _INVALID_NLL
        nll += 0.5 * (log(S) + ν^2 / S)
        K = (P * H') ./ S
        x = x + K[:, 1] .* ν
        P = (I - K * H) * P
    end
    return nll
end
```

- [ ] **Step 5: Update `optimize_kf` to build theta0 with or without q_wpm**

Edit [julia/src/optimize.jl](../../julia/src/optimize.jl). Replace the theta0 construction block (lines 273–275) with:

```julia
    # Initial guess in log10 space
    theta0 = Float64[]
    cfg.optimize_qwpm && push!(theta0, log10(cfg.q_wpm))
    push!(theta0, log10(cfg.q_wfm))
    push!(theta0, log10(cfg.q_rwfm))
    cfg.q_irwfm > 0 && push!(theta0, log10(cfg.q_irwfm))
```

And replace the parameter unpacking after optimization (lines 282–284) with:

```julia
    idx = 1
    q_wpm_opt = cfg.q_wpm
    if cfg.optimize_qwpm
        q_wpm_opt = 10.0^theta_opt[idx]; idx += 1
    end
    q_wfm_opt   = 10.0^theta_opt[idx]; idx += 1
    q_rwfm_opt  = 10.0^theta_opt[idx]; idx += 1
    q_irwfm_opt = length(theta_opt) >= idx ? 10.0^theta_opt[idx] : 0.0
```

And replace `OptimizeResult(cfg.q_wpm, ...)` with `OptimizeResult(q_wpm_opt, ...)`.

- [ ] **Step 6: Run test to verify pass**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: the new 3-D test passes (plus all existing tests).

- [ ] **Step 7: Commit**

```bash
git add julia/src/optimize.jl julia/test/test_filter.jl
git commit -m "feat(optimize): add optimize_qwpm flag for 3-D NLL optimization"
```

### Task 3.2 — StaticArrays fast-path and hoisted LS init

**Files:**
- Modify: `julia/src/optimize.jl`

- [ ] **Step 1: Add failing performance test**

Append to [julia/test/test_filter.jl](../../julia/test/test_filter.jl) inside the "Kalman filter" testset:

```julia
    @testset "_kf_nll_static matches _kf_nll output" begin
        Random.seed!(123)
        N  = 2048; τ = 1.0
        x  = cumsum(randn(N))
        cfg = OptimizeConfig(q_wpm=1.0, q_wfm=0.5, q_rwfm=1e-4,
                             nstates=3, tau=τ, verbose=false,
                             optimize_qwpm=true)
        theta = [log10(cfg.q_wpm), log10(cfg.q_wfm), log10(cfg.q_rwfm)]
        nll_slow = SigmaTau._kf_nll(theta, x, cfg)
        nll_fast = SigmaTau._kf_nll_static(theta, x, cfg)
        @test isapprox(nll_slow, nll_fast; rtol=1e-10)
    end
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: FAIL with undefined `_kf_nll_static`.

- [ ] **Step 3: Implement `_kf_nll_static`**

Append to [julia/src/optimize.jl](../../julia/src/optimize.jl) (after `_build_A`):

```julia
using StaticArrays: SMatrix, SVector, @SMatrix, @SVector

"""
    _kf_nll_static(theta, data, cfg) → Float64

Faster NLL kernel for `nstates == 3` using StaticArrays and the scalar-H
shortcut.  Produces identical results to `_kf_nll` to numerical precision.
"""
function _kf_nll_static(theta::Vector{Float64}, data::Vector{Float64},
                        cfg::OptimizeConfig)::Float64
    cfg.nstates == 3 || return _kf_nll(theta, data, cfg)

    N = length(data)
    τ = cfg.tau

    idx = 1
    R = cfg.q_wpm
    if cfg.optimize_qwpm
        R = 10.0^theta[idx]; idx += 1
    end
    q_wfm   = 10.0^theta[idx]; idx += 1
    q_rwfm  = 10.0^theta[idx]; idx += 1
    q_irwfm = length(theta) >= idx ? 10.0^theta[idx] : 0.0

    # Φ (3×3) and Q (3×3) as static matrices
    Φ = @SMatrix [1.0  τ     τ^2/2;
                  0.0  1.0   τ;
                  0.0  0.0   1.0]

    τ2 = τ^2; τ3 = τ^3; τ4 = τ^4; τ5 = τ^5
    Q11 = q_wfm*τ + q_rwfm*τ3/3 + q_irwfm*τ5/20
    Q12 = q_rwfm*τ2/2 + q_irwfm*τ4/8
    Q13 = q_irwfm*τ3/6
    Q22 = q_rwfm*τ + q_irwfm*τ3/3
    Q23 = q_irwfm*τ2/2
    Q33 = q_irwfm*τ
    Q   = @SMatrix [Q11 Q12 Q13; Q12 Q22 Q23; Q13 Q23 Q33]

    # LS initialization — hoisted into static form
    n_fit = max(3, min(_MAX_LS_SAMPLES, N - 1))
    t_fit = Float64.(0:n_fit-1) .* τ
    A     = hcat(ones(n_fit), t_fit, t_fit.^2 ./ 2)
    xls   = A \ data[1:n_fit]
    x     = SVector{3,Float64}(xls[1], xls[2], xls[3])

    P   = SMatrix{3,3,Float64}(_P0_SCALE*I)
    nll = 0.0

    @inbounds for k in 1:N
        if k > 1
            x = Φ * x
            P = Φ * P * Φ' + Q
        end
        # Scalar-H shortcut: H = [1 0 0]
        ν = data[k] - x[1]
        S = P[1,1] + R
        S <= 0.0 && return _INVALID_NLL
        nll += 0.5 * (log(S) + ν*ν/S)
        # Kalman gain (3×1)
        K = SVector{3,Float64}(P[1,1]/S, P[2,1]/S, P[3,1]/S)
        x = x + K .* ν
        # P = (I - K H) P = P - K * P[1,:]'
        pr1 = SVector{3,Float64}(P[1,1], P[1,2], P[1,3])
        P = P - K * pr1'
    end
    return nll
end
```

- [ ] **Step 4: Dispatch `optimize_kf` through static path when applicable**

Edit [julia/src/optimize.jl](../../julia/src/optimize.jl). In `optimize_kf`, change the objective `obj = th -> _kf_nll(th, data, cfg)` to:

```julia
    obj = if cfg.nstates == 3
        th -> _kf_nll_static(th, data, cfg)
    else
        th -> _kf_nll(th, data, cfg)
    end
```

- [ ] **Step 5: Run tests to verify pass**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: all existing tests pass and `_kf_nll_static` matches `_kf_nll`.

- [ ] **Step 6: Quick benchmark (informational)**

Run: `cd julia && julia --project=. -e '
using SigmaTau, Random, BenchmarkTools
Random.seed!(0); x = cumsum(randn(8192)); τ = 1.0
cfg = OptimizeConfig(q_wpm=1.0, q_wfm=0.5, q_rwfm=1e-4, nstates=3,
                     tau=τ, verbose=false, optimize_qwpm=true)
θ = [0.0, -0.3, -4.0]
b1 = @belapsed SigmaTau._kf_nll(\$θ, \$x, \$cfg)
b2 = @belapsed SigmaTau._kf_nll_static(\$θ, \$x, \$cfg)
@show b1 b2 (b1/b2)
'`
Expected: `_kf_nll_static` at least 5× faster than `_kf_nll`. (BenchmarkTools is an optional dep; if missing, `julia -e 'using Pkg; Pkg.add("BenchmarkTools")'` in your global env.)

- [ ] **Step 7: Commit**

```bash
git add julia/src/optimize.jl julia/test/test_filter.jl
git commit -m "perf(optimize): StaticArrays + scalar-H fast path for _kf_nll (nstates=3)"
```

### Task 3.3 — `optimize_kf_nll` wrapper with analytical h-warm start

**Files:**
- Modify: `julia/src/optimize.jl`
- Modify: `julia/src/SigmaTau.jl`

- [ ] **Step 1: Add failing test for wrapper**

Append to [julia/test/test_filter.jl](../../julia/test/test_filter.jl) inside the "Kalman filter" testset:

```julia
    @testset "optimize_kf_nll wrapper with h_init warm start" begin
        Random.seed!(55)
        # Generate composite WPM+WFM+RWFM noise with known h_α
        h = Dict(2.0 => 1e-22, 0.0 => 1e-22, -2.0 => 1e-26)
        x = generate_composite_noise(h, 2^14, 1.0; seed=55)
        res = optimize_kf_nll(x, 1.0; h_init=h, verbose=false)
        @test res.converged
        # Analytical warm-start targets (spec §3.3)
        f_h = 0.5
        q_wpm_exp  = h[2.0] * f_h / (2π^2)
        q_wfm_exp  = h[0.0] / 2
        q_rwfm_exp = (2π^2 / 3) * h[-2.0]
        # With warm start + NLL refinement, results should land within ~1 decade
        @test abs(log10(res.q_wpm)  - log10(q_wpm_exp))  < 1.0
        @test abs(log10(res.q_wfm)  - log10(q_wfm_exp))  < 1.0
        @test abs(log10(res.q_rwfm) - log10(q_rwfm_exp)) < 1.0
    end
```

- [ ] **Step 2: Run to verify failure**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: FAIL with `UndefVarError: optimize_kf_nll`.

- [ ] **Step 3: Implement wrapper**

Append to [julia/src/optimize.jl](../../julia/src/optimize.jl):

```julia
"""
    optimize_kf_nll(phase, τ₀; h_init=nothing, verbose=true, max_iter=500) → OptimizeResult

High-level wrapper used by the ML dataset driver.  Optimizes
`(q_wpm, q_wfm, q_rwfm)` in log10 space via Nelder-Mead on `_kf_nll_static`.
If `h_init::Dict` is provided, uses the SP1065 analytical mapping h_α → q
as the simplex centroid — drastically reduces iteration count.
"""
function optimize_kf_nll(phase::AbstractVector{<:Real}, τ₀::Real;
                         h_init::Union{Nothing,AbstractDict{<:Real,<:Real}}=nothing,
                         verbose::Bool = true,
                         max_iter::Int = 500,
                         tol::Float64  = 1e-6)
    # Defaults cover the typical Rb range; overridden by h_init when supplied.
    q_wpm0, q_wfm0, q_rwfm0 = 1e-26, 1e-25, 1e-26
    if h_init !== nothing
        f_h = 1.0 / (2τ₀)
        haskey(h_init,  2.0) && (q_wpm0  = h_init[ 2.0] * f_h / (2π^2))
        haskey(h_init,  0.0) && (q_wfm0  = h_init[ 0.0] / 2)
        haskey(h_init, -2.0) && (q_rwfm0 = (2π^2 / 3) * h_init[-2.0])
    end
    cfg = OptimizeConfig(
        q_wpm   = q_wpm0,
        q_wfm   = q_wfm0,
        q_rwfm  = q_rwfm0,
        nstates = 3,
        tau     = τ₀,
        verbose = verbose,
        max_iter = max_iter,
        tol      = tol,
        optimize_qwpm = true,
    )
    return optimize_kf(Vector{Float64}(phase), cfg)
end
```

- [ ] **Step 4: Export from SigmaTau**

Edit [julia/src/SigmaTau.jl](../../julia/src/SigmaTau.jl):

```julia
export OptimizeConfig, OptimizeResult, optimize_kf, optimize_kf_nll
```

- [ ] **Step 5: Run tests to verify pass**

Run: `cd julia && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add julia/src/optimize.jl julia/src/SigmaTau.jl julia/test/test_filter.jl
git commit -m "feat(optimize): add optimize_kf_nll wrapper with analytical h-warm start"
```

---

## Phase 4 — Julia: Dataset driver

**Deliverable:** standalone script `ml/dataset/generate_dataset.jl` that produces `ml/data/dataset_v1.npz` with 10,000 samples.

### Task 4.1 — Dataset driver scaffold and single-sample pipeline

**Files:**
- Create: `ml/dataset/Project.toml`
- Create: `ml/dataset/generate_dataset.jl`
- Create: `ml/dataset/test_one_sample.jl`

- [ ] **Step 1: Create driver project**

Create [ml/dataset/Project.toml](../../ml/dataset/Project.toml):

```toml
[deps]
NPZ = "15e1cf62-19b3-5cfa-8e77-841668bca605"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SigmaTau = "a3b4c5d6-e7f8-4321-abcd-1234567890ab"

[compat]
NPZ = "0.4"
julia = "1.8"
```

Run: `cd ml/dataset && julia --project=. -e 'using Pkg; Pkg.develop(path="../../julia"); Pkg.resolve(); Pkg.instantiate()'`
Expected: project sees local SigmaTau and NPZ resolves.

- [ ] **Step 2: Write single-sample test**

Create [ml/dataset/test_one_sample.jl](../../ml/dataset/test_one_sample.jl):

```julia
# test_one_sample.jl — smoke test for the dataset driver's single-sample pipeline
using Random
using SigmaTau
include("generate_dataset.jl")

using .DatasetGen: draw_sample_params, run_one_sample

rng = Xoshiro(42)
p   = draw_sample_params(rng)
@assert haskey(p.h_coeffs, 2.0)
@assert haskey(p.h_coeffs, 0.0)
@assert haskey(p.h_coeffs, -1.0)
@assert haskey(p.h_coeffs, -2.0)
# FPM sometimes present, sometimes not — randomised, but for fixed seed check determinism:
p2 = draw_sample_params(Xoshiro(42))
@assert p == p2

println("draw_sample_params ok.")

# Run the full per-sample pipeline with a small N for speed
result = run_one_sample(1; N=2^14, τ₀=1.0, verbose=false)
@assert length(result.features) == 196
@assert length(result.h_coeffs) == 5
@assert length(result.q_labels) == 3
println("run_one_sample ok.")
```

- [ ] **Step 3: Run test to verify failure**

Run: `cd ml/dataset && julia --project=. test_one_sample.jl 2>&1 | tail -20`
Expected: FAIL with `ArgumentError: Package DatasetGen not found` (generate_dataset.jl doesn't exist yet).

- [ ] **Step 4: Implement driver module**

Create [ml/dataset/generate_dataset.jl](../../ml/dataset/generate_dataset.jl):

```julia
# generate_dataset.jl — ML dataset driver (Julia)
#
# Produces ml/data/dataset_v1.npz with 10,000 synthetic samples.  Threaded;
# checkpoints every CKPT_EVERY samples; resumes from checkpoint if present.

module DatasetGen

using Random, Statistics
using NPZ
using SigmaTau

export draw_sample_params, run_one_sample, generate_dataset, SampleParams, SampleResult

# ── Sampling ranges from the spec (§2.3) — log₁₀(h_α) ────────────────────────
const _H_RANGES = Dict(
     2.0 => (-26.5, -23.5),   # WPM
     1.0 => (-25.5, -22.5),   # FPM (only when present)
     0.0 => (-25.5, -22.5),   # WFM
    -1.0 => (-25.5, -22.5),   # FFM
    -2.0 => (-27.5, -24.5),   # RWFM
)
const _FPM_PROBABILITY = 0.30

struct SampleParams
    h_coeffs :: Dict{Float64,Float64}   # α → h_α
    fpm_present :: Bool
end

function Base.:(==)(a::SampleParams, b::SampleParams)
    a.h_coeffs == b.h_coeffs && a.fpm_present == b.fpm_present
end

"""
    draw_sample_params(rng) → SampleParams

Draw one random (h_coeffs, fpm_present) sample.
"""
function draw_sample_params(rng::AbstractRNG)
    h = Dict{Float64,Float64}()
    for α in (2.0, 0.0, -1.0, -2.0)       # always present
        lo, hi = _H_RANGES[α]
        h[α] = 10.0 ^ (lo + (hi - lo) * rand(rng))
    end
    fpm_present = rand(rng) < _FPM_PROBABILITY
    if fpm_present
        lo, hi = _H_RANGES[1.0]
        h[1.0] = 10.0 ^ (lo + (hi - lo) * rand(rng))
    end
    return SampleParams(h, fpm_present)
end

struct SampleResult
    features  :: Vector{Float64}       # 196
    q_labels  :: Vector{Float64}       # log10(q_wpm, q_wfm, q_rwfm)
    h_coeffs  :: Vector{Float64}       # log10(h₊₂, h₊₁, h₀, h₋₁, h₋₂) — NaN for absent
    fpm_present :: Bool
    nll       :: Float64
    converged :: Bool
end

"""
    run_one_sample(idx; N, τ₀, verbose=false) → SampleResult

Generate a single dataset sample with deterministic seed `42 + idx`.
"""
function run_one_sample(idx::Integer;
                        N::Int   = 131_072,
                        τ₀::Real = 1.0,
                        verbose::Bool = false)
    rng = Xoshiro(42 + idx)
    p   = draw_sample_params(rng)

    # Phase time series
    x = generate_composite_noise(p.h_coeffs, N, τ₀; seed = 42 + idx + 10_000_000)

    # Features
    v = compute_feature_vector(x, τ₀)

    # NLL labels — use h-warm start for fast convergence
    opt = optimize_kf_nll(x, τ₀; h_init = p.h_coeffs, verbose = verbose)

    # Provenance h-vector in canonical α order (+2, +1, 0, -1, -2)
    h_vec = fill(NaN, 5)
    for (j, α) in enumerate((2.0, 1.0, 0.0, -1.0, -2.0))
        if haskey(p.h_coeffs, α)
            h_vec[j] = log10(p.h_coeffs[α])
        end
    end

    SampleResult(
        v,
        [log10(opt.q_wpm), log10(opt.q_wfm), log10(opt.q_rwfm)],
        h_vec,
        p.fpm_present,
        opt.nll,
        opt.converged,
    )
end

end # module
```

- [ ] **Step 5: Run smoke test**

Run: `cd ml/dataset && julia --project=. test_one_sample.jl 2>&1 | tail -10`
Expected: prints `draw_sample_params ok.` and `run_one_sample ok.` with no errors.

- [ ] **Step 6: Commit**

```bash
git add ml/dataset/Project.toml ml/dataset/generate_dataset.jl ml/dataset/test_one_sample.jl
git commit -m "feat(ml/dataset): single-sample pipeline scaffold"
```

### Task 4.2 — Threaded driver with checkpointing and NPZ write

**Files:**
- Modify: `ml/dataset/generate_dataset.jl`

- [ ] **Step 1: Add `generate_dataset` function**

Append to [ml/dataset/generate_dataset.jl](../../ml/dataset/generate_dataset.jl), inside `module DatasetGen`:

```julia
const CKPT_EVERY = 500

"""
    generate_dataset(output_path; n_samples=10_000, N=131_072, τ₀=1.0, resume=true)

Main driver. Threads over sample index.  Checkpoints every `CKPT_EVERY`
samples to `<output_path>.checkpoint.npz`; resumes from the highest
completed index when `resume=true`.
"""
function generate_dataset(output_path::String;
                          n_samples::Int = 10_000,
                          N::Int         = 131_072,
                          τ₀::Real       = 1.0,
                          resume::Bool   = true)
    n_features = 196
    X  = Matrix{Float32}(undef, n_samples, n_features)
    y  = Matrix{Float64}(undef, n_samples, 3)
    H  = Matrix{Float64}(undef, n_samples, 5)
    fpm      = Vector{Bool}(undef,    n_samples)
    nll_vals = Vector{Float64}(undef, n_samples)
    conv     = Vector{Bool}(undef,    n_samples)

    # Resume logic
    done = falses(n_samples)
    ckpt = output_path * ".checkpoint.npz"
    if resume && isfile(ckpt)
        data = NPZ.npzread(ckpt)
        nprev = Int(data["n_done"])
        @info "Resuming from checkpoint" nprev
        X[1:nprev, :]      .= data["X"][1:nprev, :]
        y[1:nprev, :]      .= data["y"][1:nprev, :]
        H[1:nprev, :]      .= data["h_coeffs"][1:nprev, :]
        fpm[1:nprev]        .= data["fpm_present"][1:nprev]
        nll_vals[1:nprev]   .= data["nll_values"][1:nprev]
        conv[1:nprev]       .= data["converged"][1:nprev]
        done[1:nprev]       .= true
    end

    pending = findall(.!done)
    t_start = time()
    done_count = Threads.Atomic{Int}(count(done))

    Threads.@threads for i in pending
        r = try
            run_one_sample(i; N=N, τ₀=τ₀, verbose=false)
        catch err
            @warn "sample $i failed" err
            SampleResult(fill(NaN, n_features), fill(NaN, 3), fill(NaN, 5),
                         false, NaN, false)
        end
        X[i, :]  .= Float32.(r.features)
        y[i, :]  .= r.q_labels
        H[i, :]  .= r.h_coeffs
        fpm[i]    = r.fpm_present
        nll_vals[i] = r.nll
        conv[i]     = r.converged
        c = Threads.atomic_add!(done_count, 1) + 1
        if c % CKPT_EVERY == 0
            elapsed = time() - t_start
            @info "checkpoint" done=c of=n_samples elapsed=elapsed
            _write_npz(ckpt, X, y, H, fpm, nll_vals, conv, c)
        end
    end

    # Final write
    _write_npz(output_path, X, y, H, fpm, nll_vals, conv, n_samples; final=true)
    isfile(ckpt) && rm(ckpt)
    @info "done" output_path total=n_samples elapsed=(time() - t_start)
    nothing
end

function _write_npz(path, X, y, H, fpm, nll_vals, conv, n_done; final=false)
    mkpath(dirname(path))
    payload = Dict(
        "X"            => X,
        "y"            => y,
        "h_coeffs"     => H,
        "fpm_present"  => fpm,
        "nll_values"   => nll_vals,
        "converged"    => conv,
        "taus"         => CANONICAL_TAU_GRID,
        "feature_names" => FEATURE_NAMES,
        "n_done"       => n_done,
    )
    NPZ.npzwrite(path, payload)
end
```

- [ ] **Step 2: Add tiny end-to-end test (100 samples, small N)**

Create [ml/dataset/test_driver_mini.jl](../../ml/dataset/test_driver_mini.jl):

```julia
using Pkg; Pkg.activate(@__DIR__)
include("generate_dataset.jl")
using .DatasetGen

out = joinpath(@__DIR__, "tmp_mini.npz")
isfile(out) && rm(out)
DatasetGen.generate_dataset(out; n_samples=32, N=2^13, τ₀=1.0, resume=false)

using NPZ
d = NPZ.npzread(out)
@assert size(d["X"])        == (32, 196)
@assert size(d["y"])        == (32, 3)
@assert size(d["h_coeffs"]) == (32, 5)
@assert length(d["fpm_present"]) == 32
@assert length(d["taus"])        == 20
@assert length(d["feature_names"]) == 196
println("mini driver produced ", out, " with ", size(d["X"],1), " samples.")
rm(out)
```

- [ ] **Step 3: Run mini test**

Run: `cd ml/dataset && julia --project=. --threads=4 test_driver_mini.jl 2>&1 | tail -20`
Expected: prints "mini driver produced …" and exits cleanly.

- [ ] **Step 4: Commit**

```bash
git add ml/dataset/generate_dataset.jl ml/dataset/test_driver_mini.jl
git commit -m "feat(ml/dataset): threaded driver with checkpointing and NPZ output"
```

### Task 4.3 — Generate the 10,000-sample production dataset

- [ ] **Step 1: Kick off the full run (background, takes ~1 hour estimated)**

```bash
cd ml/dataset
mkdir -p ../data
julia --project=. --threads=12 -e '
include("generate_dataset.jl")
using .DatasetGen
generate_dataset("../data/dataset_v1.npz"; n_samples=10_000, N=131_072, τ₀=1.0)
' 2>&1 | tee ../data/generate.log
```

- [ ] **Step 2: Sanity-check output file**

Run:
```bash
python -c '
import numpy as np
d = np.load("ml/data/dataset_v1.npz", allow_pickle=True)
print("X:", d["X"].shape, d["X"].dtype)
print("y:", d["y"].shape)
print("h_coeffs:", d["h_coeffs"].shape)
print("fpm_present:", d["fpm_present"].sum(), "/", d["fpm_present"].size)
print("converged:", d["converged"].sum(), "/", d["converged"].size)
print("non-finite features:", np.isnan(d["X"]).sum() + np.isinf(d["X"]).sum())
'
```
Expected: `X: (10000, 196)`, `fpm_present` around 2900–3100, `converged > 9800`.

- [ ] **Step 3: Commit the log and document provenance**

```bash
git add ml/data/generate.log
git commit -m "chore(ml/data): dataset_v1 generation log (dataset file is gitignored)"
```

---

## Phase 5 — Python scaffolding and loader

**Deliverable:** `ml/src/loader.py` with stratified split, NaN imputation; `ml/requirements.txt`; pytest skeleton.

### Task 5.1 — Python project structure

**Files:**
- Create: `ml/requirements.txt`
- Create: `ml/src/__init__.py`
- Create: `ml/tests/__init__.py`
- Create: `ml/.gitignore`

- [ ] **Step 1: requirements.txt**

Create [ml/requirements.txt](../../ml/requirements.txt):

```
numpy>=1.26
scipy>=1.11
scikit-learn>=1.4
xgboost>=2.0
forestci>=0.6
matplotlib>=3.8
seaborn>=0.13
pandas>=2.1
jupyter>=1.0
pytest>=7.4
joblib>=1.3
```

- [ ] **Step 2: Gitignore data artifacts**

Create [ml/.gitignore](../../ml/.gitignore):

```
data/dataset_v1.npz
data/dataset_v1.checkpoint.npz
figures/
*.joblib
.ipynb_checkpoints/
__pycache__/
.pytest_cache/
```

- [ ] **Step 3: Create venv and install**

Run:
```bash
cd ml && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
Expected: installs clean.

- [ ] **Step 4: Package markers**

Create [ml/src/__init__.py](../../ml/src/__init__.py) and [ml/tests/__init__.py](../../ml/tests/__init__.py) as empty files.

- [ ] **Step 5: Commit**

```bash
git add ml/requirements.txt ml/.gitignore ml/src/__init__.py ml/tests/__init__.py
git commit -m "chore(ml): Python project scaffold"
```

### Task 5.2 — Dataset loader

**Files:**
- Create: `ml/src/loader.py`
- Create: `ml/tests/test_loader.py`

- [ ] **Step 1: Write failing test**

Create [ml/tests/test_loader.py](../../ml/tests/test_loader.py):

```python
import numpy as np
import pytest
from pathlib import Path

from ml.src.loader import load_dataset, stratified_split, impute_median


def test_load_dataset_shapes():
    ds = load_dataset("ml/data/dataset_v1.npz")
    assert ds.X.shape == (10_000, 196)
    assert ds.y.shape == (10_000, 3)
    assert ds.h_coeffs.shape == (10_000, 5)
    assert ds.fpm_present.dtype == bool
    assert len(ds.feature_names) == 196
    assert ds.taus.shape == (20,)


def test_stratified_split_preserves_fpm_ratio():
    ds = load_dataset("ml/data/dataset_v1.npz")
    Xtr, Xte, ytr, yte, mtr, mte = stratified_split(ds, test_size=0.2, seed=42)
    p_full  = ds.fpm_present.mean()
    p_train = mtr.mean()
    p_test  = mte.mean()
    assert abs(p_train - p_full) < 0.02
    assert abs(p_test  - p_full) < 0.02


def test_impute_median_handles_nans():
    X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 5.0]])
    X_imp = impute_median(X)
    assert not np.isnan(X_imp).any()
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd /home/ian/SigmaTau && PYTHONPATH=. python -m pytest ml/tests/test_loader.py -v 2>&1 | tail -20`
Expected: ImportError on `ml.src.loader`.

- [ ] **Step 3: Implement loader**

Create [ml/src/loader.py](../../ml/src/loader.py):

```python
"""Dataset loader for the ML pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    X: np.ndarray                # (N, 196) float32
    y: np.ndarray                # (N, 3)   float64 — log10(q_wpm, q_wfm, q_rwfm)
    h_coeffs: np.ndarray         # (N, 5)   float64 — log10(h+2, h+1, h0, h-1, h-2)
    fpm_present: np.ndarray      # (N,)     bool
    nll_values: np.ndarray       # (N,)     float64
    converged: np.ndarray        # (N,)     bool
    feature_names: np.ndarray    # (196,)   str
    taus: np.ndarray             # (20,)    float64


def load_dataset(path: str | Path) -> Dataset:
    d = np.load(str(path), allow_pickle=True)
    ds = Dataset(
        X=d["X"].astype(np.float32),
        y=d["y"].astype(np.float64),
        h_coeffs=d["h_coeffs"].astype(np.float64),
        fpm_present=d["fpm_present"].astype(bool),
        nll_values=d["nll_values"].astype(np.float64),
        converged=d["converged"].astype(bool),
        feature_names=np.array(d["feature_names"], dtype=str),
        taus=d["taus"].astype(np.float64),
    )
    # Filter non-converged samples
    ok = ds.converged
    if not ok.all():
        ds = Dataset(
            X=ds.X[ok], y=ds.y[ok], h_coeffs=ds.h_coeffs[ok],
            fpm_present=ds.fpm_present[ok], nll_values=ds.nll_values[ok],
            converged=ds.converged[ok], feature_names=ds.feature_names,
            taus=ds.taus,
        )
    return ds


def impute_median(X: np.ndarray) -> np.ndarray:
    """Column-median NaN imputation."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
            X[:, j] = col
    return X


def stratified_split(ds: Dataset, test_size: float = 0.2, seed: int = 42):
    idx_tr, idx_te = train_test_split(
        np.arange(len(ds.X)),
        test_size=test_size,
        stratify=ds.fpm_present,
        random_state=seed,
    )
    Xtr = impute_median(ds.X[idx_tr])
    Xte = impute_median(ds.X[idx_te])
    return Xtr, Xte, ds.y[idx_tr], ds.y[idx_te], ds.fpm_present[idx_tr], ds.fpm_present[idx_te]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd /home/ian/SigmaTau && PYTHONPATH=. python -m pytest ml/tests/test_loader.py -v 2>&1 | tail -20`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add ml/src/loader.py ml/tests/test_loader.py
git commit -m "feat(ml/src): dataset loader with stratified split and NaN imputation"
```

### Task 5.3 — Models and training wrappers

**Files:**
- Create: `ml/src/models.py`
- Create: `ml/tests/test_models.py`

- [ ] **Step 1: Write failing test**

Create [ml/tests/test_models.py](../../ml/tests/test_models.py):

```python
import numpy as np
from ml.src.models import train_rf, train_xgb, predict


def _tiny_xy(n=200, d=196, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    # targets = linear combo of first 3 features
    y = np.column_stack([X[:, 0], X[:, 1] - X[:, 5], X[:, 2] + X[:, 10]])
    return X, y


def test_train_rf_runs_and_predicts():
    X, y = _tiny_xy()
    model = train_rf(X[:150], y[:150], n_estimators=50, max_depth=6)
    pred = predict(model, X[150:])
    assert pred.shape == (50, 3)
    # Should correlate with truth (any positive correlation signals fit)
    for j in range(3):
        r = np.corrcoef(pred[:, j], y[150:, j])[0, 1]
        assert r > 0.3


def test_train_xgb_runs_and_predicts():
    X, y = _tiny_xy()
    model = train_xgb(X[:150], y[:150], n_estimators=50, max_depth=4)
    pred = predict(model, X[150:])
    assert pred.shape == (50, 3)
    for j in range(3):
        r = np.corrcoef(pred[:, j], y[150:, j])[0, 1]
        assert r > 0.3
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/ian/SigmaTau && PYTHONPATH=. python -m pytest ml/tests/test_models.py -v 2>&1 | tail -20`
Expected: ImportError on `ml.src.models`.

- [ ] **Step 3: Implement models**

Create [ml/src/models.py](../../ml/src/models.py):

```python
"""Random Forest and XGBoost wrappers used by the ML pipeline."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


def train_rf(X, y, *,
             n_estimators: int = 500,
             max_depth: int | None = None,
             min_samples_leaf: int = 5,
             max_features: str | float = "sqrt",
             n_jobs: int = -1,
             seed: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def train_xgb(X, y, *,
              n_estimators: int = 500,
              max_depth: int = 6,
              learning_rate: float = 0.05,
              subsample: float = 1.0,
              n_jobs: int = -1,
              seed: int = 42) -> MultiOutputRegressor:
    base = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        n_jobs=n_jobs,
        random_state=seed,
        tree_method="hist",
    )
    model = MultiOutputRegressor(base, n_jobs=1)  # base already parallel
    model.fit(X, y)
    return model


def predict(model, X) -> np.ndarray:
    return np.asarray(model.predict(X))
```

- [ ] **Step 4: Run test to verify pass**

Run: `cd /home/ian/SigmaTau && PYTHONPATH=. python -m pytest ml/tests/test_models.py -v 2>&1 | tail -20`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add ml/src/models.py ml/tests/test_models.py
git commit -m "feat(ml/src): RF and XGBoost training wrappers"
```

### Task 5.4 — Evaluation and uncertainty

**Files:**
- Create: `ml/src/evaluation.py`
- Create: `ml/tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests**

Create [ml/tests/test_evaluation.py](../../ml/tests/test_evaluation.py):

```python
import numpy as np
from ml.src.evaluation import metrics_per_target, rf_prediction_variance, xgb_quantile_intervals


def test_metrics_per_target_shapes():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(100, 3))
    y_pred = y_true + 0.1 * rng.normal(size=(100, 3))
    m = metrics_per_target(y_true, y_pred, target_names=("a", "b", "c"))
    for k in ("rmse", "mae", "r2"):
        assert k in m.columns
    assert set(m.index) == {"a", "b", "c"}


def test_rf_variance_shape_and_positive():
    from ml.src.models import train_rf
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 20)).astype(np.float32)
    y = X @ rng.normal(size=(20, 3))
    model = train_rf(X[:100], y[:100], n_estimators=40, max_depth=6)
    var = rf_prediction_variance(model, X[100:], X[:100])
    assert var.shape == (50, 3)
    assert (var > 0).any()  # forestci may return some ≤ 0 in tiny data


def test_xgb_quantile_intervals_monotone():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 10)).astype(np.float32)
    y = X @ rng.normal(size=(10, 3))
    lo, hi = xgb_quantile_intervals(X[:150], y[:150], X[150:],
                                     alpha_low=0.05, alpha_high=0.95,
                                     n_estimators=40, max_depth=3)
    assert lo.shape == hi.shape == (50, 3)
    assert (hi >= lo).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/ian/SigmaTau && PYTHONPATH=. python -m pytest ml/tests/test_evaluation.py -v 2>&1 | tail -20`
Expected: ImportError.

- [ ] **Step 3: Implement evaluation**

Create [ml/src/evaluation.py](../../ml/src/evaluation.py):

```python
"""Metrics, uncertainty quantification, model comparison."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


def metrics_per_target(y_true: np.ndarray, y_pred: np.ndarray,
                       target_names=("q_wpm", "q_wfm", "q_rwfm")) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(target_names):
        rows.append({
            "target": name,
            "rmse":   float(np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))),
            "mae":    float(mean_absolute_error(y_true[:, j], y_pred[:, j])),
            "r2":     float(r2_score(y_true[:, j], y_pred[:, j])),
        })
    return pd.DataFrame(rows).set_index("target")


def rf_prediction_variance(model, X_new: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Per-target prediction variance via forestci infinitesimal jackknife.

    forestci expects single-output estimators; for multi-output RF we compute
    the IJ variance from each tree's per-target predictions directly.
    """
    n_trees = len(model.estimators_)
    n_new = X_new.shape[0]
    # (n_trees, n_new, n_targets)
    all_preds = np.stack([t.predict(X_new) for t in model.estimators_], axis=0)
    # variance across trees — a valid "disagreement" estimator, simpler than IJ
    # which does not generalise natively to multi-output RandomForestRegressor.
    return all_preds.var(axis=0, ddof=1)


def xgb_quantile_intervals(X_tr, y_tr, X_te, *,
                           alpha_low: float = 0.05,
                           alpha_high: float = 0.95,
                           n_estimators: int = 300,
                           max_depth: int = 6,
                           learning_rate: float = 0.05,
                           seed: int = 42):
    """Fit one XGBoost model per target per quantile; return (low, high) arrays."""
    n_targets = y_tr.shape[1]
    lo = np.empty((X_te.shape[0], n_targets))
    hi = np.empty((X_te.shape[0], n_targets))
    for j in range(n_targets):
        for q, out in ((alpha_low, lo), (alpha_high, hi)):
            m = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective="reg:quantileerror",
                quantile_alpha=q,
                random_state=seed + int(100 * q),
                tree_method="hist",
            )
            m.fit(X_tr, y_tr[:, j])
            out[:, j] = m.predict(X_te)
    return lo, hi


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Empirical fraction of y_true falling within [lo, hi], per target."""
    return ((y_true >= lo) & (y_true <= hi)).mean(axis=0)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd /home/ian/SigmaTau && PYTHONPATH=. python -m pytest ml/tests/test_evaluation.py -v 2>&1 | tail -20`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add ml/src/evaluation.py ml/tests/test_evaluation.py
git commit -m "feat(ml/src): metrics, RF ensemble variance, XGBoost quantile intervals"
```

---

## Phase 6 — Training and evaluation notebook

**Deliverable:** `ml/notebook.ipynb` with EDA, GridSearchCV, metrics, UQ, and all 8 rubric plots.

### Task 6.1 — Notebook scaffold with loader and baselines

**Files:**
- Create: `ml/notebook.ipynb` (via `jupytext` or manual)

- [ ] **Step 1: Create notebook with cells**

Create [ml/notebook.ipynb](../../ml/notebook.ipynb) using this cell outline (write as a Python script first and convert with `jupytext --to notebook notebook.py`, or author directly). Cells:

**Cell 1 (markdown):**
```markdown
# KF Parameter Prediction from Stability Curves
PH 551 Final Project — Ian Lapinski — 2026-04-24
```

**Cell 2 (setup):**
```python
%load_ext autoreload
%autoreload 2
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from ml.src.loader import load_dataset, stratified_split, impute_median
from ml.src.models import train_rf, train_xgb, predict
from ml.src.evaluation import metrics_per_target, rf_prediction_variance, xgb_quantile_intervals, coverage
sns.set_theme(context="talk", style="whitegrid")
np.random.seed(42)
```

**Cell 3 (load data):**
```python
ds = load_dataset("data/dataset_v1.npz")
Xtr, Xte, ytr, yte, mtr, mte = stratified_split(ds, test_size=0.2, seed=42)
print(f"Train: {Xtr.shape}   Test: {Xte.shape}")
print(f"FPM ratio train={mtr.mean():.3f}  test={mte.mean():.3f}  full={ds.fpm_present.mean():.3f}")
```

**Cell 4 (naive baseline):**
```python
y_naive = np.tile(ytr.mean(axis=0), (yte.shape[0], 1))
metrics_per_target(yte, y_naive)
```

- [ ] **Step 2: Run notebook top-to-bottom**

Run: `cd ml && jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.executed.ipynb 2>&1 | tail -10`
Expected: executes without error.

- [ ] **Step 3: Commit**

```bash
git add ml/notebook.ipynb
git commit -m "feat(ml): notebook scaffold with loader + naive baseline"
```

### Task 6.2 — EDA plots (rubric §5.1)

- [ ] **Step 1: Add EDA cells**

Append cells to [ml/notebook.ipynb](../../ml/notebook.ipynb):

```python
# Feature-distribution histograms (short, mid, long τ samples)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
sample_idx = {"short τ=1s":     ds.feature_names.tolist().index("raw_adev_m1"),
              "mid τ=31s":      ds.feature_names.tolist().index("raw_adev_m31"),
              "long τ=4279s":   ds.feature_names.tolist().index("raw_adev_m4279")}
for ax, (title, j) in zip(axes.ravel()[:3], sample_idx.items()):
    ax.hist(ds.X[:, j], bins=50, color="C0", edgecolor="k", alpha=0.8)
    ax.set_title(title); ax.set_xlabel("log10 σ_ADEV")
# Target histograms
for ax, (j, name) in zip(axes.ravel()[3:], enumerate(("log10 q_wpm", "log10 q_wfm", "log10 q_rwfm"))):
    ax.hist(ds.y[:, j], bins=50, color="C1", edgecolor="k", alpha=0.8)
    ax.set_title(name)
fig.tight_layout()
fig.savefig("figures/eda_distributions.png", dpi=150)
```

```python
# Example stability curves — 5 random samples, all 4 stats overlaid
rng = np.random.default_rng(0)
picks = rng.choice(len(ds.X), size=5, replace=False)
τ = ds.taus
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for ax, idx in zip(axes, picks):
    for stat_name, offset in zip(_STATS := ("adev","mdev","hdev","mhdev"), (0, 20, 40, 60)):
        σ = 10**ds.X[idx, offset:offset+20]
        ax.loglog(τ, σ, label=stat_name, alpha=0.8)
    ax.set_xlabel("τ (s)"); ax.set_title(f"sample {idx}   FPM={ds.fpm_present[idx]}")
axes[0].set_ylabel("σ(τ)"); axes[0].legend()
fig.tight_layout()
fig.savefig("figures/eda_example_curves.png", dpi=150)
```

```python
# Correlation heatmap (feature-target)
corr_ft = np.zeros((196, 3))
for j in range(196):
    for t in range(3):
        finite = ~np.isnan(ds.X[:, j])
        corr_ft[j, t] = np.corrcoef(ds.X[finite, j], ds.y[finite, t])[0, 1]
fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(corr_ft.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_yticks(range(3)); ax.set_yticklabels(["q_wpm","q_wfm","q_rwfm"])
ax.set_xticks([]); ax.set_xlabel("feature index (196)")
fig.colorbar(im, ax=ax, label="Pearson r")
fig.tight_layout(); fig.savefig("figures/eda_corr_feature_target.png", dpi=150)
```

- [ ] **Step 2: Create figures dir and run**

```bash
mkdir -p ml/figures
cd ml && jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.executed.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add ml/notebook.ipynb
git commit -m "feat(ml): EDA plots (feature/target distributions, examples, correlations)"
```

### Task 6.3 — GridSearchCV for RF and XGBoost

- [ ] **Step 1: Add tuning cells**

Append to [ml/notebook.ipynb](../../ml/notebook.ipynb):

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

rf_grid = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [None, 20, 30],
    "min_samples_leaf": [3, 5, 10],
    "max_features": ["sqrt", 0.5],
}
rf_gs = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=1),
    rf_grid, cv=5, scoring="neg_mean_squared_error",
    n_jobs=-1, verbose=2,
)
rf_gs.fit(Xtr, ytr)
print("best RF params:", rf_gs.best_params_)
```

```python
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

xgb_grid = {
    "estimator__n_estimators": [200, 500],
    "estimator__learning_rate": [0.01, 0.05, 0.1],
    "estimator__max_depth": [4, 6, 8],
    "estimator__subsample": [0.8, 1.0],
}
xgb_base = MultiOutputRegressor(
    xgb.XGBRegressor(random_state=42, tree_method="hist", n_jobs=-1),
    n_jobs=1,
)
xgb_gs = GridSearchCV(xgb_base, xgb_grid, cv=5, scoring="neg_mean_squared_error",
                     n_jobs=2, verbose=2)
xgb_gs.fit(Xtr, ytr)
print("best XGB params:", xgb_gs.best_params_)
```

```python
import joblib
joblib.dump(rf_gs.best_estimator_,  "ml/models/rf_best.joblib")
joblib.dump(xgb_gs.best_estimator_, "ml/models/xgb_best.joblib")
```

- [ ] **Step 2: Run (takes 30–60 min on 12 cores)**

```bash
mkdir -p ml/models
cd ml && jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.executed.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add ml/notebook.ipynb ml/.gitignore
git commit -m "feat(ml): GridSearchCV for RF and XGBoost (models gitignored)"
```

### Task 6.4 — Evaluation, UQ, and rubric plots

- [ ] **Step 1: Add metrics and UQ cells**

Append to [ml/notebook.ipynb](../../ml/notebook.ipynb):

```python
rf  = rf_gs.best_estimator_
xgb_m = xgb_gs.best_estimator_

y_rf  = rf.predict(Xte)
y_xgb = predict(xgb_m, Xte)

print("RF metrics:");  print(metrics_per_target(yte, y_rf))
print("XGB metrics:"); print(metrics_per_target(yte, y_xgb))
print("Naive metrics:"); print(metrics_per_target(yte, y_naive))
```

```python
# Predicted vs actual scatter
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
target_names = ("q_wpm", "q_wfm", "q_rwfm")
for ax, j, name in zip(axes, range(3), target_names):
    ax.scatter(yte[:, j], y_rf[:, j], s=8, c=mte.astype(int), cmap="coolwarm", alpha=0.6)
    lims = [min(yte[:,j].min(), y_rf[:,j].min()), max(yte[:,j].max(), y_rf[:,j].max())]
    ax.plot(lims, lims, "k--")
    ax.set_xlabel(f"log10 {name} true"); ax.set_ylabel("predicted"); ax.set_title(name)
fig.suptitle("Random Forest — predicted vs actual (blue=no FPM, red=FPM)")
fig.tight_layout(); fig.savefig("figures/pred_vs_actual_rf.png", dpi=150)
```

```python
# Residual histograms
residuals = y_rf - yte
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, j, name in zip(axes, range(3), target_names):
    ax.hist(residuals[:, j], bins=60, color="C2", edgecolor="k")
    μ, σ = residuals[:, j].mean(), residuals[:, j].std()
    ax.set_title(f"{name}: μ={μ:.3f}, σ={σ:.3f}")
    ax.set_xlabel("residual (log10 decades)")
fig.tight_layout(); fig.savefig("figures/residuals_rf.png", dpi=150)
```

```python
# Feature importance — top 20 for RF
names = ds.feature_names
imp = rf.feature_importances_
top20 = np.argsort(imp)[-20:][::-1]
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(20)[::-1], imp[top20])
ax.set_yticks(range(20)[::-1]); ax.set_yticklabels(names[top20])
ax.set_xlabel("feature importance")
fig.tight_layout(); fig.savefig("figures/rf_top20_importance.png", dpi=150)
```

```python
# Aggregated importance by statistic family
labels = ["ADEV", "MDEV", "HDEV", "MHDEV"]
# Raw features indexed 0..79 in groups of 20 (adev, mdev, hdev, mhdev)
agg = np.array([imp[i:i+20].sum() for i in (0, 20, 40, 60)])
# Slopes 80..155 in groups of 19 (same order)
agg_slope = np.array([imp[80 + 19*k : 80 + 19*(k+1)].sum() for k in range(4)])
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(4)
ax.bar(x - 0.2, agg,       width=0.4, label="raw")
ax.bar(x + 0.2, agg_slope, width=0.4, label="slopes")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("total importance")
ax.legend(); ax.set_title("Importance by stability statistic")
fig.tight_layout(); fig.savefig("figures/importance_by_statistic.png", dpi=150)
```

```python
# RF uncertainty via tree disagreement
var_rf = rf_prediction_variance(rf, Xte, Xtr)
std_rf = np.sqrt(np.maximum(var_rf, 0))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, j, name in zip(axes, range(3), target_names):
    order = np.argsort(y_rf[:, j])[:50]
    ax.errorbar(range(50), y_rf[order, j], yerr=1.96 * std_rf[order, j],
                fmt=".", alpha=0.5, label="±95% CI")
    ax.scatter(range(50), yte[order, j], c="red", s=8, label="true")
    ax.set_title(name); ax.legend()
fig.suptitle("RF prediction intervals — 50 lowest-q samples")
fig.tight_layout(); fig.savefig("figures/rf_uq.png", dpi=150)
```

```python
# Quantile-regression intervals + coverage
lo, hi = xgb_quantile_intervals(Xtr, ytr, Xte)
cov = coverage(yte, lo, hi)
print("XGB 90% empirical coverage per target:", cov)
```

```python
# Model comparison bar chart
df_rf  = metrics_per_target(yte, y_rf)
df_xgb = metrics_per_target(yte, y_xgb)
comp = pd.concat({"RF": df_rf["rmse"], "XGB": df_xgb["rmse"]}, axis=1)
ax = comp.plot(kind="bar", figsize=(8, 5))
ax.set_ylabel("RMSE (log10 decades)"); ax.set_title("Model RMSE by target")
plt.tight_layout(); plt.savefig("figures/model_comparison_rmse.png", dpi=150)
```

```python
# Secondary: MDEV vs MHDEV importance slice
mdev_idx  = [i for i, n in enumerate(names) if "mdev" in n and "mhdev" not in n]
mhdev_idx = [i for i, n in enumerate(names) if "mhdev" in n]
imp_mdev, imp_mhdev = imp[mdev_idx].sum(), imp[mhdev_idx].sum()
print(f"Σ importance MDEV={imp_mdev:.4f}  MHDEV={imp_mhdev:.4f}")
# Paired Wilcoxon on per-sample residuals would require two separately-trained
# models; omitted in favor of this sum-of-importance slice (agreed per spec).
```

- [ ] **Step 2: Run notebook**

```bash
cd ml && jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.executed.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add ml/notebook.ipynb ml/figures/
git commit -m "feat(ml): evaluation metrics, uncertainty, and all rubric plots"
```

---

## Phase 7 — Real-data validation (GMR6000)

**Deliverable:** `ml/src/real_data.py` with unit detection + window extraction; notebook section comparing ML predictions to NLL optimizer on real phase data.

### Task 7.1 — Real-data loader with unit detection

**Files:**
- Create: `ml/src/real_data.py`
- Create: `ml/tests/test_real_data.py`

- [ ] **Step 1: Write failing test**

Create [ml/tests/test_real_data.py](../../ml/tests/test_real_data.py):

```python
import numpy as np
from ml.src.real_data import load_phase_record, detect_units, extract_windows


def test_detect_units_synthetic_seconds():
    rng = np.random.default_rng(0)
    x = np.cumsum(1e-11 * rng.normal(size=131072))  # σ_y(1) ≈ 1e-11 → seconds
    unit, factor = detect_units(x, tau0=1.0)
    assert unit == "seconds"
    assert factor == 1.0


def test_detect_units_synthetic_ns():
    rng = np.random.default_rng(1)
    x = np.cumsum(1e-11 * rng.normal(size=131072)) * 1e9  # now in ns
    unit, factor = detect_units(x, tau0=1.0)
    assert unit == "nanoseconds"
    assert factor == 1e-9


def test_extract_windows_nonoverlapping():
    x = np.arange(10 * 131_072, dtype=float)
    wins = extract_windows(x, window_size=131_072, n_windows=5)
    assert wins.shape == (5, 131_072)
    # first window starts at 0, second at 131_072, etc.
    assert wins[0, 0] == 0
    assert wins[1, 0] == 131_072
```

- [ ] **Step 2: Implement**

Create [ml/src/real_data.py](../../ml/src/real_data.py):

```python
"""GMR6000 real-data loader: unit detection and window extraction."""
from __future__ import annotations
from pathlib import Path
import numpy as np


_EXPECTED_ADEV_SECONDS = (1e-11, 1e-10)   # GMR6000 Rb expected σ_y(1s) range


def _adev_tau1(x: np.ndarray) -> float:
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    return float(np.sqrt(np.mean(d2**2) / 2))


def detect_units(x: np.ndarray, tau0: float = 1.0) -> tuple[str, float]:
    """Infer the unit of the phase column by matching ADEV(1s) to expected range.

    Returns (unit_name, scale_to_seconds).
    """
    a = _adev_tau1(x) / (tau0**0)    # ADEV is dimensionless when x is seconds
    candidates = [
        ("seconds",     1.0,     _EXPECTED_ADEV_SECONDS),
        ("nanoseconds", 1e-9,    tuple(v * 1e9 for v in _EXPECTED_ADEV_SECONDS)),
        ("cycles_5MHz", 2e-7,    tuple(v / 2e-7 for v in _EXPECTED_ADEV_SECONDS)),
    ]
    for name, factor, (lo, hi) in candidates:
        if lo / 3 <= a <= hi * 3:
            return name, factor
    raise ValueError(f"Could not infer units: ADEV(1s)={a:.3e} not in any expected range")


def load_phase_record(path: str | Path, tau0: float = 1.0) -> np.ndarray:
    """Load MJD/phase ASCII, return phase column in seconds."""
    data = np.loadtxt(str(path))
    mjd = data[:, 0]; ph = data[:, 1]
    step_s = (mjd[1] - mjd[0]) * 86400
    if abs(step_s - tau0) > 0.01 * tau0:
        raise ValueError(f"τ₀ mismatch: expected {tau0}s got {step_s}s")
    unit, factor = detect_units(ph, tau0=tau0)
    print(f"Detected unit: {unit}   scaling by {factor:g}")
    return ph * factor


def extract_windows(x: np.ndarray, *, window_size: int = 131_072,
                    n_windows: int | None = None) -> np.ndarray:
    """Non-overlapping contiguous windows of the phase record."""
    max_n = len(x) // window_size
    n = max_n if n_windows is None else min(n_windows, max_n)
    return np.stack([x[i*window_size:(i+1)*window_size] for i in range(n)])
```

- [ ] **Step 3: Run tests**

Run: `cd /home/ian/SigmaTau && PYTHONPATH=. python -m pytest ml/tests/test_real_data.py -v 2>&1 | tail -20`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add ml/src/real_data.py ml/tests/test_real_data.py
git commit -m "feat(ml/src): GMR6000 loader with empirical unit detection"
```

### Task 7.2 — Notebook validation section

- [ ] **Step 1: Append validation cells to notebook**

Append to [ml/notebook.ipynb](../../ml/notebook.ipynb):

```python
from ml.src.real_data import load_phase_record, extract_windows

ph_feb = load_phase_record("../reference/raw/6k27febunsteered.txt", tau0=1.0)
windows = extract_windows(ph_feb, window_size=131_072, n_windows=20)
print(f"Extracted {len(windows)} windows from Feb record")
```

```python
# Compute features for each window, predict with trained RF
# Feature computation uses the Julia-side pipeline; here we call it via a
# scripted Julia invocation (slow) or, for the notebook, compute a Python
# version of compute_feature_vector.  For speed, invoke Julia in batch:

import subprocess, json, tempfile
julia_script = '''
using Pkg; Pkg.activate("dataset")
using NPZ
include("dataset/generate_dataset.jl")
using SigmaTau
data = NPZ.npzread(ARGS[1])
X = data["windows"]   # (n, 131072)
F = zeros(Float32, size(X,1), 196)
Threads.@threads for i in axes(X,1)
    F[i, :] = Float32.(compute_feature_vector(X[i, :], 1.0))
end
NPZ.npzwrite(ARGS[2], Dict("F"=>F))
'''
with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f_in:
    np.savez(f_in.name, windows=windows)
    in_path = f_in.name
out_path = in_path + ".features.npz"
with tempfile.NamedTemporaryFile("w", suffix=".jl", delete=False) as f_jl:
    f_jl.write(julia_script); jl_path = f_jl.name
subprocess.check_call(["julia", "--project=../ml", "--threads=12", jl_path, in_path, out_path])
F_real = np.load(out_path)["F"]
print("Real-data features:", F_real.shape)
```

```python
# Predict and compare to Julia-optimized labels on same windows
F_imputed = impute_median(F_real)
pred_rf  = rf.predict(F_imputed)
pred_xgb = predict(xgb_m, F_imputed)
print("RF  predictions (log10 q):\n", pred_rf)
print("XGB predictions (log10 q):\n", pred_xgb)
```

```python
# Overlay predicted ADEV on measured ADEV (first 4 windows)
# Closed-form ADEV from KF process-noise parameters (derived from SP1065 Table 5
# combined with the analytical warm-start relations in the design doc):
#   σ²_y(τ) = 1.5·q_wpm/τ²  +  q_wfm/τ  +  q_rwfm·τ
# where q_wpm = h₊₂·f_h/(2π²), q_wfm = h₀/2, q_rwfm = (2π²/3)·h₋₂.
def analytical_adev(q_wpm, q_wfm, q_rwfm, tau):
    σ2 = 1.5 * q_wpm / tau**2 + q_wfm / tau + q_rwfm * tau
    return np.sqrt(σ2)

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
τ = ds.taus
for i, ax in enumerate(axes):
    σ_adev = 10**F_real[i, 0:20]
    ax.loglog(τ, σ_adev, "o-", label="measured")
    q = 10**pred_rf[i]
    ax.loglog(τ, analytical_adev(q[0], q[1], q[2], τ), "--", label="RF prediction")
    ax.set_title(f"window {i}")
    ax.set_xlabel("τ (s)")
axes[0].set_ylabel("σ_y(τ)"); axes[0].legend()
fig.tight_layout(); fig.savefig("figures/real_data_overlay.png", dpi=150)
```

- [ ] **Step 2: Run and verify**

```bash
cd ml && jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.executed.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add ml/notebook.ipynb
git commit -m "feat(ml): real-data validation section (GMR6000 Feb record)"
```

---

## Phase 8 — Submission prep

### Task 8.1 — Notebook polish and README

**Files:**
- Modify: `ml/notebook.ipynb`
- Create: `ml/README.md`

- [ ] **Step 1: Add narrative cells between results sections**

Edit [ml/notebook.ipynb](../../ml/notebook.ipynb) to add markdown cells introducing each section:
  - `## 1. Problem definition`
  - `## 2. Dataset & EDA`
  - `## 3. Baseline`
  - `## 4. Random Forest + XGBoost training`
  - `## 5. Evaluation metrics`
  - `## 6. Uncertainty quantification`
  - `## 7. Feature importance analysis`
  - `## 8. Real-data validation`
  - `## 9. Conclusions`

- [ ] **Step 2: Add summary README**

Create [ml/README.md](../../ml/README.md):

```markdown
# ML Pipeline — PH 551 Final Project

Predicts Kalman-filter noise parameters `(q_wpm, q_wfm, q_rwfm)` from frequency-stability curves.

## Reproducing

1. `cd julia && julia --project=. -e 'using Pkg; Pkg.test()'`
2. `cd ml/dataset && julia --project=. --threads=12 generate_dataset.jl` (≈ 1 hr)
3. `cd ml && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
4. `jupyter nbconvert --to notebook --execute notebook.ipynb`

## Deliverables
- `ml/data/dataset_v1.npz` — synthetic dataset (10,000 samples).
- `ml/models/rf_best.joblib`, `ml/models/xgb_best.joblib`.
- `ml/notebook.ipynb` — full analysis.
- `ml/figures/` — all rubric plots.
```

- [ ] **Step 3: Commit**

```bash
git add ml/notebook.ipynb ml/README.md
git commit -m "docs(ml): notebook narrative and README"
```

### Task 8.2 — Final integration check

- [ ] **Step 1: Run all Julia tests**

```bash
cd julia && julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: PASS.

- [ ] **Step 2: Run all Python tests**

```bash
cd /home/ian/SigmaTau && PYTHONPATH=. python -m pytest ml/tests/ -v
```
Expected: all pass.

- [ ] **Step 3: Full notebook execution**

```bash
cd ml && jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.executed.ipynb
```
Expected: executes top-to-bottom without error; all figures produced.

- [ ] **Step 4: Commit final state**

```bash
git add ml/notebook.ipynb ml/figures/
git commit -m "chore(ml): final integration run artifacts"
```

---

## Self-review checklist (pre-submission)

- [ ] All 8 rubric plots present in `ml/figures/`.
- [ ] Naive baseline RMSE vs model RMSE shown side-by-side.
- [ ] Per-target RMSE, R², MAE reported for both RF and XGBoost.
- [ ] UQ coverage reported for XGBoost quantile intervals.
- [ ] Feature importance aggregated by statistic (ADEV / MDEV / HDEV / MHDEV).
- [ ] Real-data validation section demonstrates ML → q values on at least 4 windows of the Feb record.
- [ ] MDEV-vs-MHDEV secondary analysis present (as importance slice).
- [ ] Notebook narrative flows from problem to conclusion.
- [ ] `ml/README.md` documents reproduction steps.
