# generate.jl — reproducible mixed-noise validation dataset
#
# Produces a 2^15-point phase-data file composed of independent WPM (α=+2),
# WFM (α=0), and RWFM (α=−2) realisations, scaled so the ADEV slope crosses
# −1 → −1/2 near τ ≈ 32 and −1/2 → +1/2 near τ ≈ 512.
#
# Spec: docs/superpowers/specs/2026-04-13-mixed-noise-validation-dataset-design.md
#
# Run from the julia/ directory:
#     julia --project=. ../examples/mixed_noise_validation/generate.jl

using SigmaTau
using Random
using Printf

const OUTDIR  = @__DIR__
const REFDIR  = joinpath(OUTDIR, "reference")

const N       = 2^15      # 32_768
const TAU0    = 1.0
const TAU_X1  = 32.0      # WPM → WFM crossover target
const TAU_X2  = 512.0     # WFM → RWFM crossover target

const SEED_WPM  = 1001
const SEED_WFM  = 1002
const SEED_RWFM = 1003

# ── 1. Generate three independent pure-noise phase realisations ──────────────
# WPM (α=+2):  phase itself is white             → x = randn()
# WFM (α= 0):  fractional frequency is white     → x = cumsum(randn())
# RWFM (α=-2): fractional frequency is a walk    → x = cumsum(cumsum(randn()))
#
# Explicit Xoshiro RNGs keep the three streams independent and stable across
# Julia versions (no reliance on the global default RNG).

x_wpm  = randn(Xoshiro(SEED_WPM),  N)
x_wfm  = cumsum(randn(Xoshiro(SEED_WFM), N))
x_rwfm = cumsum(cumsum(randn(Xoshiro(SEED_RWFM), N)))

# ── 2. Measure A(τ=1) for each pure component ────────────────────────────────
A_wpm1  = adev(x_wpm,  TAU0).deviation[1]
A_wfm1  = adev(x_wfm,  TAU0).deviation[1]
A_rwfm1 = adev(x_rwfm, TAU0).deviation[1]

# ── 3. Solve crossover equations ─────────────────────────────────────────────
# Closed-form ADEV slopes:  WPM τ^(-1), WFM τ^(-1/2), RWFM τ^(+1/2).
#
# Crossover 1 (τ = TAU_X1, WPM=WFM):
#   c_WPM · A_WPM(1) · τ^(-1)   =  c_WFM · A_WFM(1) · τ^(-1/2)
#   ⇒  c_WPM  = c_WFM · (A_WFM(1) / A_WPM(1)) · sqrt(τ)
#
# Crossover 2 (τ = TAU_X2, WFM=RWFM):
#   c_WFM · A_WFM(1) · τ^(-1/2) = c_RWFM · A_RWFM(1) · τ^(+1/2)
#   ⇒  c_RWFM = c_WFM · (A_WFM(1) / A_RWFM(1)) / τ
const c_WFM  = 1.0
const c_WPM  = c_WFM * (A_wfm1 / A_wpm1)  * sqrt(TAU_X1)
const c_RWFM = c_WFM * (A_wfm1 / A_rwfm1) / TAU_X2

# ── 4. Build the mixed phase record ──────────────────────────────────────────
x_mixed = c_WPM .* x_wpm .+ c_WFM .* x_wfm .+ c_RWFM .* x_rwfm

# ── 5. Write mixed_noise.txt (single column, %.17g round-trips IEEE 754) ────
open(joinpath(OUTDIR, "mixed_noise.txt"), "w") do io
    for v in x_mixed
        @printf(io, "%.17g\n", v)
    end
end

# ── 6. Reference σ(τ) tables for every deviation ─────────────────────────────
# Octave m-list up to ⌊N/4⌋ — the convention used in the existing
# noise_id_validation/ examples. Kernels that can't support the largest m
# return NaN for that row (engine leaves tau/m in place, deviation=NaN).
mkpath(REFDIR)
const m_list = [2^k for k in 0:floor(Int, log2(N / 4))]

const DEV_FNS = (
    (adev,     "adev"),
    (mdev,     "mdev"),
    (hdev,     "hdev"),
    (mhdev,    "mhdev"),
    (totdev,   "totdev"),
    (mtotdev,  "mtotdev"),
    (htotdev,  "htotdev"),
    (mhtotdev, "mhtotdev"),
    (tdev,     "tdev"),
    (ldev,     "ldev"),
)

for (fn, name) in DEV_FNS
    r = fn(x_mixed, TAU0; m_list = m_list)
    open(joinpath(REFDIR, "$(name).csv"), "w") do io
        println(io, "m,tau,sigma,edf,ci_lo,ci_hi,alpha_id")
        for i in eachindex(r.tau)
            m_i = round(Int, r.tau[i] / r.tau0)
            @printf(io,
                "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%d\n",
                m_i,
                r.tau[i],
                r.deviation[i],
                r.edf[i],
                r.ci[i, 1],
                r.ci[i, 2],
                r.alpha[i],
            )
        end
    end
end

# ── 7. Raw noise_id table (continuous-valued α from lag-1 ACF / B1/Rn) ───────
alphas = noise_id(x_mixed, m_list, "phase")
open(joinpath(REFDIR, "noise_id.csv"), "w") do io
    println(io, "m,tau,alpha_id")
    for (i, m) in enumerate(m_list)
        @printf(io, "%d,%.17g,%.17g\n", m, m * TAU0, alphas[i])
    end
end

# ── 8. Summary: report achieved crossovers ───────────────────────────────────
# Print ADEV values bracketing each target crossover so the slope transition
# (−1 → −1/2 near τ≈32, −1/2 → +1/2 near τ≈512) is visible at a glance.
r_adev = adev(x_mixed, TAU0; m_list = m_list)
idx_of(m) = findfirst(==(Float64(m)), r_adev.tau)

function local_slope(m_lo, m_hi)
    ilo, ihi = idx_of(m_lo), idx_of(m_hi)
    log(r_adev.deviation[ihi] / r_adev.deviation[ilo]) /
        log(r_adev.tau[ihi]        / r_adev.tau[ilo])
end

println("mixed_noise_validation generator summary")
println("  N           = $N")
println("  tau0        = $TAU0")
println("  c_WPM       = $(c_WPM)")
println("  c_WFM       = $(c_WFM)")
println("  c_RWFM      = $(c_RWFM)")
println("  A_WPM(1)    = $(A_wpm1)")
println("  A_WFM(1)    = $(A_wfm1)")
println("  A_RWFM(1)   = $(A_rwfm1)")
println()
println("  ADEV around crossover 1 (target τ ≈ $(TAU_X1), WPM→WFM):")
for m in (16, 32, 64)
    i = idx_of(m)
    @printf("    τ=%-5d  σ=%.6f\n", m, r_adev.deviation[i])
end
@printf("    local slope log σ / log τ on [16,64] = %+.3f  (expect between -1 and -1/2)\n",
        local_slope(16, 64))
println()
println("  ADEV around crossover 2 (target τ ≈ $(TAU_X2), WFM→RWFM):")
for m in (256, 512, 1024)
    i = idx_of(m)
    @printf("    τ=%-5d  σ=%.6f\n", m, r_adev.deviation[i])
end
@printf("    local slope log σ / log τ on [256,1024] = %+.3f  (expect between -1/2 and +1/2)\n",
        local_slope(256, 1024))
println()
println("  files       = mixed_noise.txt + reference/*.csv (11 files)")
