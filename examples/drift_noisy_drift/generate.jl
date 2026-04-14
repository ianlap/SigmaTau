# generate.jl — synthetic drift/noisy-drift comparison dataset
#
# Run from repo root:
#   julia --project=julia examples/drift_noisy_drift/generate.jl

using SigmaTau
using Random
using Printf
using Statistics

const OUTDIR = joinpath(@__DIR__, "out")
mkpath(OUTDIR)

const N    = 2^17               # 131072 points: enough dynamic range for long-τ tails
const TAU0 = 1.0
const m_list = [2^k for k in 0:floor(Int, log2(N ÷ 8))]

const SEED_BASE_WFM  = 3301
const SEED_BASE_RWFM = 3302
const SEED_RRFM_LIKE = 3303

# 1) Baseline long-memory record (WFM + RWFM)
x_wfm  = cumsum(randn(Xoshiro(SEED_BASE_WFM), N))
x_rwfm = cumsum(cumsum(randn(Xoshiro(SEED_BASE_RWFM), N)))

A_wfm1  = adev(x_wfm, TAU0).deviation[1]
A_rwfm1 = adev(x_rwfm, TAU0).deviation[1]

# Place the WFM↔RWFM crossover in mid τ so long τ clearly shows RWFM-like growth.
const TAU_CROSS = 128.0
const C_RWFM = (A_wfm1 / A_rwfm1) / TAU_CROSS

x_base = x_wfm .+ C_RWFM .* x_rwfm

# 2) Deterministic drift: add quadratic phase trend x_drift(t)=k*t^2
# Choose k so quadratic term is gentle at short τ but dominant at long τ.
t = collect(0.0:(N - 1.0))
const K_QUAD = 2.0e-11
x_det = x_base .+ K_QUAD .* (t .^ 2)

# 3) Noisy drift: deterministic trend + RRFM-like random drift component.
# RRFM-like phase is approximated as triple integration of white noise.
x_rrfm_like = cumsum(cumsum(cumsum(randn(Xoshiro(SEED_RRFM_LIKE), N))))
A_rrfm1 = adev(x_rrfm_like, TAU0).deviation[1]
A_base1 = adev(x_base, TAU0).deviation[1]

# Scale so RRFM-like process is mostly a long-τ effect.
const TAU_RRFM_VIS = 1024.0
const C_RRFM = 0.45 * (A_base1 / A_rrfm1) / (TAU_RRFM_VIS^(3/2))
x_noisy = x_det .+ C_RRFM .* x_rrfm_like

const CASES = (
    ("baseline",            x_base),
    ("deterministic_drift", x_det),
    ("noisy_drift",         x_noisy),
)

const DEV_FNS = (
    (adev,  "adev"),
    (hdev,  "hdev"),
    (mdev,  "mdev"),
    (mhdev, "mhdev"),
    (tdev,  "tdev"),
    (ldev,  "ldev"),
)

function write_table(path, r)
    open(path, "w") do io
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

# Save phase records too (single-column) for external checks.
for (name, x) in CASES
    open(joinpath(OUTDIR, "phase_$(name).txt"), "w") do io
        for v in x
            @printf(io, "%.17g\n", v)
        end
    end

    for (fn, devname) in DEV_FNS
        r = fn(x, TAU0; m_list = m_list)
        write_table(joinpath(OUTDIR, "$(name)_$(devname).csv"), r)
    end
end

# Small long-τ diagnostic summary for quick terminal validation.
function tail_slope(x, fn)
    r = fn(x, TAU0; m_list = m_list)
    valid = findall(.!isnan.(r.deviation))
    i0 = valid[max(1, end - 3)]
    i1 = valid[end]
    return log(r.deviation[i1] / r.deviation[i0]) / log(r.tau[i1] / r.tau[i0]), r.tau[i0], r.tau[i1]
end

println("drift_noisy_drift generator summary")
println("  N                = $(N)")
println("  tau0             = $(TAU0)")
println("  m_max            = $(m_list[end])")
println("  C_RWFM           = $(C_RWFM)")
println("  K_QUAD           = $(K_QUAD)")
println("  C_RRFM           = $(C_RRFM)")
for (name, x) in CASES
    sA, t0, t1 = tail_slope(x, adev)
    sH, _, _ = tail_slope(x, hdev)
    @printf("  %-20s tail slopes over τ=[%.0f, %.0f]:  ADEV=%+.3f, HDEV=%+.3f\n", name, t0, t1, sA, sH)
end
println("  wrote phase_*.txt + *_{adev,hdev,mdev,mhdev,tdev,ldev}.csv in $OUTDIR")
