#!/usr/bin/env julia
# Compute all 10 SigmaTau deviations on the first N_SUBSET points of a phase
# dataset in reference/. Runs each deviation SEQUENTIALLY so per-deviation
# wall clock is directly comparable to Stable32. Writes the trimmed subset
# alongside the results so Stable32 can load the same data.
#
# Usage (from repo root):
#   julia --threads=auto --project=julia examples/kf_pipeline/compute_devs.jl <dataset>

using Dates
using DelimitedFiles
using Printf
using Statistics
using Base.Threads

isempty(ARGS) && error("usage: compute_devs.jl <dataset>  (basename in reference/)")
const DATASET     = ARGS[1]
const HERE        = @__DIR__
const REPO        = abspath(joinpath(HERE, "..", ".."))
const DATA        = joinpath(REPO, "reference", "$(DATASET).txt")
const RES_ROOT    = joinpath(HERE, "results", DATASET)
const OUT_DIR     = joinpath(RES_ROOT, "devs")
const DATA_DIR    = joinpath(RES_ROOT, "data")
const SEC_PER_DAY = 86400.0
const N_SUBSET    = 50_000   # first N points; keeps mtotdev/htotdev tractable

"""Print a timestamped status line and flush immediately."""
function log(msg::AbstractString)
    ts = Dates.format(now(), "HH:MM:SS.sss")
    println(stdout, "[$ts] $msg")
    flush(stdout)
end
isdir(OUT_DIR) || mkpath(OUT_DIR)
isdir(DATA_DIR) || mkpath(DATA_DIR)

using Pkg
Pkg.activate(joinpath(REPO, "julia"); io = devnull)

log("julia $(VERSION), threads=$(Threads.nthreads())")
log("activated project: $(joinpath(REPO, "julia"))")
log("loading SigmaTau (first call triggers precompile; can take 30-90s)...")
# `using` must be at top level — time it via a wall-clock delta.
const _T_BEFORE_USING = time()
using SigmaTau
log(@sprintf("  SigmaTau loaded in %.2fs", time() - _T_BEFORE_USING))

# ── Load data ────────────────────────────────────────────────────────────────
log("reading $(DATA) ...")
t_load = @elapsed (raw = readdlm(DATA))
log(@sprintf("  read %d rows in %.2fs", size(raw, 1), t_load))

N_full = size(raw, 1)
N      = min(N_SUBSET, N_full)
mjd    = collect(Float64, @view raw[1:N, 1])
x      = collect(Float64, @view raw[1:N, 2])          # phase column
tau0   = median(diff(mjd)) * SEC_PER_DAY              # infer from MJD spacing
log(@sprintf("  using first %d of %d rows  (tau0=%.6fs  record=%.3f days)",
             N, N_full, tau0, (N-1)*tau0/SEC_PER_DAY))

# Dump the subset so Stable32 can load the same data for comparison.
subset_file = joinpath(DATA_DIR, "$(DATASET)_first$(N).txt")
open(subset_file, "w") do io
    for i in 1:N
        @printf(io, "%.11f %.6f\n", mjd[i], x[i])
    end
end
log("  wrote subset for Stable32: $(subset_file)")

# ── Deviations to run ────────────────────────────────────────────────────────
fns = [
    ("adev",     adev),
    ("mdev",     mdev),
    ("hdev",     hdev),
    ("mhdev",    mhdev),
    ("tdev",     tdev),
    ("ldev",     ldev),
    ("totdev",   totdev),
    ("mtotdev",  mtotdev),
    ("htotdev",  htotdev),
    ("mhtotdev", mhtotdev),
]

# JIT warmup is baked into SigmaTau.jl via PrecompileTools.@compile_workload,
# so the first call to any deviation is already warm. No runtime warmup needed.

# ── Run each deviation sequentially ──────────────────────────────────────────
log("running $(length(fns)) deviations sequentially on N=$(N) points")
println(stdout)
flush(stdout)

results = Tuple{String, Any, Float64}[]
t_total = @elapsed begin
    for (name, fn) in fns
        log(@sprintf("  %-10s starting ...", name))
        t_elapsed = @elapsed res = fn(x, tau0)
        ntau = length(res.tau)
        tau_max = ntau == 0 ? NaN : maximum(res.tau)
        log(@sprintf("  %-10s done: %.3fs  (%d tau points, tau_max=%.1fs)",
                     name, t_elapsed, ntau, tau_max))
        push!(results, (name, res, t_elapsed))
    end
end

# ── Write CSVs ───────────────────────────────────────────────────────────────
println(stdout); flush(stdout)
log("writing per-deviation CSVs to $(OUT_DIR) ...")
for (name, res, _) in results
    csv = joinpath(OUT_DIR, string(name, ".csv"))
    open(csv, "w") do io
        println(io, "tau,deviation,ci_lo,ci_hi,edf,alpha,neff")
        for k in eachindex(res.tau)
            @printf(io, "%.9g,%.9g,%.9g,%.9g,%.6g,%d,%d\n",
                    res.tau[k], res.deviation[k],
                    res.ci[k, 1], res.ci[k, 2],
                    res.edf[k], res.alpha[k], res.neff[k])
        end
    end
end

# ── Summary ──────────────────────────────────────────────────────────────────
println(stdout)
@printf("%-10s %10s %10s %10s\n", "dev", "seconds", "n_tau", "tau_max")
println("-"^44)
for (name, res, t_elapsed) in results
    ntau = length(res.tau)
    tau_max = ntau == 0 ? NaN : maximum(res.tau)
    @printf("%-10s %10.3f %10d %10.1f\n", name, t_elapsed, ntau, tau_max)
end
println("-"^44)
@printf("%-10s %10.3f  (wall clock)\n", "TOTAL", t_total)
flush(stdout)

open(joinpath(OUT_DIR, "timing.csv"), "w") do io
    println(io, "deviation,seconds,n_tau")
    for (name, res, t_elapsed) in results
        @printf(io, "%s,%.6f,%d\n", name, t_elapsed, length(res.tau))
    end
    @printf(io, "# total_wall_s=%.6f threads=%d N=%d tau0=%.6f\n",
            t_total, Threads.nthreads(), N, tau0)
end
log("wrote $(length(fns)) CSVs + timing.csv to $(OUT_DIR)")
