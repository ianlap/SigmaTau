# run_production.jl — Generate the full 10,000-sample ML dataset.
#
# Usage (from ml/dataset/):
#   julia --project=. --threads=12 run_production.jl
#
# Approx runtime: ~0.7 s/sample on 12 threads at N=524288 → ~2 hr for 10k.
# Output:
#   ml/data/dataset_v1.h5                 — final HDF5 dataset
#   ml/data/dataset_v1.h5.checkpoint.h5   — removed on successful finish
#   ml/data/dataset_v1.log                — tee'd stdout (wall-time log)
#
# Checkpointing: writes every CKPT_EVERY=500 completed samples. Safe to kill
# and restart; resume=true will pick up from the last checkpoint.
#
# To force a fresh run (discard any existing checkpoint):
#   julia --project=. --threads=12 -e 'include("generate_dataset.jl"); using .DatasetGen; DatasetGen.generate_dataset("../data/dataset_v1.h5"; resume=false)'

using Pkg; Pkg.activate(@__DIR__)
include("generate_dataset.jl")
using .DatasetGen

const OUTPUT_PATH = joinpath(@__DIR__, "..", "data", "dataset_v1.h5")
const N_SAMPLES   = 10_000
const N_POINTS    = 524_288          # 2^19 ≈ 6 days at τ₀=1s
const TAU0        = 1.0

function main()
    nthreads = Threads.nthreads()
    est_per_sample_s = 0.7                                   # from dev_25 benchmark
    est_total_min    = est_per_sample_s * N_SAMPLES / nthreads / 60

    println("=" ^ 60)
    println("SigmaTau ML — production dataset generation")
    println("=" ^ 60)
    println("output     : ", OUTPUT_PATH)
    println("n_samples  : ", N_SAMPLES)
    println("N_points   : ", N_POINTS)
    println("τ₀         : ", TAU0, " s")
    println("threads    : ", nthreads)
    println("ETA        : ~", round(est_total_min, digits=1), " min")
    println("=" ^ 60)
    println("Checkpoints every ", DatasetGen.CKPT_EVERY, " samples. Safe to Ctrl-C and restart.")
    println()

    t0 = time()
    generate_dataset(OUTPUT_PATH;
                     n_samples = N_SAMPLES,
                     N         = N_POINTS,
                     τ₀        = TAU0,
                     resume    = true)
    elapsed = time() - t0

    println()
    println("=" ^ 60)
    println("Done in ", round(elapsed / 60, digits=2), " min (",
            round(elapsed / N_SAMPLES, digits=3), " s/sample actual)")
    println("Output: ", OUTPUT_PATH)
    println("=" ^ 60)
end

main()
