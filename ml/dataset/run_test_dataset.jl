# run_test_dataset.jl — Generate a 100-sample test dataset for methodology validation.
#
# Usage (from ml/dataset/):
#   julia --project=. --threads=12 run_test_dataset.jl
#
# Approx runtime: ~6 seconds on 12 threads at N=524288 (0.7 s/sample).
# Output: ml/data/dev_100.h5
#
# Purpose: exercise every notebook cell (GridSearchCV, quantile intervals, etc.)
# without waiting for the full 10k production run. Uses the SAME h-ranges and
# N as the production dataset so the schema and statistics match exactly.

using Pkg; Pkg.activate(@__DIR__)
include("generate_dataset.jl")
using .DatasetGen

const OUTPUT_PATH = joinpath(@__DIR__, "..", "data", "dev_100.h5")

println("Generating 100-sample test dataset...")
t0 = time()
generate_dataset(OUTPUT_PATH;
                 n_samples = 100,
                 N         = 524_288,
                 τ₀        = 1.0,
                 resume    = false)
println("Done in ", round(time() - t0, digits=2), " s")
println("Output: ", OUTPUT_PATH)
