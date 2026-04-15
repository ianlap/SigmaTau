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
