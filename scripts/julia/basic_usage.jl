#!/usr/bin/env julia
# basic_usage.jl — Simple example of using SigmaTau for stability analysis.

using SigmaTau
using Random
using Printf

# 1. Generate sample phase data (White FM: phase is a random walk)
Random.seed!(42)
N    = 1024
tau0 = 1.0
y    = randn(N-1)           # white frequency noise
x    = [0.0; cumsum(y)] .* tau0  # integrate to phase

println("SigmaTau Basic Usage")
println("--------------------")
@printf("Generated %d samples (tau0 = %.1fs)\n\n", N, tau0)

# 2. Compute Overlapping Allan Deviation (ADEV)
println("Computing ADEV...")
res_adev = adev(x, tau0)

# 3. Compute Modified Allan Deviation (MDEV)
println("Computing MDEV...")
res_mdev = mdev(x, tau0)

# 4. Display results
@printf("%10s | %15s | %15s | %10s\n", "Tau [s]", "ADEV", "MDEV", "Alpha (ID)")
println("-" * 60)

for i in 1:length(res_adev.tau)
    # Both use the same default m_list, so we can align them
    @printf("%10.1f | %15.6e | %15.6e | %10d\n", 
            res_adev.tau[i], res_adev.deviation[i], res_mdev.deviation[i], res_adev.alpha[i])
end

# 5. Accessing Confidence Intervals
println("\nConfidence Intervals (ADEV, 68.3%):")
for i in 1:min(3, length(res_adev.tau))
    @printf("  tau = %5.1f: [%.4e, %.4e]\n", 
            res_adev.tau[i], res_adev.ci[i, 1], res_adev.ci[i, 2])
end

println("\nDone.")
