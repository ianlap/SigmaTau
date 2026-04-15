# test_noise_gen.jl — composite power-law noise generator (Kasdin & Walter, 1992)

using Random
using Statistics

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
        # Test at τ=1s where WFM ADEV(τ=1s) = √(h₀/2) exactly (SP1065 Table 5)
        mask = (r.tau .== 1.0) .& .!isnan.(r.deviation)
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
