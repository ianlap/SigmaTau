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

    @testset "composite recovers WPM + WFM mix" begin
        # WPM dominates short τ (ADEV slope −1); WFM dominates long τ (slope −1/2)
        # h_wpm >> h_wfm so WPM ADEV exceeds WFM at short τ (WPM ADEV ∝ τ^−1,
        # WFM ADEV ∝ τ^−1/2; equal h gives WFM ~3.6× larger at τ=1s).
        h_coeffs = Dict(2.0 => 1e-19,  # WPM — large enough to dominate at short τ
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
        # sort([-2.0, 2.0]) == [-2.0, 2.0], so RWFM gets offset=1 (seed=7), WPM gets offset=2 (seed=8)
        x_rwf = generate_power_law_noise(-2.0, 1e-24, 1024, 1.0; seed=7)
        x_wpm = generate_power_law_noise( 2.0, 1e-22, 1024, 1.0; seed=8)
        @test isapprox(x, x_wpm .+ x_rwf; rtol=1e-12)
    end

end
