# test_allan_family.jl — Tests for mdev and tdev deviation wrappers

# Slope-fitting helper: log-log least squares
function fit_slope(tau, dev)
    mask = .!isnan.(dev) .& (dev .> 0)
    lt = log.(tau[mask]); ld = log.(dev[mask])
    lt_c = lt .- sum(lt)/length(lt); ld_c = ld .- sum(ld)/length(ld)
    return sum(lt_c .* ld_c) / sum(lt_c .^ 2)
end

@testset "mdev and tdev" begin

    @testset "mdev basic API" begin
        Random.seed!(101)
        x = cumsum(randn(512))
        r = mdev(x, 1.0)
        @test r isa DeviationResult
        @test r.method == "mdev"
        @test length(r.tau) > 0
        @test all(r.tau .> 0)
        @test all(d -> isnan(d) || d > 0, r.deviation)
    end

    @testset "mdev White PM slope ≈ -3/2" begin
        # White PM (α=2): phase is white noise → MDEV slope ≈ τ^{-3/2}
        Random.seed!(42)
        N = 4096
        x = randn(N)
        r = mdev(x, 1.0)
        slope = fit_slope(r.tau, r.deviation)
        @test isapprox(slope, -1.5; atol=0.2)
    end

    @testset "mdev White FM slope ≈ -1/2" begin
        # White FM (α=0): phase is cumsum of white noise → MDEV slope ≈ τ^{-1/2}
        # (same as ADEV; Modified Allan does not change the White FM slope)
        Random.seed!(43)
        N = 4096
        x = cumsum(randn(N))
        r = mdev(x, 1.0)
        slope = fit_slope(r.tau, r.deviation)
        @test isapprox(slope, -0.5; atol=0.2)
    end

    @testset "mdev data_type=:freq matches manual conversion" begin
        Random.seed!(77)
        N    = 512
        tau0 = 1.0
        y    = randn(N)   # fractional-frequency samples

        r_freq  = mdev(y, tau0; data_type=:freq)
        x_phase = cumsum(y) .* tau0
        r_phase = mdev(x_phase, tau0)

        @test r_freq.tau  == r_phase.tau
        @test r_freq.neff == r_phase.neff
        nan_freq  = isnan.(r_freq.deviation)
        nan_phase = isnan.(r_phase.deviation)
        @test nan_freq == nan_phase
        @test r_freq.deviation[.!nan_freq] ≈ r_phase.deviation[.!nan_phase]  rtol=1e-12
    end

    @testset "tdev wraps mdev correctly" begin
        Random.seed!(55)
        N    = 512
        tau0 = 1.0
        x    = cumsum(randn(N))

        mr = mdev(x, tau0)
        tr = tdev(x, tau0)

        expected = mr.tau .* mr.deviation ./ sqrt(3)
        @test tr.deviation ≈ expected  rtol=1e-12
    end

    @testset "tdev/mdev relationship to rtol=1e-12" begin
        Random.seed!(66)
        N    = 1024
        tau0 = 1.0
        x    = randn(N)

        mr = mdev(x, tau0)
        tr = tdev(x, tau0)

        @test tr.deviation ≈ mr.tau .* mr.deviation ./ sqrt(3)  rtol=1e-12
    end

    @testset "tdev method field" begin
        Random.seed!(99)
        x = cumsum(randn(256))
        r = tdev(x, 1.0)
        @test r.method == "tdev"
    end

end  # @testset "mdev and tdev"
