# test_hadamard_family.jl — Tests for hdev, mhdev, ldev
# Included inside @testset "SigmaTau" in runtests.jl.
# Random and SigmaTau are already in scope from runtests.jl.

function fit_slope(tau, dev)
    mask = .!isnan.(dev) .& (dev .> 0)
    lt = log.(tau[mask]); ld = log.(dev[mask])
    lt_c = lt .- sum(lt)/length(lt); ld_c = ld .- sum(ld)/length(ld)
    return sum(lt_c .* ld_c) / sum(lt_c .^ 2)
end

@testset "hdev" begin

    @testset "basic API" begin
        Random.seed!(100)
        x = cumsum(randn(512))
        r = hdev(x, 1.0)
        @test r isa DeviationResult
        @test r.method == "hdev"
        @test length(r.tau) > 0
        @test all(r.tau .> 0)
        @test all(d -> isnan(d) || d > 0, r.deviation)
    end

    @testset "White FM slope ≈ -1/2" begin
        # White FM: cumsum of white noise → HDEV slope ≈ τ^{-1/2}
        Random.seed!(101)
        x = cumsum(randn(4096))
        r = hdev(x, 1.0)
        slope = fit_slope(r.tau, r.deviation)
        @test isapprox(slope, -0.5; atol=0.2)
    end

    @testset "RWFM slope ≈ +1/2 (random walk FM)" begin
        # RWFM (α=-2): double cumsum → HDEV slope ≈ τ^{+1/2}
        # Hadamard suppresses deterministic linear drift (aging), not stochastic RWFM.
        Random.seed!(102)
        x = cumsum(cumsum(randn(4096)))
        r = hdev(x, 1.0)
        slope = fit_slope(r.tau, r.deviation)
        @test isapprox(slope, 0.5; atol=0.25)
    end

    @testset "data_type=:freq matches manual freq→phase conversion" begin
        Random.seed!(103)
        N    = 512
        tau0 = 1.0
        y    = randn(N)

        r_freq  = hdev(y, tau0; data_type=:freq)
        r_phase = hdev(cumsum(y) .* tau0, tau0)

        @test r_freq.tau  == r_phase.tau
        @test r_freq.neff == r_phase.neff
        nan_f = isnan.(r_freq.deviation)
        nan_p = isnan.(r_phase.deviation)
        @test nan_f == nan_p
        @test r_freq.deviation[.!nan_f] ≈ r_phase.deviation[.!nan_p]  rtol=1e-12
    end

end  # @testset "hdev"

@testset "mhdev" begin

    @testset "basic API" begin
        Random.seed!(200)
        x = cumsum(randn(512))
        r = mhdev(x, 1.0)
        @test r isa DeviationResult
        @test r.method == "mhdev"
        @test length(r.tau) > 0
        @test all(r.tau .> 0)
        @test all(d -> isnan(d) || d > 0, r.deviation)
    end

    @testset "White FM slope ≈ -1/2" begin
        # White FM (α=0) → MHDEV slope ≈ τ^{-1/2}, same as ADEV and HDEV.
        # (Modified Hadamard uses a moving-average window of length m; for white FM
        # the averaging does not change the -1/2 slope — consistent with MDEV.)
        Random.seed!(201)
        x = cumsum(randn(4096))
        r = mhdev(x, 1.0)
        slope = fit_slope(r.tau, r.deviation)
        @test isapprox(slope, -0.5; atol=0.2)
    end

    @testset "data_type=:freq passthrough" begin
        Random.seed!(202)
        N    = 512
        tau0 = 1.0
        y    = randn(N)

        r_freq  = mhdev(y, tau0; data_type=:freq)
        r_phase = mhdev(cumsum(y) .* tau0, tau0)

        @test r_freq.tau  == r_phase.tau
        @test r_freq.neff == r_phase.neff
        nan_f = isnan.(r_freq.deviation)
        nan_p = isnan.(r_phase.deviation)
        @test nan_f == nan_p
        @test r_freq.deviation[.!nan_f] ≈ r_phase.deviation[.!nan_p]  rtol=1e-12
    end

end  # @testset "mhdev"

@testset "ldev" begin

    @testset "wraps mhdev correctly" begin
        Random.seed!(300)
        x    = cumsum(randn(512))
        tau0 = 1.0
        rl   = ldev(x, tau0)
        rm   = mhdev(x, tau0)

        expected = rm.tau .* rm.deviation ./ sqrt(10 / 3)
        # Filter NaN positions before comparing (≈ does not handle NaN == NaN)
        mask = .!isnan.(rl.deviation) .& .!isnan.(expected)
        @test all(isnan.(rl.deviation) .== isnan.(expected))   # NaN positions match
        @test rl.deviation[mask] ≈ expected[mask]  rtol=1e-12
    end

    @testset "method field is ldev" begin
        Random.seed!(301)
        x = cumsum(randn(256))
        r = ldev(x, 1.0)
        @test r.method == "ldev"
    end

end  # @testset "ldev"
