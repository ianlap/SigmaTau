# test_total_family.jl — Tests for totdev, mtotdev, htotdev, mhtotdev wrappers

@testset "totdev, mtotdev, htotdev, mhtotdev" begin

    N    = 4096
    tau0 = 1.0

    @testset "totdev basic API" begin
        Random.seed!(201)
        x = cumsum(randn(N))
        r = totdev(x, tau0)
        @test r isa DeviationResult
        @test r.method == "totdev"
        @test length(r.tau) > 0
        @test all(r.tau .> 0)
        @test all(d -> isnan(d) || d > 0, r.deviation)
    end

    @testset "totdev White FM slope ≈ -0.5" begin
        # White FM (α=0): phase = cumsum(randn) → TOTDEV slope ≈ τ^{-1/2}
        Random.seed!(202)
        x = cumsum(randn(N))
        r = totdev(x, tau0)
        slope = fit_slope(r.tau, r.deviation)
        @test isapprox(slope, -0.5; atol=0.25)
    end

    @testset "totdev data_type=:freq passthrough" begin
        Random.seed!(203)
        y = randn(N)   # fractional-frequency
        r = totdev(y, tau0; data_type=:freq)
        @test r isa DeviationResult
        @test r.method == "totdev"
        @test length(r.tau) > 0
    end

    @testset "mtotdev basic API" begin
        Random.seed!(204)
        x = cumsum(randn(N))
        r = mtotdev(x, tau0)
        @test r isa DeviationResult
        @test r.method == "mtotdev"
        @test length(r.tau) > 0
        @test all(r.tau .> 0)
        @test all(d -> isnan(d) || d > 0, r.deviation)
    end

    @testset "mtotdev White FM slope ≈ -0.5" begin
        # White FM (α=0): phase = cumsum(randn) → MTOTDEV slope ≈ τ^{-1/2}
        # (same as ADEV/TOTDEV; modified variant does not change White FM slope)
        Random.seed!(205)
        x = cumsum(randn(N))
        r = mtotdev(x, tau0)
        slope = fit_slope(r.tau, r.deviation)
        @test isapprox(slope, -0.5; atol=0.25)
    end

    @testset "htotdev basic API" begin
        Random.seed!(206)
        x = cumsum(randn(N))
        r = htotdev(x, tau0)
        @test r isa DeviationResult
        @test r.method == "htotdev"
        @test length(r.tau) > 0
        @test all(r.tau .> 0)
        @test all(d -> isnan(d) || d > 0, r.deviation)
    end

    @testset "htotdev m=1 uses hdev" begin
        # CLAUDE.md critical rule: htotdev at m=1 must equal hdev at m=1
        Random.seed!(207)
        x = cumsum(randn(N))
        rh  = hdev(x, tau0; m_list=[1])
        rht = htotdev(x, tau0; m_list=[1])
        @test !isempty(rh.deviation)
        @test !isempty(rht.deviation)
        # Allow for small bias correction applied by engine (derived from method name)
        @test isapprox(rht.deviation[1], rh.deviation[1]; rtol=0.01)
    end

    @testset "htotdev White FM slope ≈ -0.5" begin
        # White FM: phase = cumsum(randn) → HTOTDEV slope ≈ τ^{-1/2}
        Random.seed!(208)
        x = cumsum(randn(N))
        r = htotdev(x, tau0)
        slope = fit_slope(r.tau, r.deviation)
        @test isapprox(slope, -0.5; atol=0.25)
    end

    @testset "mhtotdev basic API" begin
        Random.seed!(209)
        x = cumsum(randn(N))
        r = mhtotdev(x, tau0)
        @test r isa DeviationResult
        @test r.method == "mhtotdev"
        @test length(r.tau) > 0
        @test all(r.tau .> 0)
        @test all(d -> isnan(d) || d > 0, r.deviation)
    end

    @testset "mhtotdev White FM slope ≈ -0.5" begin
        # White FM (α=0): phase = cumsum(randn) → MHTOTDEV slope ≈ τ^{-1/2}
        # (same as HDEV/HTOTDEV; modified variant does not change White FM slope)
        Random.seed!(210)
        x = cumsum(randn(N))
        r = mhtotdev(x, tau0)
        slope = fit_slope(r.tau, r.deviation)
        @test isapprox(slope, -0.5; atol=0.25)
    end

    @testset "all four accept data_type=:freq" begin
        Random.seed!(211)
        y = randn(N)   # fractional-frequency samples
        @test_nowarn totdev(y, tau0; data_type=:freq)
        @test_nowarn mtotdev(y, tau0; data_type=:freq)
        @test_nowarn htotdev(y, tau0; data_type=:freq)
        @test_nowarn mhtotdev(y, tau0; data_type=:freq)
    end

end  # @testset "totdev, mtotdev, htotdev, mhtotdev"
