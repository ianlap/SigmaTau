# test_kf_diagnostics.jl — Unit tests for kf_residual_diagnostics helper.
#
# Three controlled inputs, one per diagnostic:
#   - white noise                → passes D1, D2 (when supplied), D3
#   - AR(1) ρ=0.5                → fails D1 (autocorrelated)
#   - biased N(1.0, 1.0)         → fails D3 (mean far from zero)
# Plus a small argument-validation testset.

using Test
using Random
using Statistics
using SigmaTau

@testset "kf_residual_diagnostics" begin

    @testset "white noise passes all three diagnostics" begin
        Random.seed!(42)
        N = 1000
        innov     = randn(N)
        resid     = randn(N)
        norm_innov = randn(N)

        d = kf_residual_diagnostics(innov, resid; normalized_innovations=norm_innov)

        @test d.innov_lb_passed                          # D1
        @test !ismissing(d.norm_innov_lb_passed)
        @test d.norm_innov_lb_passed                     # D2
        @test d.resid_bias_passed                        # D3
        @test d.n == N
        @test d.lag == min(20, N ÷ 5)
        @test d.significance == 0.05
    end

    @testset "AR(1) ρ=0.5 fails whiteness" begin
        Random.seed!(43)
        N = 1000
        # AR(1) generator: x[k] = 0.5·x[k-1] + ε[k]
        ar1 = zeros(N)
        ar1[1] = randn()
        for k in 2:N
            ar1[k] = 0.5 * ar1[k-1] + randn()
        end

        d = kf_residual_diagnostics(ar1, randn(N))

        @test !d.innov_lb_passed                         # D1 should fail
        @test d.innov_lb_pvalue < 0.05                   # very small p
    end

    @testset "biased N(1.0, 1.0) fails bias check" begin
        Random.seed!(44)
        N = 1000
        biased = 1.0 .+ randn(N)
        # |mean|·√N / (3σ) ≈ √1000 / 3 ≈ 10.5 → far over 3σ/√N

        d = kf_residual_diagnostics(randn(N), biased)

        @test !d.resid_bias_passed                       # D3 should fail
        @test d.resid_mean > 0.5                         # sanity
        @test d.innov_lb_passed                          # D1 still passes (innovations are white)
    end

    @testset "argument validation" begin
        @test_throws ArgumentError kf_residual_diagnostics([1.0], [1.0])           # too short
        @test_throws ArgumentError kf_residual_diagnostics(randn(10), randn(11))   # len mismatch
        @test_throws ArgumentError kf_residual_diagnostics(randn(10), randn(10); lag=0)
        @test_throws ArgumentError kf_residual_diagnostics(randn(10), randn(10); lag=10)
        @test_throws ArgumentError kf_residual_diagnostics(randn(10), randn(10);
                                                            normalized_innovations=randn(11))
    end

end
