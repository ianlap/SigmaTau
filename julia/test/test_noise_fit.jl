# test_noise_fit.jl — exercise SigmaTau.mhdev_fit on synthetic noise

using Test
using Random
using Statistics
using SigmaTau

@testset "mhdev_fit" begin

    @testset "recovers q_wpm from pure WPM" begin
        # Unit-variance white phase modulation.  For WPM the MHDEV theory gives
        # σ²_MHDEV(τ) = (10/3)·q·τ^-3 with q ≈ σ_x² = 1.  Use a long record so
        # the fit's sampling error is tiny.
        N = 32_768
        x = randn(Xoshiro(42), N)
        r = mhdev(x, 1.0)
        fit = mhdev_fit(r.tau, r.deviation, [(:wpm, 1:6)])
        @test fit.q_wpm ≈ 1.0 rtol=0.05
        @test fit.q_wfm  == 0.0
        @test fit.q_rwfm == 0.0
        # Residual should be ≪ original variance across the fitted indices.
        @test all(fit.var_residual[1:6] .< r.deviation[1:6] .^ 2 .* 0.1)
    end

    @testset "successive-subtraction: WPM + WFM mixture" begin
        # σ²(τ) = (10/3)·q_wpm·τ^-3  +  (7/16)·q_wfm·τ^-1.
        # The log-space weighted mean is biased when regions aren't
        # noise-type-pure — the WPM fit over idx 1:3 picks up a small WFM
        # contribution, and vice versa. Verify we're within a few percent
        # rather than exactly equal.
        q_wpm, q_wfm = 4.0, 0.03
        tau = Float64.([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        var_exact = (10/3) .* q_wpm .* tau .^ (-3) .+
                    (7/16) .* q_wfm .* tau .^ (-1)
        sigma = sqrt.(var_exact)

        fit = mhdev_fit(tau, sigma, [(:wpm, 1:3), (:wfm, 6:10)])
        @test fit.q_wpm ≈ q_wpm rtol=0.02    # ~0.7% cross-talk
        @test fit.q_wfm ≈ q_wfm rtol=0.02
    end

    @testset "regions field exposes per-fit diagnostics" begin
        tau   = Float64.(1:8)
        sigma = sqrt.((10/3) .* 2.0 .* tau .^ (-3))   # pure WPM with q=2
        fit   = mhdev_fit(tau, sigma, [(:wpm, 1:5)])
        @test length(fit.regions) == 1
        reg = fit.regions[1]
        @test reg.noise_type == :wpm
        @test reg.indices == 1:5
        @test reg.value ≈ 2.0 rtol=1e-10
        @test !reg.skipped
    end

    @testset "flicker branch runs and subtracts in σ-space" begin
        # Pure FFM (σ = sig0 at all τ). Simplest test that exercises the
        # flicker code path without cross-talk with WPM.
        sig0_ffm = 0.5
        tau   = Float64.([1, 2, 4, 8, 16, 32, 64, 128])
        sigma = fill(sig0_ffm, length(tau))
        fit   = mhdev_fit(tau, sigma, [(:ffm, 1:8)])
        @test fit.sig0_ffm ≈ sig0_ffm rtol=1e-10
        @test fit.q_wpm    == 0.0
    end

    @testset "CI weighting applies when ci supplied" begin
        # With equal weights the equal-weight and :symmetric results agree when
        # the CI is symmetric around sigma.
        tau   = Float64.(1:6)
        sigma = sqrt.((10/3) .* 2.0 .* tau .^ (-3))
        ci    = hcat(sigma .* 0.9, sigma .* 1.1)
        fit1 = mhdev_fit(tau, sigma, [(:wpm, 1:5)])
        fit2 = mhdev_fit(tau, sigma, [(:wpm, 1:5)];
                         ci = ci, weight_method = :symmetric)
        @test fit1.q_wpm ≈ fit2.q_wpm rtol=1e-6
    end

    @testset "unknown noise type errors" begin
        tau = Float64.(1:4); sigma = fill(1.0, 4)
        @test_throws ArgumentError mhdev_fit(tau, sigma, [(:bogus, 1:3)])
    end

    @testset "regression: non-contiguous index vectors [1,3,7]" begin
        # _to_indices fix: ensure we can fit non-contiguous tau indices.
        # This occurs when we have a sparse tau list but want to fit regions
        # that are not contiguous in the vector (e.g., indices 1, 3, 7).
        tau = Float64.([1, 2, 4, 8, 16, 32, 64, 128])
        sigma = sqrt.((10/3) .* 1.0 .* tau .^ (-3)) # pure WPM, q=1
        fit = mhdev_fit(tau, sigma, [(:wpm, [1, 3, 7])])
        @test fit.q_wpm ≈ 1.0 rtol=1e-10
        @test fit.regions[1].indices == [1, 3, 7]
    end

end  # @testset "mhdev_fit"
