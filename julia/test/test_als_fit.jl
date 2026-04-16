# test_als_fit.jl — Test Autocovariance Least Squares (ALS) Estimator

using Test
using Random

@testset "ALS Tuning Estimator" begin


    @testset "ALS Synthetic Recovery" begin
        # If we simulate standard noise and run kalman_filter, the innovations
        # should have an empirical autocovariance that lets us recover the noise.
        # Given finite data, it won't be perfect, but it should be close.
        # Better: run with identical config and check code functionality.
        Random.seed!(42)
        N = 1000
        tau0 = 1.0
        
        noise = ClockNoiseParams(q_wpm=1e-8, q_wfm=1e-10, q_rwfm=0.0)
        
        # Generate some synthetic phase data with this noise (white + random walk)
        wpm = sqrt(noise.q_wpm) .* randn(N)
        wfm = sqrt(noise.q_wfm) .* randn(N)
        rw = cumsum(wfm)
        phase = wpm .+ rw
        
        start_noise = ClockNoiseParams(q_wpm=1e-9, q_wfm=1e-11, q_rwfm=1e-12)
        
        opt_noise = als_fit(phase, tau0; noise_init=start_noise, 
                            max_iter=3, burn_in=20, lags=15, 
                            optimize_qwpm=true, optimize_irwfm=false, verbose=false)
        
        @test opt_noise isa ClockNoiseParams
        @test opt_noise.q_wpm > 0.0
        @test opt_noise.q_wfm >= 0.0
        
        # WPM is highly observable via C_0 and C_1. It should recover order-of-magnitude.
        @test 1e-10 < opt_noise.q_wpm < 1e-7
    end

    @testset "ALS with irwfm enabled" begin
        phase = cumsum(cumsum(randn(200))) # strong drift properties
        start_noise = ClockNoiseParams(q_wpm=1e-2, q_wfm=1e-2, q_rwfm=1e-2, q_irwfm=1e-2)
        
        # Just ensure it doesn't crash when optimizing IRWFM
        opt_noise = als_fit(phase, 1.0; noise_init=start_noise, 
                            max_iter=1, burn_in=10, lags=5, 
                            optimize_qwpm=true, optimize_irwfm=true, verbose=false)
        
        @test opt_noise.q_irwfm >= 0.0
    end
end
