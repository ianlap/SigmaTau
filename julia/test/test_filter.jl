# test_filter.jl — Kalman filter tests
# Verifies: (1) filter runs without error on white FM data,
#           (2) residuals have zero mean,
#           (3) covariance converges,
#           (4) kf_predict runs and returns RMS stats,
#           (5) kf_optimize — skipped pending PR #13.

using Statistics

@testset "Kalman filter" begin

    # ── Synthetic data generation ──────────────────────────────────────────────
    Random.seed!(2024)
    N    = 2000
    tau0 = 1.0
    # White FM noise: cumsum of white noise → phase random walk
    x_wfm = cumsum(randn(N))

    # ── Test 1: filter runs without error ─────────────────────────────────────
    @testset "kalman_filter / kf_filter run on white FM" begin
        cfg = KalmanConfig(
            q_wfm  = 1.0,
            q_rwfm = 0.0,
            R      = 1.0,
            g_p    = 0.0, g_i = 0.0, g_d = 0.0,
            nstates = 3,
            tau    = tau0,
            P0     = 1e6,
        )
        result = kalman_filter(x_wfm, cfg)
        @test result isa KalmanResult
        @test length(result.phase_est)   == N
        @test length(result.freq_est)    == N
        @test length(result.drift_est)   == N
        @test length(result.residuals)   == N
        @test length(result.innovations) == N
        @test length(result.P_history)   == N
        @test all(isfinite, result.phase_est)
        @test all(isfinite, result.freq_est)
        @test all(isfinite, result.residuals)

        # kf_filter is an alias — must return the same type
        cfg2   = deepcopy(cfg)
        result2 = kf_filter(x_wfm, cfg2)
        @test result2 isa KalmanResult
    end

    # ── Test 2: residuals have zero mean ──────────────────────────────────────
    @testset "residuals have zero mean (|mean| < 3σ/√N)" begin
        cfg = KalmanConfig(
            q_wfm   = 1.0,
            q_rwfm  = 0.0,
            R       = 1.0,
            g_p = 0.0, g_i = 0.0, g_d = 0.0,
            nstates = 3,
            tau     = tau0,
            P0      = 1e6,
        )
        result = kalman_filter(x_wfm, cfg)
        resid = result.residuals
        μ = mean(resid)
        σ = std(resid)
        # For a correctly tuned filter, residuals should be near zero mean.
        # Tolerance: 3 standard errors (3σ/√N).
        @test abs(μ) < 3 * σ / sqrt(N)
    end

    # ── Test 3: covariance P[1,1] converges (much less than P0 at steady state) ──
    @testset "covariance P[1,1] converges" begin
        P0_scalar = 1e6
        cfg = KalmanConfig(
            q_wfm   = 1.0,
            q_rwfm  = 0.0,
            R       = 1.0,
            g_p = 0.0, g_i = 0.0, g_d = 0.0,
            nstates = 3,
            tau     = tau0,
            P0      = P0_scalar,
        )
        result = kalman_filter(x_wfm, cfg)
        p11_final = result.P_history[end][1, 1]
        # Steady-state P[1,1] must be much smaller than the initial P0
        @test p11_final < P0_scalar / 1000
        # Steady-state P[1,1] must be finite and positive
        @test isfinite(p11_final)
        @test p11_final > 0.0
    end

    # ── Test 4: 2-state and 5-state models run without error ──────────────────
    @testset "2-state model runs" begin
        cfg = KalmanConfig(
            q_wfm = 1.0, q_rwfm = 0.0, R = 1.0,
            g_p = 0.0, g_i = 0.0, g_d = 0.0,
            nstates = 2, tau = tau0, P0 = 1e4,
        )
        result = kalman_filter(x_wfm, cfg)
        @test result isa KalmanResult
        @test all(result.drift_est .== 0.0)   # no drift state
        @test all(isfinite, result.phase_est)
    end

    @testset "5-state model (diurnal) runs" begin
        cfg = KalmanConfig(
            q_wfm = 1.0, q_rwfm = 0.0, R = 1.0,
            q_diurnal = 1e-4,
            g_p = 0.0, g_i = 0.0, g_d = 0.0,
            nstates = 5, tau = tau0, P0 = 1e4,
        )
        result = kalman_filter(x_wfm, cfg)
        @test result isa KalmanResult
        @test all(isfinite, result.phase_est)
    end

    # ── Test 5: PID steering runs without NaN ─────────────────────────────────
    @testset "PID steering produces finite steers" begin
        cfg = KalmanConfig(
            q_wfm = 1.0, q_rwfm = 1e-4, R = 1.0,
            g_p = 0.1, g_i = 0.01, g_d = 0.05,
            nstates = 3, tau = tau0, P0 = 1e4,
        )
        result = kalman_filter(x_wfm, cfg)
        @test all(isfinite, result.steers)
        @test all(isfinite, result.sumsteers)
        @test all(isfinite, result.sum2steers)
    end

    # ── Test 6: kf_predict runs and returns valid RMS stats ───────────────────
    @testset "kf_predict returns valid RMS stats" begin
        N2 = 500
        Random.seed!(42)
        data = cumsum(randn(N2))
        tau  = 1.0

        kf_cfg = KalmanConfig(
            q_wfm = 1.0, q_rwfm = 0.0, R = 1.0,
            g_p = 0.0, g_i = 0.0, g_d = 0.0,
            nstates = 3, tau = tau, P0 = 1e6,
        )
        pred_cfg = PredictConfig(maturity = 100, max_horizon = 50)

        pr = kf_predict(data, tau, kf_cfg, pred_cfg)
        @test pr isa PredictResult
        @test length(pr.horizons)  == length(pr.rms_error) == length(pr.n_samples)
        @test length(pr.horizons) > 0
        @test all(pr.horizons .>= 1)
        @test all(isfinite, pr.rms_error)
        @test all(pr.rms_error .>= 0.0)
        @test all(pr.n_samples .>= 1)
    end

    # ── Test 7: kf_optimize — skipped until optimize.jl lands in PR #13 ────────
    # @testset "kf_optimize finds finite optimal Q" — see PR #13

end  # @testset "Kalman filter"
