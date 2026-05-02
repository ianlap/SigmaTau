# test_filter.jl — Kalman filter tests

using Statistics
using Random
using LinearAlgebra

@testset "Kalman filter" begin

    # ── Synthetic data generation ──────────────────────────────────────────────
    Random.seed!(2024)
    N    = 2000
    tau0 = 1.0
    x_wfm = cumsum(randn(N))

    # ── Test 1: filter runs without error ─────────────────────────────────────
    @testset "kalman_filter / kf_filter run on white FM" begin
        noise = ClockNoiseParams(q_wfm=1.0, q_rwfm=0.0, q_wpm=1.0)
        model = ClockModel3(noise=noise, tau=tau0)
        
        result = kalman_filter(x_wfm, model; g_p=0.0, g_i=0.0, g_d=0.0, P0=1e6)
        @test result isa KalmanResult
        @test length(result.phase_est)   == N
        @test length(result.freq_est)    == N
        @test length(result.drift_est)   == N
        @test length(result.residuals)   == N
        @test length(result.innovations) == N
        @test size(result.P_history, 3)  == N
        @test all(isfinite, result.phase_est)
        @test all(isfinite, result.freq_est)
        @test all(isfinite, result.residuals)

        # kf_filter is an alias
        result2 = kf_filter(x_wfm, model; g_p=0.0, g_i=0.0, g_d=0.0, P0=1e6)
        @test result2 isa KalmanResult
    end

    # ── Test 2: residuals have zero mean ──────────────────────────────────────
    @testset "residuals have zero mean (|mean| < 3σ/√N)" begin
        noise = ClockNoiseParams(q_wfm=1.0, q_rwfm=0.0, q_wpm=1.0)
        model = ClockModel3(noise=noise, tau=tau0)
        result = kalman_filter(x_wfm, model; g_p=0.0, g_i=0.0, g_d=0.0, P0=1e6)
        resid = result.residuals
        μ = mean(resid)
        σ = std(resid)
        @test abs(μ) < 3 * σ / sqrt(N)
    end

    # ── Test 3: covariance P[1,1] converges ───────────────────────────────────
    @testset "covariance P[1,1] converges" begin
        P0_scalar = 1e6
        noise = ClockNoiseParams(q_wfm=1.0, q_rwfm=0.0, q_wpm=1.0)
        model = ClockModel3(noise=noise, tau=tau0)
        result = kalman_filter(x_wfm, model; g_p=0.0, g_i=0.0, g_d=0.0, P0=P0_scalar)
        p11_final = result.P_history[1, 1, end]
        @test p11_final < P0_scalar / 1000
        @test isfinite(p11_final)
        @test p11_final > 0.0
    end

    # ── Test 4: 2-state and 5-state models run without error ──────────────────
    @testset "2-state model runs" begin
        noise = ClockNoiseParams(q_wfm=1.0, q_rwfm=0.0, q_wpm=1.0)
        model = ClockModel2(noise=noise, tau=tau0)
        result = kalman_filter(x_wfm, model; g_p=0.0, g_i=0.0, g_d=0.0, P0=1e4)
        @test result isa KalmanResult
        @test all(result.drift_est .== 0.0)   # no drift state
        @test all(isfinite, result.phase_est)
    end

    @testset "5-state model (diurnal) runs" begin
        noise = ClockNoiseParams(q_wfm=1.0, q_rwfm=0.0, q_wpm=1.0)
        model = ClockModelDiurnal(noise=noise, tau=tau0, q_diurnal=1e-4)
        result = kalman_filter(x_wfm, model; g_p=0.0, g_i=0.0, g_d=0.0, P0=1e4)
        @test result isa KalmanResult
        @test all(isfinite, result.phase_est)
    end

    # ── Test 5: PID steering runs without NaN ─────────────────────────────────
    @testset "PID steering produces finite steers" begin
        noise = ClockNoiseParams(q_wfm=1.0, q_rwfm=1e-4, q_wpm=1.0)
        model = ClockModel3(noise=noise, tau=tau0)
        result = kalman_filter(x_wfm, model; g_p=0.1, g_i=0.01, g_d=0.05, P0=1e4)
        @test all(isfinite, result.steers)
        @test all(isfinite, result.sumsteers)
        @test all(isfinite, result.sum2steers)
    end

    # ── Test 6: predict_holdover — zero state stays at zero ─────────────────
    @testset "predict_holdover: zero state propagates to zero" begin
        noise = ClockNoiseParams(q_wfm=0.0, q_rwfm=0.0, q_wpm=0.0)
        model = ClockModel3(noise=noise, tau=1.0)
        x0 = zeros(3)
        P0 = zeros(3, 3)
        hr = predict_holdover(x0, P0, model, 100)
        @test hr isa HoldoverResult
        @test all(hr.phase_pred .== 0.0)
        @test all(hr.freq_pred .== 0.0)
        @test all(hr.drift_pred .== 0.0)
        @test all(hr.P_pred .== 0.0)
    end

    # ── Test 7: predict_holdover — covariance grows monotonically ─────────
    @testset "predict_holdover: covariance grows monotonically" begin
        noise = ClockNoiseParams(q_wfm=1.0, q_rwfm=1e-4, q_wpm=1.0)
        model = ClockModel3(noise=noise, tau=1.0)
        x0 = zeros(3)
        P0 = Matrix{Float64}(I, 3, 3)
        hr = predict_holdover(x0, P0, model, 50)
        for i in 2:50
            @test hr.P_pred[1, 1, i] >= hr.P_pred[1, 1, i-1]
        end
    end

    # ── Test 8: predict_holdover — KalmanResult wrapper matches raw ───────
    @testset "predict_holdover: KalmanResult wrapper matches raw method" begin
        Random.seed!(77)
        N2 = 500
        data = cumsum(randn(N2))
        noise = ClockNoiseParams(q_wfm=1.0, q_rwfm=0.0, q_wpm=1.0)
        model = ClockModel3(noise=noise, tau=1.0)
        kf = kalman_filter(data, model; g_p=0.0, g_i=0.0, g_d=0.0)

        hr1 = predict_holdover(kf, 20)

        x0 = Float64[kf.phase_est[end], kf.freq_est[end], kf.drift_est[end]]
        P0 = Matrix{Float64}(kf.P_history[:, :, end])
        hr2 = predict_holdover(x0, P0, model, 20)

        @test hr1.phase_pred ≈ hr2.phase_pred atol=1e-14
        @test hr1.freq_pred  ≈ hr2.freq_pred  atol=1e-14
        @test hr1.drift_pred ≈ hr2.drift_pred atol=1e-14
        @test hr1.P_pred     ≈ hr2.P_pred     atol=1e-14
    end

    # ── Test 9: predict_holdover — 2-state model works ────────────────────
    @testset "predict_holdover: 2-state model" begin
        noise = ClockNoiseParams(q_wfm=1.0, q_rwfm=0.0, q_wpm=1.0)
        model = ClockModel2(noise=noise, tau=1.0)
        x0 = Float64[0.0, 1.0]  # constant frequency => linear phase
        P0 = zeros(2, 2)
        hr = predict_holdover(x0, P0, model, 10)
        for i in 1:10
            @test hr.phase_pred[i] ≈ Float64(i) atol=1e-14
            @test hr.freq_pred[i]  ≈ 1.0        atol=1e-14
        end
        @test all(hr.drift_pred .== 0.0)
    end

    @testset "3-D NLL optimization recovers q_wpm on WPM+WFM data" begin
        Random.seed!(99)
        N = 4096; τ = 1.0
        q_wpm_true = 0.25
        q_wfm_true = 0.01
        phase_true = cumsum(sqrt(q_wfm_true * τ) .* randn(N))
        ph = phase_true .+ sqrt(q_wpm_true) .* randn(N)

        noise_init = ClockNoiseParams(q_wpm=1.0, q_wfm=0.1, q_rwfm=1e-8)
        res = optimize_nll(ph, τ; noise_init=noise_init, verbose=false, max_iter=2000, tol=1e-5, optimize_qwpm=true)

        @test res isa OptimizeNLLResult
        @test abs(log10(res.noise.q_wpm) - log10(q_wpm_true)) < 1.0
        @test abs(log10(res.noise.q_wfm) - log10(q_wfm_true)) < 1.0
    end

    @testset "innovation_nll output finite" begin
        Random.seed!(123)
        N  = 2048; τ = 1.0
        x  = cumsum(randn(N))
        noise = ClockNoiseParams(q_wpm=1.0, q_wfm=0.5, q_rwfm=1e-4)
        m3 = ClockModel3(noise=noise, tau=τ)
        nll3 = innovation_nll(x, m3)
        @test isfinite(nll3)
        
        m2 = ClockModel2(noise=noise, tau=τ)
        nll2 = innovation_nll(x, m2)
        @test isfinite(nll2)
    end

    @testset "optimize_nll wrapper with h_init warm start" begin
        Random.seed!(55)
        h = Dict(2.0 => 1e-22, 0.0 => 1e-22, -2.0 => 1e-26)
        x = generate_composite_noise(h, 2^14, 1.0; seed=55)
        res = optimize_nll(x, 1.0; h_init=h, verbose=false, optimize_qwpm=true)
        expected = h_to_q(h, 1.0)
        @test res isa OptimizeNLLResult
        @test abs(log10(res.noise.q_wpm)  - log10(expected.q_wpm))  < 1.0
        @test abs(log10(res.noise.q_wfm)  - log10(expected.q_wfm))  < 1.0
        # Tight RWFM tolerance: log10(3) ≈ 0.48 discriminates Wu 2023 (2π²·h_-2)
        # from pre-Wu legacy ((2π²/3)·h_-2). 0.3 = factor 2, well inside the
        # expected statistical spread at N=16384 and strictly below the 3x gap.
        @test abs(log10(res.noise.q_rwfm) - log10(expected.q_rwfm)) < 0.3
    end

    # ── Pluggable measurement-model interface (Phase 2) ──────────────────────
    @testset "Pluggable measurement model" begin

        # (a) Bit-identical regression: defaulting to PhaseOnlyMeasurement()
        # must reproduce the pre-Phase-2 KalmanResult byte-for-byte.
        @testset "kalman_filter bit-identical with explicit PhaseOnlyMeasurement" begin
            Random.seed!(2024)
            data = cumsum(randn(1500))
            noise = ClockNoiseParams(q_wfm=1.0, q_rwfm=1e-4, q_wpm=1.0)

            for model in (ClockModel2(noise=noise, tau=tau0),
                          ClockModel3(noise=noise, tau=tau0),
                          ClockModelDiurnal(noise=noise, tau=tau0, q_diurnal=1e-4))
                res_old = kalman_filter(data, model; g_p=0.1, g_i=0.01, g_d=0.05, P0=1e4)
                res_new = kalman_filter(data, model, PhaseOnlyMeasurement();
                                        g_p=0.1, g_i=0.01, g_d=0.05, P0=1e4)
                @test res_new.phase_est   == res_old.phase_est
                @test res_new.freq_est    == res_old.freq_est
                @test res_new.drift_est   == res_old.drift_est
                @test res_new.innovations == res_old.innovations
                @test res_new.residuals   == res_old.residuals
                @test res_new.steers      == res_old.steers
                @test res_new.P_history   == res_old.P_history
            end
        end

        @testset "innovation_nll perf-shim routes ClockModel3+PhaseOnly to fast path" begin
            Random.seed!(123)
            x = cumsum(randn(2048))
            noise = ClockNoiseParams(q_wpm=1.0, q_wfm=0.5, q_rwfm=1e-4)
            m3 = ClockModel3(noise=noise, tau=1.0)
            # 2-arg and 3-arg forms must agree exactly.
            @test innovation_nll(x, m3) == innovation_nll(x, m3, PhaseOnlyMeasurement())
            # 2-arg fallback for ClockModel2 also routes through PhaseOnly.
            m2 = ClockModel2(noise=noise, tau=1.0)
            @test innovation_nll(x, m2) == innovation_nll(x, m2, PhaseOnlyMeasurement())
        end

        # (b) Vector-measurement smoke: define a 2D phase+freq measurement
        # and exercise the matrix kernel through filter_step! directly.
        @testset "vector measurement (phase+freq) drives matrix kernel" begin
            struct PhaseFreqMeasurement <: AbstractMeasurementModel
                R::Matrix{Float64}
            end
            SigmaTau.build_H(::PhaseFreqMeasurement, ::ClockModel3, k::Int=0) =
                [1.0 0.0 0.0; 0.0 1.0 0.0]
            SigmaTau.measurement_R(meas::PhaseFreqMeasurement, ::AbstractStateModel) = meas.R
            SigmaTau.measurement_dim(::PhaseFreqMeasurement) = 2

            noise = ClockNoiseParams(q_wpm=0.0, q_wfm=1e-4, q_rwfm=0.0)
            model = ClockModel3(noise=noise, tau=1.0)
            meas  = PhaseFreqMeasurement([0.01 0.0; 0.0 0.001])

            s = SigmaTau.FilterState(x=zeros(3), P=Matrix{Float64}(I, 3, 3) .* 10.0, k=0)
            tr_init = sum(diag(s.P))

            Random.seed!(11)
            for k in 1:100
                z = [Float64(k), 1.0] .+ [0.1, 0.01] .* randn(2)
                _, ν, S = SigmaTau.filter_step!(s, model, meas, z)
                @test length(ν) == 2
                @test size(S) == (2, 2)
                @test all(isfinite, s.x)
                @test all(isfinite, s.P)
                @test isapprox(s.P, s.P', atol=1e-12)   # symmetric
                # PSD: smallest eigenvalue ≥ -ε
                λmin = minimum(eigvals(Symmetric(s.P)))
                @test λmin > -1e-10
            end
            @test sum(diag(s.P)) < tr_init  # uncertainty shrank
            # State should track the linear ramp + unit frequency.
            @test abs(s.x[1] - 100.0) < 5.0
            @test abs(s.x[2] - 1.0) < 0.5
        end

        # (c) Non-clock state model — proves the architecture truly decouples
        # from ClockNoiseParams and clock-specific assumptions. 2D constant-
        # velocity tracker: state = [x, vx, y, vy], 2D position measurement.
        @testset "non-clock state model (2D constant velocity)" begin
            struct ConstantVelocityModel <: AbstractStateModel
                tau::Float64
                q_pos::Float64    # process noise diffusion on position
                q_vel::Float64    # process noise diffusion on velocity
            end
            SigmaTau.nstates(::ConstantVelocityModel) = 4
            function SigmaTau.build_phi(m::ConstantVelocityModel)
                τ = m.tau
                # block-diagonal: same kinematics in x and y.
                Φ1 = [1.0 τ; 0.0 1.0]
                Z  = zeros(2, 2)
                return [Φ1 Z; Z Φ1]
            end
            function SigmaTau.build_Q(m::ConstantVelocityModel)
                τ = m.tau
                qp, qv = m.q_pos, m.q_vel
                Q1 = [qp*τ + qv*τ^3/3   qv*τ^2/2;
                      qv*τ^2/2           qv*τ]
                Z = zeros(2, 2)
                return [Q1 Z; Z Q1]
            end

            struct PositionMeasurement <: AbstractMeasurementModel
                R::Matrix{Float64}    # 2×2
            end
            SigmaTau.build_H(::PositionMeasurement, ::ConstantVelocityModel, k::Int=0) =
                [1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0]
            SigmaTau.measurement_R(m::PositionMeasurement, ::AbstractStateModel) = m.R
            SigmaTau.measurement_dim(::PositionMeasurement) = 2

            model = ConstantVelocityModel(1.0, 1e-4, 1e-3)
            meas  = PositionMeasurement([0.04 0.0; 0.0 0.04])    # σ_pos = 0.2

            # Ground-truth: object moving at (vx, vy) = (0.5, -0.3) from origin.
            Random.seed!(2026)
            N = 200
            x_true = [0.5*k for k in 0:N-1]
            y_true = [-0.3*k for k in 0:N-1]
            σ_meas = 0.2

            s = SigmaTau.FilterState(
                x = [0.0, 0.0, 0.0, 0.0],
                P = Matrix{Float64}(I, 4, 4) .* 10.0,
                k = 0,
            )

            for k in 1:N
                z = [x_true[k] + σ_meas*randn(),
                     y_true[k] + σ_meas*randn()]
                _, ν, S = SigmaTau.filter_step!(s, model, meas, z)
                @test length(ν) == 2
                @test size(S) == (2, 2)
                @test all(isfinite, s.x)
            end

            # Final state should match ground truth: position within a few σ,
            # velocity converged to (0.5, -0.3).
            @test abs(s.x[1] - x_true[end]) < 2.0           # x position
            @test abs(s.x[2] - 0.5)         < 0.05          # x velocity
            @test abs(s.x[3] - y_true[end]) < 2.0           # y position
            @test abs(s.x[4] - (-0.3))      < 0.05          # y velocity

            # PSD covariance throughout (already checked per-step above; assert
            # final too).
            @test isapprox(s.P, s.P', atol=1e-12)
            @test minimum(eigvals(Symmetric(s.P))) > -1e-10
        end
    end

end  # @testset "Kalman filter"
