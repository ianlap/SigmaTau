using Test
using Random
using SigmaTau

@testset "SigmaTau" begin

    @testset "DevParams construction" begin
        p = DevParams("adev", 2, 2, m -> m, 0, 2)
        @test p.name == "adev"
        @test p.min_factor == 2
        @test p.d == 2
        @test p.F_fn(4) == 4          # unmodified: F = m
        @test p.dmin == 0
        @test p.dmax == 2
        p_mod = DevParams("mdev", 3, 2, m -> 1, 0, 2)
        @test p_mod.F_fn(99) == 1     # modified: F = 1 always
    end

    @testset "DeviationResult construction" begin
        L = 4
        r = DeviationResult(
            [1.0, 2.0, 4.0, 8.0],          # tau
            [1e-11, 8e-12, 6e-12, 5e-12],  # deviation
            fill(NaN, L),                   # edf (placeholder)
            fill(NaN, L, 2),                # ci  (placeholder)
            [0, 0, 0, 0],                   # alpha
            [100, 50, 25, 12],              # neff
            1.0,                            # tau0
            200,                            # N
            "adev",
            0.683
        )
        @test length(r.tau) == L
        @test r.tau0 == 1.0
        @test r.method == "adev"
        @test r.confidence == 0.683
        @test all(isnan, r.edf)
        @test all(isnan, r.ci)
    end

    @testset "Engine compiles and runs — adev-like kernel" begin
        # Synthetic white FM noise: 512 phase points
        rng_seed = 42
        Random.seed!(rng_seed)
        N     = 512
        tau0  = 1.0
        # White FM: cumsum of white noise (h₀=1) → ADEV slope ≈ τ^{-1/2}
        x = cumsum(randn(N))

        # Minimal adev kernel: second-difference overlapping variance
        function adev_kernel(x, m, tau0, x_cs)
            L = length(x) - 2m
            L <= 0 && return (NaN, 0)
            d2 = @view(x[1+2m:end]) .- 2 .* @view(x[1+m:end-m]) .+ @view(x[1:L])
            v  = sum(abs2, d2) / (L * 2 * m^2 * tau0^2)
            return (v, L)
        end

        params = DevParams("adev", 2, 2, m -> m, 0, 2)
        result = engine(x, tau0, nothing, adev_kernel, params)

        @test result isa DeviationResult
        @test result.method == "adev"
        @test length(result.tau) > 0
        @test all(result.tau .> 0)
        @test all(d -> isnan(d) || d > 0, result.deviation)
        @test result.tau0 == tau0
        @test result.N == N
        @test length(result.edf) == length(result.tau)

        # Engine now fills CI by default: chi-squared where EDF is finite.
        finite = .!isnan.(result.deviation)
        @test !any(isnan, result.ci[finite, :])
        @test all(result.ci[finite, 1] .<= result.deviation[finite])
        @test all(result.ci[finite, 2] .>= result.deviation[finite])
    end

    @testset "Engine respects explicit m_list" begin
        x      = cumsum(randn(256))
        kernel = (x, m, tau0, x_cs) -> begin
            L = length(x) - 2m
            L <= 0 && return (NaN, 0)
            d2 = @view(x[1+2m:end]) .- 2 .* @view(x[1+m:end-m]) .+ @view(x[1:L])
            (sum(abs2, d2) / (L * 2 * m^2 * tau0^2), L)
        end
        params = DevParams("adev", 2, 2, m -> m, 0, 2)
        ms     = [1, 2, 4, 8]
        result = engine(x, 1.0, ms, kernel, params)

        @test length(result.tau) == length(ms)
        @test result.tau == ms .* 1.0
    end

    @testset "validate_phase_data rejects bad inputs" begin
        @test_throws ArgumentError validate_phase_data([1.0, NaN, 3.0])
        @test_throws ArgumentError validate_phase_data([1.0, Inf])
        @test_throws ArgumentError validate_phase_data([1.0, 2.0])   # too short
    end

    @testset "validate_tau0 rejects bad inputs" begin
        @test_throws ArgumentError validate_tau0(-1.0)
        @test_throws ArgumentError validate_tau0(0.0)
        @test_throws ArgumentError validate_tau0(Inf)
    end

    @testset "unpack_result" begin
        r = DeviationResult([1.0], [1e-11], [10.0], [9e-12 1.1e-11],
                            [0], [50], 1.0, 100, "adev", 0.683)
        @test unpack_result(r, Val(2)) == (r.tau, r.deviation)
        @test unpack_result(r, Val(3)) == (r.tau, r.deviation, r.edf)
        τ, σ, edf, ci, α = unpack_result(r, Val(5))
        @test τ === r.tau
        @test α === r.alpha
    end

    @testset "compute_ci populates EDF" begin
        # build a result with known alpha and N
        L  = 3
        r0 = DeviationResult(
            [1.0, 2.0, 4.0],
            [1e-11, 8e-12, 6e-12],
            fill(NaN, L),
            fill(NaN, L, 2),
            [0, 0, 0],
            [100, 50, 25],
            1.0, 200, "adev", 0.683
        )
        r1 = compute_ci(r0)
        @test !any(isnan, r1.edf)          # EDF should be filled
        @test !any(isnan, r1.ci)           # CI should be filled
        @test all(r1.ci[:, 1] .< r1.deviation)   # lower < deviation
        @test all(r1.ci[:, 2] .> r1.deviation)   # upper > deviation
    end

    @testset "adev public API — returns DeviationResult" begin
        Random.seed!(1234)
        x = cumsum(randn(512))
        r = adev(x, 1.0)
        @test r isa DeviationResult
        @test r.method == "adev"
        @test length(r.tau) > 0
        @test all(r.tau .> 0)
        @test all(d -> isnan(d) || d > 0, r.deviation)

        # CI is populated by the engine (non-NaN wherever deviation is finite).
        finite = .!isnan.(r.deviation)
        @test !any(isnan, r.ci[finite, :])
        @test all(r.ci[finite, 1] .<= r.deviation[finite])
        @test all(r.ci[finite, 2] .>= r.deviation[finite])

        # explicit m_list
        r2 = adev(x, 1.0; m_list=[1, 2, 4, 8])
        @test r2.tau == [1.0, 2.0, 4.0, 8.0]
    end

    @testset "adev noise slopes" begin
        # Fit log-log slope: log(adev) ~ slope * log(tau)
        # via least-squares on non-NaN points.
        function fit_slope(tau, dev)
            mask = .!isnan.(dev) .& (dev .> 0)
            lt = log.(tau[mask])
            ld = log.(dev[mask])
            # slope = cov(lt, ld) / var(lt)
            lt_c = lt .- sum(lt) / length(lt)
            ld_c = ld .- sum(ld) / length(ld)
            return sum(lt_c .* ld_c) / sum(lt_c .^ 2)
        end

        N = 4096
        tau0 = 1.0

        # White PM (α=2): phase is white noise → ADEV slope ≈ τ^{-1}
        Random.seed!(42)
        x_wpm = randn(N)
        r_wpm = adev(x_wpm, tau0)
        slope_wpm = fit_slope(r_wpm.tau, r_wpm.deviation)
        @test isapprox(slope_wpm, -1.0; atol=0.15)

        # White FM (α=0): phase is cumsum of white noise → ADEV slope ≈ τ^{-1/2}
        Random.seed!(43)
        x_wfm = cumsum(randn(N))
        r_wfm = adev(x_wfm, tau0)
        slope_wfm = fit_slope(r_wfm.tau, r_wfm.deviation)
        @test isapprox(slope_wfm, -0.5; atol=0.15)
    end

    @testset "adev data_type=:freq matches manual freq→phase conversion" begin
        Random.seed!(77)
        N    = 512
        tau0 = 1.0
        y    = randn(N)   # fractional-frequency samples

        # Engine path: pass freq data directly
        r_freq = adev(y, tau0; data_type=:freq)

        # Manual path: convert first, then call adev on phase
        x_phase = cumsum(y) .* tau0
        r_phase = adev(x_phase, tau0)

        @test r_freq.tau  == r_phase.tau
        @test r_freq.neff == r_phase.neff
        # NaN positions must match; non-NaN values must agree to floating-point precision
        nan_freq  = isnan.(r_freq.deviation)
        nan_phase = isnan.(r_phase.deviation)
        @test nan_freq == nan_phase
        @test r_freq.deviation[.!nan_freq] ≈ r_phase.deviation[.!nan_phase]  rtol=1e-12
    end

    @testset "adev rejects unknown data_type" begin
        x = cumsum(randn(64))
        @test_throws ArgumentError adev(x, 1.0; data_type=:bad)
    end

    include("test_allan_family.jl")
    include("test_hadamard_family.jl")
    include("test_total_family.jl")
    include("test_filter.jl")
    include("test_kf_diagnostics.jl")
    include("test_noise_id.jl")
    include("test_noise_fit.jl")
    include("test_noise_gen.jl")
    include("test_ml_features.jl")
    include("test_als_fit.jl")
    include("test_matlab_parity.jl")

end  # @testset "SigmaTau"
