# test_ml_features.jl — feature-extraction pipeline for ML dataset

using Random

@testset "ml_features" begin

    @testset "canonical tau grid shape" begin
        grid = SigmaTau.CANONICAL_TAU_GRID
        mlist = SigmaTau.CANONICAL_M_LIST
        @test length(grid)  == 20
        @test length(mlist) == 20
        @test all(mlist .>= 1)
        @test mlist == sort(unique(mlist))   # strictly increasing, no collisions
        @test mlist[1]  == 1
        @test mlist[end] <= 131072 ÷ 10       # τ_max = N·τ₀/10 for safety factor 10
        @test grid == Float64.(mlist)          # τ = m·τ₀, τ₀=1
    end

    @testset "feature vector length and ordering" begin
        Random.seed!(0)
        x = cumsum(randn(2^15))
        v = compute_feature_vector(x, 1.0)
        @test length(v) == 196   # 80 raw + 76 slopes + 40 ratios
        @test all(isfinite.(v) .| isnan.(v))  # never Inf
    end

    @testset "feature names parallel the vector" begin
        names = SigmaTau.FEATURE_NAMES
        @test length(names) == 196
        # First 80: raw; slopes: 76; ratios: 40
        raw_n   = count(startswith.(names, "raw_"))
        slope_n = count(startswith.(names, "slope_"))
        ratio_n = count(startswith.(names, "ratio_"))
        @test raw_n   == 80
        @test slope_n == 76
        @test ratio_n == 40
    end

end
