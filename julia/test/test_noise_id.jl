# test_noise_id.jl — noise_id long-τ regression
# Before the B1/R(n) m² fix and carry-forward policy, every noise type except
# WHPM spuriously returned α=+2 once N_eff dropped below the lag-1 ACF
# threshold (B1's R(n) discrimination degenerated). Assert that red-spectrum
# inputs (WHFM, RWFM) do not default to +2 at the tail.

@testset "noise_id no spurious α=+2 at long τ" begin
    # Exact closed-form generators (no FFT needed for these cases):
    #   WHFM (α=0):  y = white noise,           x = cumsum(y)
    #   RWFM (α=-2): y = cumsum(white noise),   x = cumsum(y) = double integration
    N  = 2^14                                        # 16384
    ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # tail N_eff: 128 → 16

    # RWFM: tail α must be firmly negative (bug would give α=+2)
    Random.seed!(42)
    x_rwfm = cumsum(cumsum(randn(N)))
    a_rwfm = SigmaTau.noise_id(x_rwfm, ms, "phase", 0, 2)
    @test round(Int, a_rwfm[end]) <= -1

    # WHFM: tail α must be ≤0 (WHFM or FLFM — both defensible at the boundary)
    Random.seed!(43)
    x_wfm  = cumsum(randn(N))
    a_wfm  = SigmaTau.noise_id(x_wfm, ms, "phase", 0, 2)
    @test round(Int, a_wfm[end]) <= 0
end
