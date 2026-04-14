# generate.jl — reproducible pure-α phase series for noise-ID validation
#
# Produces five phase records, one per α ∈ {+2, +1, 0, −1, −2}, using the
# Kasdin/Walter FFT synthesis (matches matlab/+sigmatau/+noise/generate.m).
# Each record is 2^16 samples at τ0 = 1 s. Output is a space-separated
# two-column text file: MJD, phase(sec).
#
# Reference:
#   N. J. Kasdin & T. Walter, "Discrete simulation of power law noise",
#   Proc. IEEE Frequency Control Symposium, pp. 274-283, 1992.
#
# Run from the repo root:
#   julia --project=julia examples/noise_id_validation/generate.jl

using SigmaTau
using Random
using FFTW
using Printf
using Statistics

const OUTDIR   = @__DIR__
const N        = 2^16
const TAU0     = 1.0
const MJD_START = 60790.0
const DT_DAYS  = TAU0 / 86400.0

# (α, filename, seed) — seeds are arbitrary but fixed for reproducibility.
const CASES = (
    ( 2,  "alpha+2_whpm.csv", 2001),
    ( 1,  "alpha+1_flpm.csv", 2002),
    ( 0,  "alpha+0_whfm.csv", 2003),
    (-1,  "alpha-1_flfm.csv", 2004),
    (-2,  "alpha-2_rwfm.csv", 2005),
)

# Kasdin FFT synthesis: shape a white Gaussian spectrum with |·|^(α/2),
# IFFT to frequency data, normalise to unit std, integrate to phase.
# Mirrors matlab/+sigmatau/+noise/generate.m line-for-line so cross-language
# spot checks reproduce the same statistics (though not the same realisation,
# since RNG streams differ between MATLAB and Julia).
function kasdin_phase(alpha::Integer, N::Integer, tau0::Real; rng::AbstractRNG)
    iseven(N) || throw(ArgumentError("N must be even for FFT symmetry"))
    f   = collect(1:(N ÷ 2))
    S_f = Float64.(f) .^ (alpha / 2)
    phase_rand = 2π .* rand(rng, (N ÷ 2) - 1)

    half = zeros(ComplexF64, N ÷ 2 + 1)
    half[1] = 0                                                  # DC
    half[2:(N ÷ 2)] .= S_f[1:end-1] .* cis.(phase_rand)           # positive freqs
    half[N ÷ 2 + 1] = 0                                           # Nyquist → real IFFT

    full = vcat(half, conj.(reverse(half[2:(N ÷ 2)])))
    x_freq = real.(ifft(full))
    x_freq .-= mean(x_freq)
    x_freq ./= std(x_freq)
    return cumsum(x_freq) .* tau0
end

for (alpha, fname, seed) in CASES
    x = kasdin_phase(alpha, N, TAU0; rng = Xoshiro(seed))
    open(joinpath(OUTDIR, fname), "w") do io
        for i in 1:N
            @printf(io, "%.11f %.9e\n", MJD_START + (i - 1) * DT_DAYS, x[i])
        end
    end
    # Quick sanity: report α recovered by noise_id on a mid-τ slice.
    α_id = noise_id(x, [1, 2, 4, 8, 16, 32, 64, 128, 256], "phase")
    @printf("α=%+d → %s  (N=%d)  noise_id α̂ at m=64: %+.2f\n",
            alpha, fname, N, α_id[7])
end

println("wrote $(length(CASES)) files to $OUTDIR")
