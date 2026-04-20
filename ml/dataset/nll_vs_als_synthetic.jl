# Quick probe: do NLL and ALS produce different q labels on synthetic data?
# Runs 10 random synthetic samples, compares the converged q from optimize_nll
# (current label source) vs als_fit (proposed alternative).

using Pkg; Pkg.activate(@__DIR__)
using SigmaTau, Random, Printf

const N = 524_288
const TAU0 = 1.0

h_ranges = Dict{Float64,Tuple{Float64,Float64}}(
    2.0 => (-19.0, -16.0),
    0.0 => (-23.0, -20.0),
   -1.0 => (-28.0, -25.0),
   -2.0 => (-34.0, -28.0),
)

function run_sample(idx::Int)
    rng = Xoshiro(42 + idx)
    h = Dict{Float64,Float64}()
    for α in (2.0, 0.0, -1.0, -2.0)
        lo, hi = h_ranges[α]
        h[α] = 10.0 ^ (lo + (hi - lo) * rand(rng))
    end
    x = generate_composite_noise(h, N, TAU0; seed = 42 + idx + 10_000_000)

    # Reference: true h → q
    q_true = h_to_q(h, TAU0)

    # NLL fit (matches generate_dataset.jl)
    nll_res = optimize_nll(x, TAU0; h_init = h,
                          optimize_qwpm = false,
                          verbose = false, max_iter = 1000)
    q_nll = nll_res.noise

    # ALS fit (same warm start)
    q_als = als_fit(x, TAU0; h_init = h, verbose = false, max_iter = 10)

    return (h_m2 = log10(h[-2.0]),
            q_true = q_true,
            q_nll = q_nll,
            q_als = q_als)
end

function main()
    println("  h_-2 (log10)    |  log10 q_rwfm true  |  NLL     |  ALS     |  Δ")
    println("  " * "─"^85)
    for i in 1:10
        r = run_sample(i)
        @printf("    %6.2f        |     %6.2f          |  %6.2f  |  %6.2f  |  %5.2f\n",
                r.h_m2,
                log10(max(r.q_true.q_rwfm, 1e-99)),
                log10(max(r.q_nll.q_rwfm, 1e-99)),
                log10(max(r.q_als.q_rwfm, 1e-99)),
                log10(max(r.q_nll.q_rwfm, 1e-99)) - log10(max(r.q_als.q_rwfm, 1e-99)))
    end
end
main()
