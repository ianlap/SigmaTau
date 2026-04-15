# generate_dataset.jl — ML dataset driver (Julia)
#
# Produces ml/data/dataset_v1.npz with 10,000 synthetic samples.  Threaded;
# checkpoints every CKPT_EVERY samples; resumes from checkpoint if present.

module DatasetGen

using Random, Statistics
using NPZ
using SigmaTau

export draw_sample_params, run_one_sample, generate_dataset, SampleParams, SampleResult

# ── Sampling ranges from the spec (§2.3) — log₁₀(h_α) ────────────────────────
const _H_RANGES = Dict(
     2.0 => (-26.5, -23.5),   # WPM
     1.0 => (-25.5, -22.5),   # FPM (only when present)
     0.0 => (-25.5, -22.5),   # WFM
    -1.0 => (-25.5, -22.5),   # FFM
    -2.0 => (-27.5, -24.5),   # RWFM
)
const _FPM_PROBABILITY = 0.30

struct SampleParams
    h_coeffs :: Dict{Float64,Float64}   # α → h_α
    fpm_present :: Bool
end

function Base.:(==)(a::SampleParams, b::SampleParams)
    a.h_coeffs == b.h_coeffs && a.fpm_present == b.fpm_present
end

"""
    draw_sample_params(rng) → SampleParams

Draw one random (h_coeffs, fpm_present) sample.
"""
function draw_sample_params(rng::AbstractRNG)
    h = Dict{Float64,Float64}()
    for α in (2.0, 0.0, -1.0, -2.0)       # always present
        lo, hi = _H_RANGES[α]
        h[α] = 10.0 ^ (lo + (hi - lo) * rand(rng))
    end
    fpm_present = rand(rng) < _FPM_PROBABILITY
    if fpm_present
        lo, hi = _H_RANGES[1.0]
        h[1.0] = 10.0 ^ (lo + (hi - lo) * rand(rng))
    end
    return SampleParams(h, fpm_present)
end

struct SampleResult
    features  :: Vector{Float64}       # 196
    q_labels  :: Vector{Float64}       # log10(q_wpm, q_wfm, q_rwfm)
    h_coeffs  :: Vector{Float64}       # log10(h₊₂, h₊₁, h₀, h₋₁, h₋₂) — NaN for absent
    fpm_present :: Bool
    nll       :: Float64
    converged :: Bool
end

"""
    run_one_sample(idx; N, τ₀, verbose=false) → SampleResult

Generate a single dataset sample with deterministic seed `42 + idx`.
"""
function run_one_sample(idx::Integer;
                        N::Int   = 131_072,
                        τ₀::Real = 1.0,
                        verbose::Bool = false)
    rng = Xoshiro(42 + idx)
    p   = draw_sample_params(rng)

    # Phase time series
    x = generate_composite_noise(p.h_coeffs, N, τ₀; seed = 42 + idx + 10_000_000)

    # Features
    v = compute_feature_vector(x, τ₀)

    # NLL labels — use h-warm start for fast convergence
    opt = optimize_kf_nll(x, τ₀; h_init = p.h_coeffs, verbose = verbose)

    # Provenance h-vector in canonical α order (+2, +1, 0, -1, -2)
    h_vec = fill(NaN, 5)
    for (j, α) in enumerate((2.0, 1.0, 0.0, -1.0, -2.0))
        if haskey(p.h_coeffs, α)
            h_vec[j] = log10(p.h_coeffs[α])
        end
    end

    SampleResult(
        v,
        [log10(opt.q_wpm), log10(opt.q_wfm), log10(opt.q_rwfm)],
        h_vec,
        p.fpm_present,
        opt.nll,
        opt.converged,
    )
end

end # module
