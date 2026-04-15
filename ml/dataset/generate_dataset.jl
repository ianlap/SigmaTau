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

const CKPT_EVERY = 500

"""
    generate_dataset(output_path; n_samples=10_000, N=131_072, τ₀=1.0, resume=true)

Main driver. Threads over sample index.  Checkpoints every `CKPT_EVERY`
samples to `<output_path>.checkpoint.npz`; resumes from the highest
completed index when `resume=true`.
"""
function generate_dataset(output_path::String;
                          n_samples::Int = 10_000,
                          N::Int         = 131_072,
                          τ₀::Real       = 1.0,
                          resume::Bool   = true)
    n_features = 196
    X  = Matrix{Float32}(undef, n_samples, n_features)
    y  = Matrix{Float64}(undef, n_samples, 3)
    H  = Matrix{Float64}(undef, n_samples, 5)
    fpm      = Vector{Bool}(undef,    n_samples)
    nll_vals = Vector{Float64}(undef, n_samples)
    conv     = Vector{Bool}(undef,    n_samples)

    # Resume logic
    done = falses(n_samples)
    ckpt = output_path * ".checkpoint.npz"
    if resume && isfile(ckpt)
        data = NPZ.npzread(ckpt)
        nprev = Int(data["n_done"])
        @info "Resuming from checkpoint" nprev
        X[1:nprev, :]      .= data["X"][1:nprev, :]
        y[1:nprev, :]      .= data["y"][1:nprev, :]
        H[1:nprev, :]      .= data["h_coeffs"][1:nprev, :]
        fpm[1:nprev]        .= data["fpm_present"][1:nprev]
        nll_vals[1:nprev]   .= data["nll_values"][1:nprev]
        conv[1:nprev]       .= data["converged"][1:nprev]
        done[1:nprev]       .= true
    end

    pending = findall(.!done)
    t_start = time()
    done_count = Threads.Atomic{Int}(count(done))

    Threads.@threads for i in pending
        r = try
            run_one_sample(i; N=N, τ₀=τ₀, verbose=false)
        catch err
            @warn "sample $i failed" err
            SampleResult(fill(NaN, n_features), fill(NaN, 3), fill(NaN, 5),
                         false, NaN, false)
        end
        X[i, :]  .= Float32.(r.features)
        y[i, :]  .= r.q_labels
        H[i, :]  .= r.h_coeffs
        fpm[i]    = r.fpm_present
        nll_vals[i] = r.nll
        conv[i]     = r.converged
        c = Threads.atomic_add!(done_count, 1) + 1
        if c % CKPT_EVERY == 0
            elapsed = time() - t_start
            @info "checkpoint" done=c of=n_samples elapsed=elapsed
            _write_npz(ckpt, X, y, H, fpm, nll_vals, conv, c)
        end
    end

    # Final write
    _write_npz(output_path, X, y, H, fpm, nll_vals, conv, n_samples; final=true)
    isfile(ckpt) && rm(ckpt)
    @info "done" output_path total=n_samples elapsed=(time() - t_start)
    nothing
end

function _write_npz(path, X, y, H, fpm, nll_vals, conv, n_done; final=false)
    mkpath(dirname(path))
    payload = Dict(
        "X"            => X,
        "y"            => y,
        "h_coeffs"     => H,
        "fpm_present"  => UInt8.(fpm),
        "nll_values"   => nll_vals,
        "converged"    => UInt8.(conv),
        "taus"         => CANONICAL_TAU_GRID,
        "n_done"       => n_done,
    )
    NPZ.npzwrite(path, payload)
    # NPZ.jl does not support string arrays; write feature names as companion file
    names_path = path * ".feature_names.txt"
    open(names_path, "w") do io
        println(io, join(FEATURE_NAMES, "|"))
    end
end

end # module
