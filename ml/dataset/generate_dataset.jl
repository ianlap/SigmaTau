# generate_dataset.jl — ML dataset driver (Julia)
#
# Produces ml/data/dataset_v1.h5 with 10,000 synthetic samples.  Threaded;
# checkpoints every CKPT_EVERY samples; resumes from checkpoint if present.

module DatasetGen

using Random, Statistics
using HDF5
using SigmaTau

export draw_sample_params, run_one_sample, generate_dataset, SampleParams, SampleResult

# ── h_α sampling ranges (log₁₀) — anchored to GMR6000 Rb (HSO-3 class) ───────
#
# Anchored to GMR6000 NLL fits (see ml/data/proposed_h_ranges.md):
#   h_+2 ≈ -17.56 (WPM), h_0 ≈ -21.32 (WFM), h_-2 near zero
#   flicker floor ≈ 3e-13 → h_-1 ≈ -25 (inferred)
# Ranges tightened to cover Rb / HSO / OCXO class (flicker floor 1e-14 to 1e-11);
# do NOT extend to TCXOs (they'd dominate with WPM levels ≥ 1e-8 at τ=1).
const _H_RANGES = Dict(
     2.0 => (-20.0, -15.0),   # WPM  — σ_y(1) ∈ [1.9e-11, 6e-9]; Rb anchor -17.56
     1.0 => (-26.0, -21.0),   # FPM  — no direct fit; proxy tightened around typical Rb/OCXO values
     0.0 => (-24.0, -18.0),   # WFM  — σ_y(1) ∈ [7e-13, 7e-10]; Rb anchor -21.32
    -1.0 => (-27.0, -22.0),   # FFM  — flicker floor ∈ [1e-14, 1e-11]; user target ~1e-12
    -2.0 => (-32.0, -24.0),   # RWFM — both Rb fits near-zero; range covers "absent" to "modest"
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
samples to `<output_path>.checkpoint.h5`; resumes from the highest
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
    ckpt = output_path * ".checkpoint.h5"
    if resume && isfile(ckpt)
        h5open(ckpt, "r") do f
            nprev = Int(read(f["meta/n_done"]))
            @info "Resuming from checkpoint" nprev
            X[1:nprev, :]    .= f["features/X"][1:nprev, :]
            y[1:nprev, :]    .= f["labels/q_log10"][1:nprev, :]
            H[1:nprev, :]    .= f["labels/h_log10"][1:nprev, :]
            fpm[1:nprev]     .= Bool.(f["labels/fpm_present"][1:nprev])
            nll_vals[1:nprev] .= f["diagnostics/nll_values"][1:nprev]
            conv[1:nprev]    .= Bool.(f["diagnostics/converged"][1:nprev])
            done[1:nprev]    .= true
        end
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
            _write_h5(ckpt, X, y, H, fpm, nll_vals, conv, c)
        end
    end

    # Final write
    _write_h5(output_path, X, y, H, fpm, nll_vals, conv, n_samples; final=true)
    isfile(ckpt) && rm(ckpt)
    @info "done" output_path total=n_samples elapsed=(time() - t_start)
    nothing
end

function _write_h5(path, X, y, H, fpm, nll_vals, conv, n_done; final=false)
    mkpath(dirname(path))
    h5open(path, "w") do f
        # Feature matrix (float32) and label matrices
        f["features/X"]          = X              # Float32 (n_samples × 196)
        f["labels/q_log10"]      = y              # Float64 (n_samples × 3)
        f["labels/h_log10"]      = H              # Float64 (n_samples × 5)
        f["labels/fpm_present"]  = UInt8.(fpm)    # 0/1 so HDF5 handles it cleanly
        f["diagnostics/nll_values"] = nll_vals    # Float64 (n_samples,)
        f["diagnostics/converged"]  = UInt8.(conv)  # 0/1
        # Metadata (τ grid, feature names, progress)
        f["meta/taus"]            = CANONICAL_TAU_GRID   # Float64 (20,)
        f["meta/feature_names"]   = FEATURE_NAMES         # Vector{String} (196,)
        f["meta/n_done"]          = n_done               # Int
        f["meta/n_samples_total"] = size(X, 1)
    end
end

end # module
