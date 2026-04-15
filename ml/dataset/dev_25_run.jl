# dev_25_run.jl — Generate 25-sample dev dataset + overlay plot data.
#
# Produces:
#   ml/data/dev_25.npz              — features, labels, h_coeffs etc.
#   ml/data/dev_25_adev.npz         — full ADEV curves for each synthetic sample + real data
#   ml/data/dev_25_log.txt          — timing + convergence summary
#   ml/data/dev_25_adev_overlay.png — overlay plot (synthetic vs real)

using Pkg; Pkg.activate(@__DIR__)
include("generate_dataset.jl")
using .DatasetGen
using SigmaTau
using NPZ
using Random
using Statistics, Printf
using Plots

const N         = 131_072
const τ₀        = 1.0
const n_samples = 25

# ── helpers ──────────────────────────────────────────────────────────────────

"""Read a whitespace-separated MJD/phase file, skip comment/header rows."""
function readdlm_file(path)
    vals  = Float64[]
    nrows = 0
    for line in eachline(path)
        line = strip(line)
        isempty(line)           && continue
        startswith(line, "#")   && continue
        tokens = split(line)
        length(tokens) < 2      && continue
        try
            mjd = parse(Float64, tokens[1])
            ph  = parse(Float64, tokens[2])
            push!(vals, mjd, ph)
            nrows += 1
        catch
            # skip non-numeric header rows
        end
    end
    return reshape(vals, 2, nrows)' |> collect
end

factor_to_scalar(unit) =
    unit == "seconds"      ? 1.0  :
    unit == "nanoseconds"  ? 1e-9 :
    unit == "microseconds" ? 1e-6 :
    NaN

# ── main ─────────────────────────────────────────────────────────────────────

function main()
    out_dir  = joinpath(@__DIR__, "..", "data")
    mkpath(out_dir)

    npz_path  = joinpath(out_dir, "dev_25.npz")
    adev_path = joinpath(out_dir, "dev_25_adev.npz")
    log_path  = joinpath(out_dir, "dev_25_log.txt")
    plot_path = joinpath(out_dir, "dev_25_adev_overlay.png")

    isfile(npz_path) && rm(npz_path)
    isfile(npz_path * ".checkpoint.npz") && rm(npz_path * ".checkpoint.npz")

    # ── Generate the 25-sample dataset (features + q labels) ─────────────────
    t0 = time()
    DatasetGen.generate_dataset(npz_path; n_samples=n_samples, N=N, τ₀=τ₀, resume=false)
    gen_elapsed = time() - t0
    @info "Dataset generation done" elapsed=gen_elapsed

    # ── Re-run phase generation to extract per-sample ADEV curves ────────────
    # (dataset stores features, not raw phase; double compute is fine for n=25)
    adev_curves = Matrix{Float64}(undef, n_samples, length(CANONICAL_M_LIST))
    h_table     = Matrix{Float64}(undef, n_samples, 5)   # log10(h₊₂,₊₁,₀,₋₁,₋₂)
    fpm_table   = falses(n_samples)

    t1 = time()
    for i in 1:n_samples
        rng = Xoshiro(42 + i)
        p   = DatasetGen.draw_sample_params(rng)
        x   = generate_composite_noise(p.h_coeffs, N, τ₀; seed = 42 + i + 10_000_000)
        r   = adev(x, τ₀; m_list = CANONICAL_M_LIST)
        adev_curves[i, :] .= r.deviation
        for (j, α) in enumerate((2.0, 1.0, 0.0, -1.0, -2.0))
            h_table[i, j] = haskey(p.h_coeffs, α) ? log10(p.h_coeffs[α]) : NaN
        end
        fpm_table[i] = p.fpm_present
    end
    adev_elapsed = time() - t1

    # ── Load real-data and compute ADEV on canonical grid ────────────────────
    real_path = joinpath(@__DIR__, "..", "..", "reference", "raw", "6k27febunsteered.txt")
    real_adev = fill(NaN, length(CANONICAL_M_LIST))
    real_unit = "unknown"
    factor    = NaN

    if isfile(real_path)
        raw      = readdlm_file(real_path)
        mjd      = raw[:, 1]
        ph_raw   = raw[:, 2]
        step_s   = (mjd[2] - mjd[1]) * 86400.0
        @info "Real-data file found" path=real_path step_s=step_s n_points=length(ph_raw)

        # Unit detection: compute ADEV(1s) treating values as-is,
        # then pick the scale factor that puts σ_y(1s) in Rb ballpark [1e-12, 1e-9]
        ad1 = let x_probe = ph_raw[1:min(65_536, length(ph_raw))]
            r = adev(x_probe, 1.0)
            r.deviation[1]   # m=1 → τ=1 s
        end
        target_lo, target_hi = 1e-12, 1e-9
        factor    = 1.0
        real_unit = "seconds"
        if !(target_lo <= ad1 <= target_hi)
            if target_lo * 1e9 <= ad1 <= target_hi * 1e9
                factor    = 1e-9
                real_unit = "nanoseconds"
            elseif target_lo * 1e6 <= ad1 <= target_hi * 1e6
                factor    = 1e-6
                real_unit = "microseconds"
            else
                @warn "Could not auto-detect real-data units; ADEV(1)=$(ad1). Leaving as seconds."
                factor    = 1.0
                real_unit = "seconds"
            end
        end
        @info "Unit detection" raw_adev1=ad1 unit=real_unit factor=factor

        ph = ph_raw .* factor
        w  = min(N, length(ph))
        r  = adev(view(ph, 1:w), τ₀; m_list = CANONICAL_M_LIST)
        real_adev = r.deviation
    else
        @warn "Real-data file not found; skipping real-data ADEV" real_path
    end

    # ── Save ADEV data ────────────────────────────────────────────────────────
    NPZ.npzwrite(adev_path, Dict(
        "taus"                   => CANONICAL_TAU_GRID,
        "adev_synth"             => adev_curves,
        "adev_real"              => real_adev,
        "h_log10"                => h_table,
        "fpm_present"            => UInt8.(fpm_table),
        "real_unit_factor_used"  => factor_to_scalar(real_unit),
    ))
    @info "ADEV NPZ saved" adev_path

    # ── Overlay plot ──────────────────────────────────────────────────────────
    plot_saved = try
        τ = CANONICAL_TAU_GRID
        p = plot(xscale=:log10, yscale=:log10,
                 xlabel="τ (s)", ylabel="σ_y(τ)",
                 title="Synthetic (n=25) vs Real GMR6000 ADEV",
                 legend=:topright, size=(900, 600))
        for i in 1:n_samples
            plot!(p, τ, adev_curves[i, :]; color=:steelblue, alpha=0.4, lw=0.8, label=false)
        end
        if isfile(real_path)
            plot!(p, τ, real_adev; color=:red, lw=2.5, label="GMR6000 Feb (real)")
        end
        savefig(p, plot_path)
        @info "Overlay plot saved" plot_path
        true
    catch e
        @warn "Plots.jl unavailable or failed, skipping PNG" exception=e
        false
    end

    # ── Summary log ──────────────────────────────────────────────────────────
    d = NPZ.npzread(npz_path)
    n_converged = sum(Int, d["converged"])

    open(log_path, "w") do io
        println(io, "n_samples:          ", n_samples)
        println(io, "N:                  ", N)
        println(io, "τ₀:                 ", τ₀)
        println(io, "threads:            ", Threads.nthreads())
        println(io, "dataset_gen_s:      ", round(gen_elapsed,   digits=3))
        println(io, "adev_recompute_s:   ", round(adev_elapsed,  digits=3))
        println(io, "per_sample_s:       ", round(gen_elapsed / n_samples, digits=3))
        println(io, "converged:          ", n_converged, " / ", n_samples)
        println(io, "real_data_unit:     ", real_unit)
        println(io, "real_data_factor:   ", isnan(factor) ? "N/A" : factor)
        # Extrapolate to 10k with current thread count
        est_10k_s = (gen_elapsed / n_samples) * 10_000
        println(io, "estimated_10k_s:    ", round(est_10k_s, digits=1),
                    "  (", round(est_10k_s / 3600, digits=2), " hr)")
        println(io, "plot_saved:         ", plot_saved)
        println(io, "npz_path:           ", npz_path)
        println(io, "adev_path:          ", adev_path)
        plot_saved && println(io, "plot_path:          ", plot_path)
    end

    println(read(log_path, String))
    println("done.")
    println("  dataset NPZ : ", npz_path)
    println("  ADEV NPZ    : ", adev_path)
    plot_saved && println("  overlay plot: ", plot_path)
end

main()
