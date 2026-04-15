# dev_25_run.jl — Generate 25-sample dev dataset + overlay plot data.
#
# Produces:
#   ml/data/dev_25.h5                 — features, labels, h_coeffs etc. (HDF5)
#   ml/data/dev_25_adev.csv           — ADEV/MDEV/HDEV/MHDEV curves per sample (long format)
#   ml/data/dev_25_real.csv           — real-data ADEV on canonical τ grid
#   ml/data/dev_25_meta.csv           — per-sample metadata (h's, fpm_present, nll, converged)
#   ml/data/dev_25_log.txt            — timing + convergence summary
#   ml/data/dev_25_adev_overlay.png   — overlay plot (synthetic vs real)

using Pkg; Pkg.activate(@__DIR__)
include("generate_dataset.jl")
using .DatasetGen
using SigmaTau
using HDF5
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

"""Write a NamedTuple of equal-length vectors as a CSV file."""
function write_csv(path::String, data::NamedTuple)
    keys_ = collect(keys(data))
    n = length(first(values(data)))
    open(path, "w") do io
        println(io, join(keys_, ","))
        for i in 1:n
            row = [v[i] for v in values(data)]
            row_strs = map(x -> x isa AbstractFloat ? (@sprintf "%.6e" x) : string(x), row)
            println(io, join(row_strs, ","))
        end
    end
end

# ── main ─────────────────────────────────────────────────────────────────────

function main()
    out_dir  = joinpath(@__DIR__, "..", "data")
    mkpath(out_dir)

    h5_path   = joinpath(out_dir, "dev_25.h5")
    adev_path = joinpath(out_dir, "dev_25_adev.csv")
    real_path_out = joinpath(out_dir, "dev_25_real.csv")
    meta_path = joinpath(out_dir, "dev_25_meta.csv")
    log_path  = joinpath(out_dir, "dev_25_log.txt")
    plot_path = joinpath(out_dir, "dev_25_adev_overlay.png")

    isfile(h5_path) && rm(h5_path)
    isfile(h5_path * ".checkpoint.h5") && rm(h5_path * ".checkpoint.h5")

    # ── Generate the 25-sample dataset (features + q labels) ─────────────────
    t0 = time()
    DatasetGen.generate_dataset(h5_path; n_samples=n_samples, N=N, τ₀=τ₀, resume=false)
    gen_elapsed = time() - t0
    @info "Dataset generation done" elapsed=gen_elapsed

    # ── Re-run phase generation to extract per-sample ADEV/MDEV/HDEV/MHDEV ──
    # (dataset stores features, not raw phase; double compute is fine for n=25)
    n_taus       = length(CANONICAL_M_LIST)
    adev_curves  = Matrix{Float64}(undef, n_samples, n_taus)
    mdev_curves  = Matrix{Float64}(undef, n_samples, n_taus)
    hdev_curves  = Matrix{Float64}(undef, n_samples, n_taus)
    mhdev_curves = Matrix{Float64}(undef, n_samples, n_taus)
    h_table      = Matrix{Float64}(undef, n_samples, 5)   # log10(h₊₂,₊₁,₀,₋₁,₋₂)
    fpm_table    = falses(n_samples)

    t1 = time()
    for i in 1:n_samples
        rng = Xoshiro(42 + i)
        p   = DatasetGen.draw_sample_params(rng)
        x   = generate_composite_noise(p.h_coeffs, N, τ₀; seed = 42 + i + 10_000_000)
        r_adev  = adev( x, τ₀; m_list = CANONICAL_M_LIST)
        r_mdev  = mdev( x, τ₀; m_list = CANONICAL_M_LIST)
        r_hdev  = hdev( x, τ₀; m_list = CANONICAL_M_LIST)
        r_mhdev = mhdev(x, τ₀; m_list = CANONICAL_M_LIST)
        adev_curves[i, :]  .= r_adev.deviation
        mdev_curves[i, :]  .= r_mdev.deviation
        hdev_curves[i, :]  .= r_hdev.deviation
        mhdev_curves[i, :] .= r_mhdev.deviation
        for (j, α) in enumerate((2.0, 1.0, 0.0, -1.0, -2.0))
            h_table[i, j] = haskey(p.h_coeffs, α) ? log10(p.h_coeffs[α]) : NaN
        end
        fpm_table[i] = p.fpm_present
    end
    adev_elapsed = time() - t1

    # ── Load real-data and compute ADEV on canonical grid ────────────────────
    real_path = joinpath(@__DIR__, "..", "..", "reference", "raw", "6k27febunsteered.txt")
    real_adev = fill(NaN, n_taus)
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

    # ── Save ADEV CSV (long format, 25×20 = 500 rows) ────────────────────────
    τ_grid = CANONICAL_TAU_GRID
    open(adev_path, "w") do io
        println(io, "sample_idx,tau,adev,mdev,hdev,mhdev")
        for i in 1:n_samples
            for j in 1:n_taus
                @printf(io, "%d,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    i, τ_grid[j],
                    adev_curves[i, j], mdev_curves[i, j],
                    hdev_curves[i, j], mhdev_curves[i, j])
            end
        end
    end
    @info "ADEV CSV saved" adev_path

    # ── Save real-data ADEV CSV (20 rows) ────────────────────────────────────
    open(real_path_out, "w") do io
        println(io, "tau,adev_real")
        for j in 1:n_taus
            @printf(io, "%.6e,%.6e\n", τ_grid[j], real_adev[j])
        end
    end
    @info "Real ADEV CSV saved" real_path_out

    # ── Save per-sample metadata CSV ─────────────────────────────────────────
    # Read back converged / nll from HDF5
    nll_read  = Vector{Float64}(undef, n_samples)
    conv_read = Vector{Bool}(undef, n_samples)
    h5open(h5_path, "r") do f
        nll_read  .= f["diagnostics/nll_values"][]
        conv_read .= Bool.(f["diagnostics/converged"][])
    end

    open(meta_path, "w") do io
        println(io, "sample_idx,h_log10_p2,h_log10_p1,h_log10_0,h_log10_m1,h_log10_m2,fpm_present,nll,converged")
        for i in 1:n_samples
            @printf(io, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.6e,%d\n",
                i,
                h_table[i,1], h_table[i,2], h_table[i,3], h_table[i,4], h_table[i,5],
                Int(fpm_table[i]),
                nll_read[i],
                Int(conv_read[i]))
        end
    end
    @info "Meta CSV saved" meta_path

    # ── Overlay plot ──────────────────────────────────────────────────────────
    plot_saved = try
        τ = τ_grid
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
    n_converged = sum(Int, conv_read)

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
        println(io, "h5_path:            ", h5_path)
        println(io, "adev_csv:           ", adev_path)
        println(io, "real_csv:           ", real_path_out)
        println(io, "meta_csv:           ", meta_path)
        plot_saved && println(io, "plot_path:          ", plot_path)
    end

    println(read(log_path, String))
    println("done.")
    println("  dataset HDF5 : ", h5_path)
    println("  ADEV CSV     : ", adev_path)
    println("  real CSV     : ", real_path_out)
    println("  meta CSV     : ", meta_path)
    plot_saved && println("  overlay plot : ", plot_path)
end

main()
