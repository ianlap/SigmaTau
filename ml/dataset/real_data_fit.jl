# real_data_fit.jl — Diagnostic: fit KF noise params on the real GMR6000 file.
#
# Runs both mhdev_fit (legacy) and optimize_kf_nll (NLL) on the full phase
# record (6k27febunsteered.txt), computes theoretical ADEV/MHDEV from the
# fitted q values, and overlays them with the measured deviations.

using Pkg; Pkg.activate(@__DIR__)
using SigmaTau
using Printf, Statistics, LinearAlgebra
using Plots
using NPZ

const real_path = joinpath(@__DIR__, "..", "..", "reference", "raw", "6k27febunsteered.txt")

function load_phase_seconds(path)
    # Two columns: MJD, phase[ns]. Skip non-numeric header lines.
    mjd = Float64[]; ph = Float64[]
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || startswith(line, "#")) && continue
        tok = split(line)
        length(tok) < 2 && continue
        try
            push!(mjd, parse(Float64, tok[1]))
            push!(ph,  parse(Float64, tok[2]))
        catch; end
    end
    # Unit detection: if ADEV(1) treating as-is is in [1e-12 .. 1e-9], it's seconds;
    # if in [1e-3 .. 1], treat as nanoseconds.
    test_x = ph[1:min(65_536, end)]
    ad1 = adev(test_x, 1.0).deviation[1]
    factor = 1.0
    unit   = "seconds"
    if ad1 > 1e-3
        factor = 1e-9
        unit   = "nanoseconds"
    elseif ad1 > 1e-6
        factor = 1e-6
        unit   = "microseconds"
    end
    @info "Real-data unit detection" first_adev=ad1 chosen_unit=unit factor=factor
    return mjd, ph .* factor, factor, unit
end

function main()
    mjd, ph, factor, unit = load_phase_seconds(real_path)
    τ₀ = round((mjd[2] - mjd[1]) * 86400, digits=6)
    @info "File loaded" n_phase=length(ph) τ₀=τ₀ unit=unit

    # --- Compute deviations on an extended log-spaced τ grid that reaches into RWFM ---
    N = length(ph)
    m_max = min(N ÷ 10, 1_000_000)  # safety factor 10; also cap
    m_list = unique(round.(Int, exp10.(range(0, log10(m_max), length=40))))
    @info "τ grid" m_count=length(m_list) τ_min=m_list[1] τ_max=m_list[end]

    r_adev  = adev(ph, τ₀; m_list=m_list)
    r_mdev  = mdev(ph, τ₀; m_list=m_list)
    r_hdev  = hdev(ph, τ₀; m_list=m_list)
    r_mhdev = mhdev(ph, τ₀; m_list=m_list)

    τs = r_adev.tau
    @info "Deviations computed"

    # --- Fit 1: mhdev_fit legacy tool ---
    # Need to pick region τ-index ranges. For Rb at τ₀=1s:
    #   WPM dominates m=1..10 (τ=1..10s)  — slope -3 in MHDEV²
    #   WFM dominates m=10..1000 (τ=10..1000s)
    #   RWFM dominates m=10000..max (τ=1e4..∞)
    # We use the τ vector from r_mhdev (its tau is log-spaced same as m_list).
    # Each region is a Vector{Int} of indices into r_mhdev.deviation.
    σ_mhdev = r_mhdev.deviation
    τ_mhdev = r_mhdev.tau
    # Build index ranges:
    idx_wpm  = findall(τ -> 1.0    <= τ <= 10.0,    τ_mhdev)
    idx_wfm  = findall(τ -> 30.0   <= τ <= 1000.0,  τ_mhdev)
    idx_rwfm = findall(τ -> 1e4    <= τ <= 1e5,     τ_mhdev)
    @info "mhdev_fit regions" idx_wpm=length(idx_wpm) idx_wfm=length(idx_wfm) idx_rwfm=length(idx_rwfm)
    regions = [
        (noise_type=:wpm,  indices=idx_wpm),
        (noise_type=:wfm,  indices=idx_wfm),
        (noise_type=:rwfm, indices=idx_rwfm),
    ]

    fit_res = mhdev_fit(τ_mhdev, σ_mhdev, regions)
    @info "mhdev_fit result" q_wpm=fit_res.q_wpm q_wfm=fit_res.q_wfm q_rwfm=fit_res.q_rwfm

    # --- Fit 2: optimize_kf_nll on a windowed subset (full 3M is too slow) ---
    # Use a long window, say 2^19 = 524k samples (~6 days).
    # h-warm init from mhdev_fit via the inverse of the analytical mapping.
    # Analytical forward mapping was:
    #   q_wpm  = h[2.0] * f_h / (2π²)   → h[2.0]  = q_wpm * 2π² / f_h
    #   q_wfm  = h[0.0] / 2             → h[0.0]  = 2 * q_wfm
    #   q_rwfm = (2π²/3) * h[-2.0]      → h[-2.0] = 3 * q_rwfm / (2π²)
    f_h = 1.0 / (2τ₀)
    h_init = Dict(
         2.0 => fit_res.q_wpm  * 2π^2 / f_h,
         0.0 => 2 * fit_res.q_wfm,
        -2.0 => 3 * fit_res.q_rwfm / (2π^2),
    )
    # Enforce positivity (mhdev_fit may yield zeros for unused regions)
    for (α, v) in collect(h_init)
        if v <= 0
            @warn "h_init $α was non-positive $(v); defaulting" α=α
            h_init[α] = 10.0 ^ (α == 2.0 ? -22 : (α == 0.0 ? -22 : -26))
        end
    end
    @info "h_init from mhdev_fit (for NLL warm start)" h_plus2=h_init[2.0] h_0=h_init[0.0] h_minus2=h_init[-2.0]

    window_N = min(2^19, length(ph))
    nll_res = optimize_kf_nll(view(ph, 1:window_N), τ₀;
                              h_init = h_init, verbose=false, max_iter=1000)
    @info "optimize_kf_nll result" q_wpm=nll_res.q_wpm q_wfm=nll_res.q_wfm q_rwfm=nll_res.q_rwfm converged=nll_res.converged

    # --- Theoretical ADEV from each fit ---
    # For KF 3-state q params:
    #   σ²_y(τ) = 3·q_wpm/τ² + q_wfm/τ + q_rwfm·τ · (something)
    # Use the SP1065 mapping through h's:
    #   h[2]    → σ²_y(τ) = 3·f_h·h[2]/(4π²·τ²)   (WPM)
    #   h[0]    → σ²_y(τ) = h[0]/(2τ)              (WFM)
    #   h[-2]   → σ²_y(τ) = h[-2] · (2π²/3) · τ    (RWFM)
    function adev_from_q(q_wpm, q_wfm, q_rwfm, τ_vec)
        # Invert the q↔h mapping to recover h's, then use the SP1065 formulas.
        h_plus2  = q_wpm * 2π^2 / f_h
        h_0      = 2 * q_wfm
        h_minus2 = 3 * q_rwfm / (2π^2)
        # Theoretical two-sided PSDs sum; ADEV^2 is:
        σ2 = [3 * f_h * h_plus2 / (4π^2 * τ^2) + h_0 / (2τ) + h_minus2 * (2π^2/3) * τ for τ in τ_vec]
        return sqrt.(max.(σ2, 0.0))
    end

    adev_theo_mhdev = adev_from_q(fit_res.q_wpm, fit_res.q_wfm, fit_res.q_rwfm, τs)
    adev_theo_nll   = adev_from_q(nll_res.q_wpm, nll_res.q_wfm, nll_res.q_rwfm, τs)

    # --- Save NPZ of raw data ---
    out_npz = joinpath(@__DIR__, "..", "data", "real_data_fit.npz")
    NPZ.npzwrite(out_npz, Dict(
        "tau"              => collect(τs),
        "adev_real"        => collect(r_adev.deviation),
        "mdev_real"        => collect(r_mdev.deviation),
        "hdev_real"        => collect(r_hdev.deviation),
        "mhdev_real"       => collect(r_mhdev.deviation),
        "adev_theo_mhdevfit" => adev_theo_mhdev,
        "adev_theo_nllfit"   => adev_theo_nll,
        "q_wpm_mhdev"   => fit_res.q_wpm,
        "q_wfm_mhdev"   => fit_res.q_wfm,
        "q_rwfm_mhdev"  => fit_res.q_rwfm,
        "q_wpm_nll"     => nll_res.q_wpm,
        "q_wfm_nll"     => nll_res.q_wfm,
        "q_rwfm_nll"    => nll_res.q_rwfm,
    ))
    @info "Saved NPZ" out_npz

    # --- Plot ---
    plt = plot(xscale=:log10, yscale=:log10,
         xlabel="τ (s)", ylabel="σ_y(τ)",
         title="6k27febunsteered: measured vs theoretical ADEV",
         legend=:bottomleft, size=(900, 600))
    plot!(plt, τs, r_adev.deviation,    lw=2.5, color=:black,    label="measured ADEV")
    plot!(plt, τs, r_mdev.deviation,    lw=1.5, color=:gray50,   label="measured MDEV", alpha=0.7)
    plot!(plt, τs, r_hdev.deviation,    lw=1.5, color=:gray70,   label="measured HDEV", alpha=0.5)
    plot!(plt, τs, r_mhdev.deviation,   lw=1.5, color=:gray40,   label="measured MHDEV", alpha=0.7)
    plot!(plt, τs, adev_theo_mhdev,     lw=2.5, color=:red,      label="theory ADEV (mhdev_fit)")
    plot!(plt, τs, adev_theo_nll,       lw=2.5, color=:blue, ls=:dash, label="theory ADEV (NLL fit)")

    plot_path = joinpath(@__DIR__, "..", "data", "real_data_fit.png")
    savefig(plt, plot_path)
    @info "Plot saved" plot_path

    # --- Summary log ---
    log_path = joinpath(@__DIR__, "..", "data", "real_data_fit.txt")
    open(log_path, "w") do io
        println(io, "file:                   ", real_path)
        println(io, "unit:                   ", unit, " (factor ", factor, ")")
        println(io, "n_samples:              ", length(ph))
        println(io, "τ₀:                     ", τ₀)
        println(io, "τ range:                ", minimum(τs), " … ", maximum(τs))
        println(io, "")
        println(io, "--- mhdev_fit (legacy) ---")
        println(io, @sprintf("q_wpm   = %.3e   (log10 = %.3f)", fit_res.q_wpm,  log10(max(fit_res.q_wpm,  1e-99))))
        println(io, @sprintf("q_wfm   = %.3e   (log10 = %.3f)", fit_res.q_wfm,  log10(max(fit_res.q_wfm,  1e-99))))
        println(io, @sprintf("q_rwfm  = %.3e   (log10 = %.3f)", fit_res.q_rwfm, log10(max(fit_res.q_rwfm, 1e-99))))
        println(io, "")
        println(io, "--- optimize_kf_nll (window N=", window_N, ") ---")
        println(io, @sprintf("q_wpm   = %.3e   (log10 = %.3f)", nll_res.q_wpm,  log10(max(nll_res.q_wpm,  1e-99))))
        println(io, @sprintf("q_wfm   = %.3e   (log10 = %.3f)", nll_res.q_wfm,  log10(max(nll_res.q_wfm,  1e-99))))
        println(io, @sprintf("q_rwfm  = %.3e   (log10 = %.3f)", nll_res.q_rwfm, log10(max(nll_res.q_rwfm, 1e-99))))
        println(io, "converged: ", nll_res.converged)
        println(io, "")
        println(io, "Implied h_α (from q values via inverse mapping):")
        for (name, q1, q2, q3) in [
            ("mhdev", fit_res.q_wpm, fit_res.q_wfm, fit_res.q_rwfm),
            ("nll",   nll_res.q_wpm, nll_res.q_wfm, nll_res.q_rwfm),
        ]
            h2 = q1 * 2π^2 / f_h
            h0 = 2 * q2
            hm = 3 * q3 / (2π^2)
            println(io, "  $name: log10(h_+2)=", @sprintf("%.2f", log10(max(h2, 1e-99))),
                                "  log10(h_0)=",  @sprintf("%.2f", log10(max(h0, 1e-99))),
                                "  log10(h_-2)=", @sprintf("%.2f", log10(max(hm, 1e-99))))
        end
    end
    println(read(log_path, String))
end

main()
