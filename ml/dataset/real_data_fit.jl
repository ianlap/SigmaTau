# real_data_fit.jl — Diagnostic: fit KF noise params on the real GMR6000 file.
#
# Runs both mhdev_fit (legacy) and optimize_kf_nll (NLL) on the full phase
# record (6k27febunsteered.txt), computes theoretical ADEV/MHDEV from the
# fitted q values, and overlays them with the measured deviations.

using Pkg; Pkg.activate(@__DIR__)
using SigmaTau
using Printf, Statistics, LinearAlgebra
using Plots

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

    # --- Fit 2: optimize_nll on a windowed subset (full 3M is too slow) ---
    # Use a long window, say 2^19 = 524k samples (~6 days).
    # Warm-start from mhdev_fit's q values directly via ClockNoiseParams;
    # optimize_qwpm=false fixes R at its analytical value, preserving the
    # pre-refactor optimize_kf_nll semantic.
    q_wpm_init  = fit_res.q_wpm  > 0 ? fit_res.q_wpm  : 1e-22
    q_wfm_init  = fit_res.q_wfm  > 0 ? fit_res.q_wfm  : 1e-22
    q_rwfm_init = fit_res.q_rwfm > 0 ? fit_res.q_rwfm : 1e-30
    if !(fit_res.q_wpm > 0 && fit_res.q_wfm > 0 && fit_res.q_rwfm > 0)
        @warn "mhdev_fit yielded non-positive q(s); defaults applied for NLL warm-start"
    end
    noise_init = ClockNoiseParams(q_wpm = q_wpm_init,
                                  q_wfm = q_wfm_init,
                                  q_rwfm = q_rwfm_init)
    @info "noise_init for NLL warm start" q_wpm=q_wpm_init q_wfm=q_wfm_init q_rwfm=q_rwfm_init

    window_N = min(2^19, length(ph))
    nll_res = optimize_nll(view(ph, 1:window_N), τ₀;
                           noise_init = noise_init,
                           optimize_qwpm = false,
                           verbose = false,
                           max_iter = 1000)
    # optimize_nll drops .converged; hard-wire true as a placeholder (see FIX_PARKING_LOT.md).
    nll_converged = true
    @info "optimize_nll result" q_wpm=nll_res.q_wpm q_wfm=nll_res.q_wfm q_rwfm=nll_res.q_rwfm converged=nll_converged

    # --- Theoretical ADEV from each fit ---
    # Direct q → σ_y(τ) under the Wu 2023 clock-model convention
    # (matches julia/src/clock_model.jl h_to_q / q_to_h):
    #   σ²_y,WPM(τ)  = 3·q_wpm / τ²
    #   σ²_y,WFM(τ)  = q_wfm / τ
    #   σ²_y,RWFM(τ) = q_rwfm · τ / 3
    function adev_from_q(q_wpm, q_wfm, q_rwfm, τ_vec)
        σ2 = [3 * q_wpm / τ^2 + q_wfm / τ + q_rwfm * τ / 3 for τ in τ_vec]
        return sqrt.(max.(σ2, 0.0))
    end

    adev_theo_mhdev = adev_from_q(fit_res.q_wpm, fit_res.q_wfm, fit_res.q_rwfm, τs)
    adev_theo_nll   = adev_from_q(nll_res.q_wpm, nll_res.q_wfm, nll_res.q_rwfm, τs)

    # --- Save CSV of raw data ---
    out_csv = joinpath(@__DIR__, "..", "data", "real_data_fit.csv")
    open(out_csv, "w") do io
        println(io, "tau,adev,mdev,hdev,mhdev,adev_theo_mhdevfit,adev_theo_nllfit")
        for i in eachindex(τs)
            @printf(io, "%.3f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                τs[i],
                r_adev.deviation[i], r_mdev.deviation[i],
                r_hdev.deviation[i], r_mhdev.deviation[i],
                adev_theo_mhdev[i], adev_theo_nll[i])
        end
    end
    @info "Saved CSV" out_csv

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
        println(io, "--- optimize_nll (window N=", window_N, ") ---")
        println(io, @sprintf("q_wpm   = %.3e   (log10 = %.3f)", nll_res.q_wpm,  log10(max(nll_res.q_wpm,  1e-99))))
        println(io, @sprintf("q_wfm   = %.3e   (log10 = %.3f)", nll_res.q_wfm,  log10(max(nll_res.q_wfm,  1e-99))))
        println(io, @sprintf("q_rwfm  = %.3e   (log10 = %.3f)", nll_res.q_rwfm, log10(max(nll_res.q_rwfm, 1e-99))))
        println(io, "converged: ", nll_converged)
        println(io, "")
        println(io, "Implied h_α (from q values via q_to_h; Wu 2023 convention):")
        for (name, q_params) in [
            ("mhdev", ClockNoiseParams(q_wpm=max(fit_res.q_wpm, 1e-99),
                                       q_wfm=max(fit_res.q_wfm, 1e-99),
                                       q_rwfm=max(fit_res.q_rwfm, 1e-99))),
            ("nll",   ClockNoiseParams(q_wpm=max(nll_res.q_wpm, 1e-99),
                                       q_wfm=max(nll_res.q_wfm, 1e-99),
                                       q_rwfm=max(nll_res.q_rwfm, 1e-99))),
        ]
            h = q_to_h(q_params, Float64(τ₀))
            println(io, "  $name: log10(h_+2)=", @sprintf("%.2f", log10(max(h.h2, 1e-99))),
                                "  log10(h_0)=",  @sprintf("%.2f", log10(max(h.h0, 1e-99))),
                                "  log10(h_-2)=", @sprintf("%.2f", log10(max(h.h_2, 1e-99))))
        end
    end
    println(read(log_path, String))
end

main()
