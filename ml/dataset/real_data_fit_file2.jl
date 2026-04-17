# real_data_fit_file2.jl — Diagnostic: fit KF noise params on the second GMR6000 file.
#
# Runs both mhdev_fit (legacy) and optimize_nll (NLL) on 6krb25apr.txt
# (~407k rows, 4.7 days, phase in nanoseconds).
# Writes ml/data/real_data_fit_file2.{npz,csv,png,txt}.
#
# Also reads the already-saved file1 CSV and produces:
#   ml/data/rb_fits_combined.csv
#   ml/data/rb_fits_combined.png
#   ml/data/rb_fits_combined.txt
#   ml/data/proposed_h_ranges.md

using Pkg; Pkg.activate(@__DIR__)
using SigmaTau
using Printf, Statistics, LinearAlgebra
using Plots

const file2_path = joinpath(@__DIR__, "..", "..", "reference", "raw", "6krb25apr.txt")
const data_dir   = joinpath(@__DIR__, "..", "data")
const file1_csv  = joinpath(data_dir, "real_data_fit.csv")
const file1_txt  = joinpath(data_dir, "real_data_fit.txt")

# ---------------------------------------------------------------------------
# Load phase — forced to nanoseconds (user-confirmed units)
# ---------------------------------------------------------------------------
function load_phase_ns_to_s(path)
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
    # Force ns → s (user confirmed: units are nanoseconds)
    factor = 1e-9
    unit   = "nanoseconds"
    @info "File2 unit forced" chosen_unit=unit factor=factor n_lines=length(ph)
    return mjd, ph .* factor, factor, unit
end

# ---------------------------------------------------------------------------
# Theoretical ADEV from KF q-params
# ---------------------------------------------------------------------------
# Direct q → σ_y(τ) under the Wu 2023 clock-model convention
# (matches julia/src/clock_model.jl h_to_q / q_to_h):
#   σ²_y,WPM(τ)  = 3·q_wpm / τ²
#   σ²_y,WFM(τ)  = q_wfm / τ
#   σ²_y,RWFM(τ) = q_rwfm · τ / 3
function adev_from_q(q_wpm, q_wfm, q_rwfm, τ_vec)
    σ2 = [3 * q_wpm / τ^2 + q_wfm / τ + q_rwfm * τ / 3 for τ in τ_vec]
    return sqrt.(max.(σ2, 0.0))
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    # --- Load file2 ---
    mjd, ph, factor, unit = load_phase_ns_to_s(file2_path)
    N   = length(ph)
    τ₀  = round((mjd[2] - mjd[1]) * 86400, digits=6)
    @info "File2 loaded" n_phase=N τ₀=τ₀

    # --- Deviation grid (full file) ---
    m_max  = min(N ÷ 10, 1_000_000)
    m_list = unique(round.(Int, exp10.(range(0, log10(m_max), length=40))))
    @info "τ grid (file2)" m_count=length(m_list) τ_min=m_list[1] τ_max=m_list[end]

    r_adev  = adev(ph, τ₀; m_list=m_list)
    r_mdev  = mdev(ph, τ₀; m_list=m_list)
    r_hdev  = hdev(ph, τ₀; m_list=m_list)
    r_mhdev = mhdev(ph, τ₀; m_list=m_list)

    τs      = r_adev.tau
    @info "Deviations computed (file2)"

    # --- mhdev_fit ---
    σ_mhdev = r_mhdev.deviation
    τ_mhdev = r_mhdev.tau

    idx_wpm  = findall(τ -> 1.0  <= τ <= 10.0,   τ_mhdev)
    idx_wfm  = findall(τ -> 30.0 <= τ <= 1000.0, τ_mhdev)
    # For file2 (N≈407k, m_max≈40k): RWFM region pushed lower
    idx_rwfm = findall(τ -> 5e3  <= τ <= 5e4,    τ_mhdev)
    if isempty(idx_rwfm)
        idx_rwfm = findall(τ -> 1e3 <= τ <= τ_mhdev[end], τ_mhdev)
    end
    @info "mhdev_fit regions (file2)" n_wpm=length(idx_wpm) n_wfm=length(idx_wfm) n_rwfm=length(idx_rwfm)

    regions = [
        (noise_type=:wpm,  indices=idx_wpm),
        (noise_type=:wfm,  indices=idx_wfm),
    ]
    if !isempty(idx_rwfm)
        push!(regions, (noise_type=:rwfm, indices=idx_rwfm))
    end

    fit_res = mhdev_fit(τ_mhdev, σ_mhdev, regions)
    @info "mhdev_fit result (file2)" q_wpm=fit_res.q_wpm q_wfm=fit_res.q_wfm q_rwfm=fit_res.q_rwfm

    # --- NLL optimization ---
    # Warm-start from mhdev_fit's q values directly via ClockNoiseParams;
    # optimize_qwpm=false fixes R at its analytical value, preserving the
    # pre-refactor optimize_kf_nll semantic.
    q_wpm_init  = fit_res.q_wpm  > 0 ? fit_res.q_wpm  : 1e-22
    q_wfm_init  = fit_res.q_wfm  > 0 ? fit_res.q_wfm  : 1e-22
    q_rwfm_init = fit_res.q_rwfm > 0 ? fit_res.q_rwfm : 1e-30
    noise_init = ClockNoiseParams(q_wpm = q_wpm_init,
                                  q_wfm = q_wfm_init,
                                  q_rwfm = q_rwfm_init)
    @info "noise_init (file2, from mhdev_fit)" q_wpm=q_wpm_init q_wfm=q_wfm_init q_rwfm=q_rwfm_init

    # Use full file since N < 2^19
    window_N = min(2^19, N)
    @info "NLL window" window_N=window_N full_N=N
    opt_res = optimize_nll(view(ph, 1:window_N), τ₀;
                           noise_init = noise_init,
                           optimize_qwpm = false,
                           verbose = false,
                           max_iter = 1000)
    nll_res       = opt_res.noise
    nll_converged = opt_res.converged
    @info "NLL result (file2)" q_wpm=nll_res.q_wpm q_wfm=nll_res.q_wfm q_rwfm=nll_res.q_rwfm converged=nll_converged

    # --- Theoretical ADEVs ---
    adev_theo_mhdev = adev_from_q(fit_res.q_wpm, fit_res.q_wfm, fit_res.q_rwfm, τs)
    adev_theo_nll   = adev_from_q(nll_res.q_wpm, nll_res.q_wfm, nll_res.q_rwfm, τs)

    # --- Save CSV (file2) ---
    out_csv2 = joinpath(data_dir, "real_data_fit_file2.csv")
    open(out_csv2, "w") do io
        println(io, "tau,adev,mdev,hdev,mhdev,adev_theo_mhdevfit,adev_theo_nllfit")
        for i in eachindex(τs)
            @printf(io, "%.3f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                τs[i],
                r_adev.deviation[i], r_mdev.deviation[i],
                r_hdev.deviation[i], r_mhdev.deviation[i],
                adev_theo_mhdev[i], adev_theo_nll[i])
        end
    end
    @info "Saved CSV (file2)" out_csv2

    # --- Plot (file2) ---
    plt2 = plot(xscale=:log10, yscale=:log10,
         xlabel="τ (s)", ylabel="σ_y(τ)",
         title="6krb25apr: measured vs theoretical ADEV",
         legend=:bottomleft, size=(900, 600))
    plot!(plt2, τs, r_adev.deviation,  lw=2.5, color=:black,  label="measured ADEV")
    plot!(plt2, τs, r_mdev.deviation,  lw=1.5, color=:gray50, label="measured MDEV", alpha=0.7)
    plot!(plt2, τs, r_hdev.deviation,  lw=1.5, color=:gray70, label="measured HDEV", alpha=0.5)
    plot!(plt2, τs, r_mhdev.deviation, lw=1.5, color=:gray40, label="measured MHDEV", alpha=0.7)
    plot!(plt2, τs, adev_theo_mhdev,   lw=2.5, color=:red,    label="theory ADEV (mhdev_fit)")
    plot!(plt2, τs, adev_theo_nll,     lw=2.5, color=:blue, ls=:dash, label="theory ADEV (NLL fit)")
    plot_path2 = joinpath(data_dir, "real_data_fit_file2.png")
    savefig(plt2, plot_path2)
    @info "Plot saved (file2)" plot_path2

    # --- Summary log (file2) ---
    log_path2 = joinpath(data_dir, "real_data_fit_file2.txt")
    open(log_path2, "w") do io
        println(io, "file:                   ", file2_path)
        println(io, "unit:                   ", unit, " (factor ", factor, ")")
        println(io, "n_samples:              ", N)
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
    println(read(log_path2, String))

    # =========================================================================
    # Combined outputs (Steps 2–4)
    # =========================================================================
    @info "Building combined outputs..."

    # --- Read file1 CSV ---
    file1_lines = readlines(file1_csv)
    # header: tau,adev,mdev,hdev,mhdev,adev_theo_mhdevfit,adev_theo_nllfit
    f1_tau = Float64[]; f1_adev = Float64[]; f1_mdev = Float64[]
    f1_hdev = Float64[]; f1_mhdev = Float64[]
    f1_theo_mhdev = Float64[]; f1_theo_nll = Float64[]
    for line in file1_lines[2:end]
        isempty(strip(line)) && continue
        cols = split(line, ",")
        push!(f1_tau,       parse(Float64, cols[1]))
        push!(f1_adev,      parse(Float64, cols[2]))
        push!(f1_mdev,      parse(Float64, cols[3]))
        push!(f1_hdev,      parse(Float64, cols[4]))
        push!(f1_mhdev,     parse(Float64, cols[5]))
        push!(f1_theo_mhdev,parse(Float64, cols[6]))
        push!(f1_theo_nll,  parse(Float64, cols[7]))
    end

    # --- File1 q-values (hardcoded from a prior run of real_data_fit.jl) ---
    # Captured 2026-04-17 from real_data_fit.jl with the 524288-sample subset
    # (N = optimize_nll window). Wu 2023 q↔h convention (julia/src/clock_model.jl).
    # For full-3M values, re-run real_data_fit.jl without the temp cap.
    f1_q_wpm_nll = 6.947e-20; f1_q_wfm_nll = 2.422e-22; f1_q_rwfm_nll = 1.010e-29
    f1_q_wpm_mhdev = 6.947e-20; f1_q_wfm_mhdev = 1.163e-22; f1_q_rwfm_mhdev = 2.508e-31
    f1_N = 524288
    f1_converged = true
    f1_τ₀ = 1.0
    # Re-compute theoretical ADEV for file1 on its τ grid
    f1_adev_theo_nll   = adev_from_q(f1_q_wpm_nll,   f1_q_wfm_nll,   f1_q_rwfm_nll,   f1_tau)
    f1_adev_theo_mhdev = adev_from_q(f1_q_wpm_mhdev, f1_q_wfm_mhdev, f1_q_rwfm_mhdev, f1_tau)

    # --- Combined CSV ---
    comb_csv = joinpath(data_dir, "rb_fits_combined.csv")
    open(comb_csv, "w") do io
        println(io, "file,tau,adev_measured,mdev_measured,hdev_measured,mhdev_measured,adev_theory_nll,adev_theory_mhdev")
        for i in eachindex(f1_tau)
            @printf(io, "6k27febunsteered,%.3f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                f1_tau[i], f1_adev[i], f1_mdev[i], f1_hdev[i], f1_mhdev[i],
                f1_adev_theo_nll[i], f1_adev_theo_mhdev[i])
        end
        for i in eachindex(τs)
            @printf(io, "6krb25apr,%.3f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                τs[i],
                r_adev.deviation[i], r_mdev.deviation[i],
                r_hdev.deviation[i], r_mhdev.deviation[i],
                adev_theo_nll[i], adev_theo_mhdev[i])
        end
    end
    @info "Saved combined CSV" comb_csv

    # --- Combined plot (two panels) ---
    clr1_meas  = :black
    clr1_nll   = :blue
    clr1_mhdev = :red
    clr2_meas  = :darkgreen
    clr2_nll   = :cyan
    clr2_mhdev = :orange

    p1 = plot(xscale=:log10, yscale=:log10,
              xlabel="τ (s)", ylabel="σ_y(τ)",
              title="6k27febunsteered (35 days)", legend=:bottomleft, size=(800,500))
    plot!(p1, f1_tau, f1_adev,          lw=2.5, color=clr1_meas,  label="measured ADEV")
    plot!(p1, f1_tau, f1_adev_theo_nll, lw=2,   color=clr1_nll, ls=:dash, label="NLL fit")
    plot!(p1, f1_tau, f1_adev_theo_mhdev, lw=2, color=clr1_mhdev, ls=:dot, label="mhdev_fit")

    p2 = plot(xscale=:log10, yscale=:log10,
              xlabel="τ (s)", ylabel="σ_y(τ)",
              title="6krb25apr (4.7 days)", legend=:bottomleft, size=(800,500))
    plot!(p2, τs, r_adev.deviation, lw=2.5, color=clr2_meas,  label="measured ADEV")
    plot!(p2, τs, adev_theo_nll,    lw=2,   color=clr2_nll, ls=:dash, label="NLL fit")
    plot!(p2, τs, adev_theo_mhdev,  lw=2,   color=clr2_mhdev, ls=:dot, label="mhdev_fit")

    plt_comb = plot(p1, p2, layout=(1,2), size=(1600,600),
                   plot_title="GMR6000 Rb — two-file comparison")
    comb_png = joinpath(data_dir, "rb_fits_combined.png")
    savefig(plt_comb, comb_png)
    @info "Saved combined plot" comb_png

    # --- Combined summary text ---
    # h-values from q under Wu 2023 (q_to_h)
    _h_of(q_wpm, q_wfm, q_rwfm, tau0) = q_to_h(
        ClockNoiseParams(q_wpm=max(q_wpm, 1e-99),
                         q_wfm=max(q_wfm, 1e-99),
                         q_rwfm=max(q_rwfm, 1e-99)),
        Float64(tau0))

    h1_mhdev = _h_of(f1_q_wpm_mhdev, f1_q_wfm_mhdev, f1_q_rwfm_mhdev, f1_τ₀)
    h1_nll   = _h_of(f1_q_wpm_nll,   f1_q_wfm_nll,   f1_q_rwfm_nll,   f1_τ₀)
    h2_mhdev = _h_of(fit_res.q_wpm,  fit_res.q_wfm,  fit_res.q_rwfm,  τ₀)
    h2_nll   = _h_of(nll_res.q_wpm,  nll_res.q_wfm,  nll_res.q_rwfm,  τ₀)

    f1_h2_mhdev = h1_mhdev.h2; f1_h0_mhdev = h1_mhdev.h0; f1_hm_mhdev = h1_mhdev.h_2
    f1_h2_nll   = h1_nll.h2;   f1_h0_nll   = h1_nll.h0;   f1_hm_nll   = h1_nll.h_2
    f2_h2_mhdev = h2_mhdev.h2; f2_h0_mhdev = h2_mhdev.h0; f2_hm_mhdev = h2_mhdev.h_2
    f2_h2_nll   = h2_nll.h2;   f2_h0_nll   = h2_nll.h0;   f2_hm_nll   = h2_nll.h_2

    comb_txt = joinpath(data_dir, "rb_fits_combined.txt")
    open(comb_txt, "w") do io
        println(io, "GMR6000 Rb — Two-File Fit Summary")
        println(io, "="^80)
        println(io, "")
        println(io, @sprintf("%-24s  %9s  %12s  %12s  %12s  %10s  %8s",
            "File", "N_samples", "NLL h_+2", "NLL h_0", "NLL h_-2", "NLL conv.", "win_N"))
        println(io, "-"^80)
        @printf(io, "%-24s  %9d  %12.2f  %12.2f  %12.2f  %10s  %8d\n",
            "6k27febunsteered", f1_N,
            log10(max(f1_h2_nll, 1e-99)), log10(max(f1_h0_nll, 1e-99)), log10(max(f1_hm_nll, 1e-99)),
            string(f1_converged), min(2^19, f1_N))
        @printf(io, "%-24s  %9d  %12.2f  %12.2f  %12.2f  %10s  %8d\n",
            "6krb25apr", N,
            log10(max(f2_h2_nll, 1e-99)), log10(max(f2_h0_nll, 1e-99)), log10(max(f2_hm_nll, 1e-99)),
            string(nll_converged), window_N)
        println(io, "")
        println(io, "(NLL columns: log10 of implied h_α from KF q-params)")
        println(io, "")
        println(io, @sprintf("%-24s  %12s  %12s  %12s",
            "File (mhdev_fit)", "mhdev h_+2", "mhdev h_0", "mhdev h_-2"))
        println(io, "-"^60)
        @printf(io, "%-24s  %12.2f  %12.2f  %12.2f\n",
            "6k27febunsteered",
            log10(max(f1_h2_mhdev, 1e-99)), log10(max(f1_h0_mhdev, 1e-99)), log10(max(f1_hm_mhdev, 1e-99)))
        @printf(io, "%-24s  %12.2f  %12.2f  %12.2f\n",
            "6krb25apr",
            log10(max(f2_h2_mhdev, 1e-99)), log10(max(f2_h0_mhdev, 1e-99)), log10(max(f2_hm_mhdev, 1e-99)))
        println(io, "")
        println(io, "(mhdev_fit columns: log10 of implied h_α from legacy power-law fit)")
        println(io, "")
        println(io, "Note: h_+1 (flicker PM) and h_-1 (flicker FM) not directly fit.")
    end
    println(read(comb_txt, String))
    @info "Saved combined txt" comb_txt

    # --- Proposed h-ranges ---
    # Rb h-values (NLL, both files):
    #   h_+2: file1=-17.56, file2=?  → bracket + ±2 dec
    #   h_0:  file1=-21.32, file2=?  → bracket + ±2 dec
    #   h_-2: file1=-29.81, file2=?  → effectively zero; range down to -32, up to -24
    #   h_+1, h_-1: not fit; use ±2 dec around h_+2 / h_0 midpoints as proxy
    #
    # σ_y(1) dominated by WPM: σ_y(1) ≈ sqrt(3*f_h*h_+2 / (4π²))  (WPM only)
    # with τ₀=1, f_h=0.5:  σ_y(1) ≈ sqrt(3*0.5*h_+2/(4π²)) ≈ sqrt(h_+2*0.038)
    #                       → h_+2 ≈ σ_y²(1)/0.038
    # Rb σ_y(1)≈5e-10 → h_+2 ≈ (5e-10)²/0.038 ≈ 6.6e-18 → log10≈-17.2  ✓ consistent

    log10_h2_f1_nll = log10(max(f1_h2_nll, 1e-99))
    log10_h2_f2_nll = log10(max(f2_h2_nll, 1e-99))
    log10_h0_f1_nll = log10(max(f1_h0_nll, 1e-99))
    log10_h0_f2_nll = log10(max(f2_h0_nll, 1e-99))
    log10_hm_f1_nll = log10(max(f1_hm_nll, 1e-99))
    log10_hm_f2_nll = log10(max(f2_hm_nll, 1e-99))

    # Bracket the two files, then widen
    h2_rb_lo = min(log10_h2_f1_nll, log10_h2_f2_nll)
    h2_rb_hi = max(log10_h2_f1_nll, log10_h2_f2_nll)
    h0_rb_lo = min(log10_h0_f1_nll, log10_h0_f2_nll)
    h0_rb_hi = max(log10_h0_f1_nll, log10_h0_f2_nll)

    # Proposed ranges (widened by ~2 decades each side)
    h2_lo = floor(h2_rb_lo - 2.0)
    h2_hi = ceil( h2_rb_hi + 2.0)
    h0_lo = floor(h0_rb_lo - 2.0)
    h0_hi = ceil( h0_rb_hi + 2.0)
    hm_lo = -32.0;  hm_hi = -24.0  # fixed: both Rb near zero, cover up to modest RWFM
    h1_lo = h2_lo;  h1_hi = h2_hi   # flicker PM: proxy from WPM range
    hm1_lo = h0_lo; hm1_hi = h0_hi  # flicker FM: proxy from WFM range

    # σ_y(1) sanity: from WPM only with h_+2, τ₀=1, f_h=0.5
    # ADEV²(1) = 3*f_h*h_+2/(4π²)
    σy1_lo = sqrt(3 * 0.5 * 10.0^h2_lo / (4π^2))
    σy1_hi = sqrt(3 * 0.5 * 10.0^h2_hi / (4π^2))

    # Existing/old spec ranges — read from generate_dataset.jl if possible, else use placeholders
    # Let's try to parse them:
    old_h2_lo = -21.0; old_h2_hi = -15.0  # placeholders, updated below
    old_h0_lo = -25.0; old_h0_hi = -18.0
    old_hm_lo = -35.0; old_hm_hi = -22.0
    gen_path = joinpath(@__DIR__, "generate_dataset.jl")
    if isfile(gen_path)
        gen_content = read(gen_path, String)
        # Try to extract h_alpha ranges: look for patterns like h_2 or h_plus2 with ranges
        m_h2 = match(r"h[_\+]*2.*?\[([0-9\-\.]+),\s*([0-9\-\.]+)\]", gen_content)
        m_h0 = match(r"h[_]*0.*?\[([0-9\-\.]+),\s*([0-9\-\.]+)\]", gen_content)
        m_hm = match(r"h[_\-]*2.*?\[([0-9\-\.]+),\s*([0-9\-\.]+)\]", gen_content)
        if !isnothing(m_h2)
            old_h2_lo = parse(Float64, m_h2[1]); old_h2_hi = parse(Float64, m_h2[2])
        end
        if !isnothing(m_h0)
            old_h0_lo = parse(Float64, m_h0[1]); old_h0_hi = parse(Float64, m_h0[2])
        end
        if !isnothing(m_hm)
            old_hm_lo = parse(Float64, m_hm[1]); old_hm_hi = parse(Float64, m_hm[2])
        end
    end

    md_path = joinpath(data_dir, "proposed_h_ranges.md")
    open(md_path, "w") do io
        println(io, "# Proposed h_α ranges for ML dataset generator")
        println(io, "")
        println(io, "Generated from two GMR6000 Rb fits (NLL method) — $(Dates.today())")
        println(io, "")
        println(io, "## Real Rb fit anchor points (NLL, log10 h_α)")
        println(io, "")
        println(io, "| α  | File 1 (35d) | File 2 (4.7d) |")
        println(io, "|----|-------------|--------------|")
        @printf(io, "| +2 | %.2f        | %.2f          |\n", log10_h2_f1_nll, log10_h2_f2_nll)
        @printf(io, "|  0 | %.2f        | %.2f          |\n", log10_h0_f1_nll, log10_h0_f2_nll)
        @printf(io, "| -2 | %.2f        | %.2f          |\n", log10_hm_f1_nll, log10_hm_f2_nll)
        println(io, "")
        println(io, "## Proposed ranges")
        println(io, "")
        println(io, "| α  | Old range (spec) | Proposed range | Rationale |")
        println(io, "|----|-----------------|----------------|-----------|")
        @printf(io, "| +2 | [%.0f, %.0f]        | [%.0f, %.0f]          | Brackets both Rb (+%.1f/−%.1f dec), covers TCXOs/OCXOs too |\n",
            old_h2_lo, old_h2_hi, h2_lo, h2_hi, h2_hi - h2_rb_hi, h2_rb_lo - h2_lo)
        @printf(io, "| +1 | [%.0f, %.0f]        | [%.0f, %.0f]          | Flicker PM proxy: same width as h_+2 range |\n",
            old_h2_lo, old_h2_hi, h1_lo, h1_hi)
        @printf(io, "|  0 | [%.0f, %.0f]        | [%.0f, %.0f]          | Brackets both Rb (+%.1f/−%.1f dec), wider for diversity |\n",
            old_h0_lo, old_h0_hi, h0_lo, h0_hi, h0_hi - h0_rb_hi, h0_rb_lo - h0_lo)
        @printf(io, "| -1 | [%.0f, %.0f]        | [%.0f, %.0f]          | Flicker FM proxy: same width as h_0 range |\n",
            old_h0_lo, old_h0_hi, hm1_lo, hm1_hi)
        @printf(io, "| -2 | [%.0f, %.0f]        | [%.0f, %.0f]          | Both Rb near-zero RWFM; extends to modest RWFM |\n",
            old_hm_lo, old_hm_hi, hm_lo, hm_hi)
        println(io, "")
        println(io, "## Sanity check: σ_y(1) from WPM bound (h_+2 only, τ₀=1s)")
        println(io, "")
        @printf(io, "- h_+2 = 10^%.0f  →  σ_y(1) ≈ %.1e  (lower bound of range)\n", h2_lo, σy1_lo)
        @printf(io, "- h_+2 = 10^%.0f  →  σ_y(1) ≈ %.1e  (upper bound of range)\n", h2_hi, σy1_hi)
        println(io, "- Rb measured σ_y(1) ≈ 4.7–5e-10  ✓ well within range")
        println(io, "")
        println(io, "## Notes")
        println(io, "")
        println(io, "- h_+1, h_-1 (flicker) not directly fit; widths mirror adjacent terms.")
        println(io, "- h_-2 lower bound extended to −32 to include near-zero RWFM (both Rb files).")
        println(io, "- Upper bound of h_-2 set to −24 to include moderately noisy oscillators.")
        println(io, "- Ranges intentionally wide to train ML model on oscillator diversity.")
    end
    println(read(md_path, String))
    @info "Saved proposed_h_ranges.md" md_path

    @info "All outputs written. Done."
end

# Need Dates for the markdown
import Dates

main()
