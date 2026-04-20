# real_data_all_fits.jl — Per-window fits on the 6k27febunsteered Rb record.
#
# For each 524288-sample window (non-overlapping, contiguous from index 1),
# run:
#   (a) mhdev_fit  (three-region slope fit)
#   (b) optimize_nll (Kalman innovation NLL, warm-started from mhdev_fit)
#   (c) als_fit    (autocovariance least-squares)
# plus compute the empirical ADEV on an octave-spaced τ grid.
#
# Outputs (written to ml/data/):
#   real_per_window_fits.csv   — per-window q values for each method
#   real_per_window_adev.csv   — per-window (tau, adev) long-format table
#
# Usage:
#   julia --project=ml/dataset --threads=auto ml/dataset/real_data_all_fits.jl

using Pkg; Pkg.activate(@__DIR__)
using SigmaTau
using Printf, Statistics, Random

const REAL_PATH = joinpath(@__DIR__, "..", "..", "reference", "raw", "6k27febunsteered.txt")
const OUT_DIR   = joinpath(@__DIR__, "..", "data")
const WINDOW_N  = 524_288
const N_WIN     = 4
const TAU0      = 1.0

function load_phase_ns_to_seconds(path)
    ph = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        tok = split(s)
        length(tok) < 2 && continue
        try
            push!(ph, parse(Float64, tok[2]))
        catch; end
    end
    # Unit detection via adev(1)
    test_x = ph[1:min(65_536, length(ph))]
    ad1    = adev(test_x, TAU0).deviation[1]
    factor = ad1 > 1e-3 ? 1e-9 : (ad1 > 1e-6 ? 1e-6 : 1.0)
    return ph .* factor, factor
end

function fit_one_window(x::AbstractVector{<:Real})
    # --- empirical deviations on octave τ grid (log-spaced, capped at N/10)
    N       = length(x)
    m_max   = min(N ÷ 10, 1_000_000)
    m_list  = unique(round.(Int, exp10.(range(0, log10(m_max), length=30))))
    r_adev  = adev(x, TAU0; m_list=m_list)
    r_mhdev = mhdev(x, TAU0; m_list=m_list)

    τs    = r_adev.tau
    σadev = r_adev.deviation

    # --- MHDEV three-region fit (WPM / WFM / RWFM)
    τmh = r_mhdev.tau
    σmh = r_mhdev.deviation
    idx_wpm  = findall(τ -> 1.0  <= τ <= 10.0,    τmh)
    idx_wfm  = findall(τ -> 30.0 <= τ <= 1000.0,  τmh)
    idx_rwfm = findall(τ -> 1e4  <= τ <= 1e5,     τmh)
    regions  = [
        (noise_type=:wpm,  indices=idx_wpm),
        (noise_type=:wfm,  indices=idx_wfm),
        (noise_type=:rwfm, indices=idx_rwfm),
    ]
    mhf = mhdev_fit(τmh, σmh, regions)

    # --- NLL optimize (warm-started from MHDEV fit)
    # Note: on the real GMR6000 Rb record, the NLL under the 3-state clock model
    # systematically lands in a q_WFM-dominated basin for windows 1-3 — the model
    # lacks FFM (flicker FM) so NLL absorbs that spectral content into q_WFM. We
    # confirmed this by running multi-restart and optimize_qwpm=true; NLL prefers
    # the same basin regardless. MHDEV-fit / ALS give the physically sensible
    # three-slope decomposition.
    q0 = ClockNoiseParams(
        q_wpm  = mhf.q_wpm  > 0 ? mhf.q_wpm  : 1e-22,
        q_wfm  = mhf.q_wfm  > 0 ? mhf.q_wfm  : 1e-22,
        q_rwfm = mhf.q_rwfm > 0 ? mhf.q_rwfm : 1e-30,
    )
    nll = optimize_nll(x, TAU0;
                      noise_init = q0,
                      optimize_qwpm = false,
                      verbose = false,
                      max_iter = 1000)

    # --- ALS (same warm start). als_fit dispatches only on Vector{Float64}, so
    # materialize the view to a concrete vector before passing.
    xvec = Vector{Float64}(x)
    als = try
        als_fit(xvec, TAU0; noise_init = q0, verbose = false, max_iter = 5)
    catch err
        @warn "als_fit failed" err
        nothing
    end

    return (τs = τs, adev = σadev, mhdev_fit = mhf,
            nll = nll.noise, nll_conv = nll.converged,
            als = als)
end

function main()
    mkpath(OUT_DIR)
    ph, factor = load_phase_ns_to_seconds(REAL_PATH)
    @info "Loaded Rb record" n=length(ph) factor=factor

    @assert length(ph) ≥ N_WIN * WINDOW_N "record too short: $(length(ph)) < $(N_WIN*WINDOW_N)"

    # Accumulate results
    fit_rows  = Vector{NamedTuple}(undef, N_WIN)
    adev_rows = []  # (window, tau, adev)

    for i in 1:N_WIN
        rng = (1 + (i-1)*WINDOW_N) : (i*WINDOW_N)
        @info "window $i" n=length(rng)
        r = fit_one_window(view(ph, rng))
        fit_rows[i] = (window = i-1,
                       mhf_qwpm = r.mhdev_fit.q_wpm,
                       mhf_qwfm = r.mhdev_fit.q_wfm,
                       mhf_qrwfm = r.mhdev_fit.q_rwfm,
                       nll_qwpm = r.nll.q_wpm,
                       nll_qwfm = r.nll.q_wfm,
                       nll_qrwfm = r.nll.q_rwfm,
                       nll_converged = r.nll_conv,
                       als_qwpm = r.als === nothing ? NaN : r.als.q_wpm,
                       als_qwfm = r.als === nothing ? NaN : r.als.q_wfm,
                       als_qrwfm = r.als === nothing ? NaN : r.als.q_rwfm)
        for (τ, σ) in zip(r.τs, r.adev)
            push!(adev_rows, (window = i-1, tau = τ, adev = σ))
        end
        @info "fit $i done" mhf=fit_rows[i].mhf_qwpm,fit_rows[i].mhf_qwfm,fit_rows[i].mhf_qrwfm  nll=fit_rows[i].nll_qwpm,fit_rows[i].nll_qwfm,fit_rows[i].nll_qrwfm
    end

    # --- Write fits CSV
    out_fits = joinpath(OUT_DIR, "real_per_window_fits.csv")
    open(out_fits, "w") do io
        println(io, "window,mhf_qwpm,mhf_qwfm,mhf_qrwfm,",
                    "nll_qwpm,nll_qwfm,nll_qrwfm,nll_converged,",
                    "als_qwpm,als_qwfm,als_qrwfm")
        for r in fit_rows
            @printf(io, "%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d,%.6e,%.6e,%.6e\n",
                    r.window, r.mhf_qwpm, r.mhf_qwfm, r.mhf_qrwfm,
                    r.nll_qwpm, r.nll_qwfm, r.nll_qrwfm, r.nll_converged ? 1 : 0,
                    r.als_qwpm, r.als_qwfm, r.als_qrwfm)
        end
    end
    @info "wrote" out_fits

    out_adev = joinpath(OUT_DIR, "real_per_window_adev.csv")
    open(out_adev, "w") do io
        println(io, "window,tau,adev")
        for r in adev_rows
            @printf(io, "%d,%.6e,%.6e\n", r.window, r.tau, r.adev)
        end
    end
    @info "wrote" out_adev
end

main()
