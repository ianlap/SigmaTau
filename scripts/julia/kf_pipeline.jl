#!/usr/bin/env julia
# kf_pipeline.jl — Kalman filter pipeline on a phase dataset (non-interactive).
#
# Non-interactive port of matlab/legacy/kflab/main_kf_pipeline_unified.m.
# Stages:
#   1. Load phase data + tau0 from reference/<dataset>.txt.
#   2. Compute MHDEV for noise characterisation.
#   3. Obtain initial q values — prefers an interactive fit from
#      mhdev_fit_interactive.py (results/<dataset>/kf/mhdev_fit.csv); falls
#      back to SigmaTau.mhdev_fit on FIT_REGIONS declared below.
#   4. Run the KF (no steering, diffuse P0) with fitted q — initial test.
#   5. Refine q via innovation NLL (optimize_kf / Nelder-Mead).
#   6. Run the KF again with optimised q — final test.
#   7. Compare: q values, innovation RMS, NLL, multi-step prediction RMS,
#      sample the NLL surface around the optimum.
#
# Run from repo root:
#   julia --threads=auto --project=julia scripts/julia/kf_pipeline.jl <dataset>
# e.g.
#   julia --threads=auto --project=julia scripts/julia/kf_pipeline.jl 6krb25apr

using Dates
using DelimitedFiles
using Printf
using Statistics

const HERE        = @__DIR__
const REPO        = abspath(joinpath(HERE, "..", ".."))
const SEC_PER_DAY = 86400.0

using Pkg
Pkg.activate(joinpath(REPO, "julia"); io = devnull)
using SigmaTau

isempty(ARGS) && error("usage: kf_pipeline.jl <dataset>  (basename in reference/)")
const DATASET = ARGS[1]
const DATA    = joinpath(REPO, "reference", "$(DATASET).txt")
const OUT_DIR = joinpath(HERE, "results", DATASET, "kf")

# ── User configuration ───────────────────────────────────────────────────────
# FIT_REGIONS is only used if mhdev_fit.csv is absent. For the interactive
# flow (recommended), run mhdev_fit_interactive.py first and leave these alone.
const NSTATES     = 3          # 2 = phase+freq, 3 = phase+freq+drift
const FIT_REGIONS = [
    (:wpm,  1:6),
    (:rwfm, 11:12),
]
const HORIZONS_TO_REPORT = (1, 10, 60, 300, 3_600, 10_000, 100_000)
const MATURITY_FRACTION  = 0.2

# NLL-surface grid for the diagnostic plot: span ±SURFACE_DECADES around the
# optimum in (q_wfm, q_rwfm) at fixed q_wpm. SURFACE_NGRID × SURFACE_NGRID
# evaluations of _kf_nll (~0.03s each at N=50k; scales ~linearly with N).
const SURFACE_NGRID    = 9
const SURFACE_DECADES  = 3.0

# ── Stage 1: load data ───────────────────────────────────────────────────────
function say(msg)
    println(stdout, "[", Dates.format(now(), "HH:MM:SS"), "] ", msg)
    flush(stdout)
end

isfile(DATA) || error("missing $DATA; check the dataset name")

say("dataset = $DATASET")
say("reading $DATA ...")
raw   = readdlm(DATA)
N     = size(raw, 1)
mjd   = Float64.(raw[1:N, 1])
x     = Float64.(raw[1:N, 2])
tau0  = median(diff(mjd)) * SEC_PER_DAY
say(@sprintf("  loaded %d samples  tau0=%.6fs  record=%.3f days",
             N, tau0, (N - 1) * tau0 / SEC_PER_DAY))

# ── Stage 2: MHDEV ───────────────────────────────────────────────────────────
say("computing MHDEV ...")
mh    = mhdev(x, tau0)
tau_m = mh.tau
sig_m = mh.deviation
var_m = sig_m .^ 2
alpha = mh.alpha

println("\n=== MHDEV ===")
@printf("%5s %12s %14s %14s %10s %8s\n",
        "idx", "tau[s]", "sigma", "sigma^2", "slope(σ)", "α_id")
for i in eachindex(tau_m)
    slope = if i < length(tau_m) && sig_m[i] > 0 && sig_m[i+1] > 0
        @sprintf("%+6.2f", log(sig_m[i+1] / sig_m[i]) / log(tau_m[i+1] / tau_m[i]))
    else
        "  -   "
    end
    @printf("%5d %12.4g %14.6g %14.6g %10s %8d\n",
            i, tau_m[i], sig_m[i], var_m[i], slope, alpha[i])
end

# ── Stage 3: initial q values ────────────────────────────────────────────────
# Preferred: results/<dataset>/kf/mhdev_fit.csv from the interactive Python
# fitter. Fallback: SigmaTau.mhdev_fit on FIT_REGIONS above.
const FIT_CSV = joinpath(OUT_DIR, "mhdev_fit.csv")
q_fit = Dict(:wpm => 0.0, :wfm => 0.0, :rwfm => 0.0)

if isfile(FIT_CSV)
    println("\n=== Stage 3: loading interactive fit from $(basename(FIT_CSV)) ===")
    raw_fit = readdlm(FIT_CSV, ','; skipstart = 1)
    for i in 1:size(raw_fit, 1)
        key, val = String(raw_fit[i, 1]), Float64(raw_fit[i, 2])
        key == "q_wpm"  && (q_fit[:wpm]  = val)
        key == "q_wfm"  && (q_fit[:wfm]  = val)
        key == "q_rwfm" && (q_fit[:rwfm] = val)
    end
    @printf("  loaded: q_wpm=%.4e  q_wfm=%.4e  q_rwfm=%.4e\n",
            q_fit[:wpm], q_fit[:wfm], q_fit[:rwfm])
else
    println("\n=== Stage 3: SigmaTau.mhdev_fit on FIT_REGIONS ===")
    ci_matrix = [mh.ci[i, j] for i in eachindex(tau_m), j in 1:2]
    fit = mhdev_fit(tau_m, sig_m, FIT_REGIONS;
                    ci = ci_matrix, weight_method = :conservative)
    q_fit[:wpm]  = fit.q_wpm
    q_fit[:wfm]  = fit.q_wfm
    q_fit[:rwfm] = fit.q_rwfm
    for reg in fit.regions
        @printf("  fit %-5s  idx=%-8s  value=%.4e  ±%.2e%s\n",
                reg.noise_type, string(reg.indices), reg.value, reg.value_std,
                reg.skipped ? "  [skipped]" : "")
    end
end

# Fallback defaults (scaled to R) if a component wasn't fitted.
q_wpm0  = q_fit[:wpm]  > 0 ? q_fit[:wpm]  : (isempty(filter(>(0), var_m)) ? 1.0 : var_m[1])
q_wfm0  = q_fit[:wfm]  > 0 ? q_fit[:wfm]  : q_wpm0 * 1e-4
q_rwfm0 = q_fit[:rwfm] > 0 ? q_fit[:rwfm] : q_wpm0 * 1e-8
@printf("\n  initial q_wpm  = %.4e\n", q_wpm0)
@printf("  initial q_wfm  = %.4e\n", q_wfm0)
@printf("  initial q_rwfm = %.4e\n", q_rwfm0)

# ── Stage 4: initial KF ──────────────────────────────────────────────────────
maturity = max(NSTATES * 10, round(Int, MATURITY_FRACTION * N))
max_hor  = N - maturity - 1

function run_pipeline_kf(q_wpm, q_wfm, q_rwfm)
    kf_cfg = KalmanConfig(
        q_wpm   = q_wpm,
        q_wfm   = q_wfm,
        q_rwfm  = q_rwfm,
        R       = q_wpm,
        g_p     = 0.0, g_i = 0.0, g_d = 0.0,
        P0      = 1e6,
        nstates = NSTATES,
        tau     = tau0,
    )
    pred_cfg = PredictConfig(maturity = maturity, max_horizon = max_hor)
    return kf_predict(x, tau0, kf_cfg, pred_cfg)
end

say("running initial KF (fitted q) ...")
t_init  = @elapsed res0 = run_pipeline_kf(q_wpm0, q_wfm0, q_rwfm0)
kf0     = res0.kf_result
innov0  = kf0.innovations[maturity:end]   # exclude warm-up
stats0  = (rms  = sqrt(mean(innov0 .^ 2)),
           std  = std(innov0),
           mean = mean(innov0))
say(@sprintf("  initial KF: %.2fs  innov RMS=%.4g  std=%.4g",
             t_init, stats0.rms, stats0.std))

# ── Stage 5: NLL optimisation ────────────────────────────────────────────────
say("optimising q via NLL ...")
opt_cfg = OptimizeConfig(
    q_wpm   = q_wpm0,
    q_wfm   = q_wfm0,
    q_rwfm  = q_rwfm0,
    nstates = NSTATES,
    tau     = tau0,
    verbose = true,
)
t_opt = @elapsed opt = optimize_kf(x, opt_cfg)
say(@sprintf("  optimiser: %.2fs  %d NLL evals  converged=%s",
             t_opt, opt.n_evals, opt.converged))

# ── Stage 6: optimised KF ────────────────────────────────────────────────────
say("running optimised KF ...")
t_final = @elapsed res1 = run_pipeline_kf(opt.q_wpm, opt.q_wfm, opt.q_rwfm)
kf1     = res1.kf_result
innov1  = kf1.innovations[maturity:end]
stats1  = (rms  = sqrt(mean(innov1 .^ 2)),
           std  = std(innov1),
           mean = mean(innov1))
say(@sprintf("  optimised KF: %.2fs  innov RMS=%.4g  std=%.4g",
             t_final, stats1.rms, stats1.std))

# ── Stage 7: comparison ──────────────────────────────────────────────────────
println("\n=== Comparison ===")
@printf("%-14s %18s %18s\n", "parameter", "fitted", "NLL-optimal")
@printf("%-14s %18.4e %18.4e\n", "q_wpm  (R)",  q_wpm0,  opt.q_wpm)
@printf("%-14s %18.4e %18.4e\n", "q_wfm",       q_wfm0,  opt.q_wfm)
@printf("%-14s %18.4e %18.4e\n", "q_rwfm",      q_rwfm0, opt.q_rwfm)
println()
@printf("%-14s %18.4g %18.4g\n", "innov RMS",  stats0.rms,  stats1.rms)
@printf("%-14s %18.4g %18.4g\n", "innov std",  stats0.std,  stats1.std)
@printf("%-14s %18.4g %18.4g\n", "innov mean", stats0.mean, stats1.mean)
@printf("%-14s %18s %18.6f\n",    "NLL (opt)",  "—", opt.nll)

println("\n=== Multi-step prediction RMS error ===")
@printf("%-14s %18s %18s\n", "horizon[s]", "fitted", "NLL-optimal")
for h_samp in HORIZONS_TO_REPORT
    h_samp > max_hor && continue
    r0h = res0.rms_error[h_samp]
    r1h = res1.rms_error[h_samp]
    @printf("%-14.4g %18.4g %18.4g\n", h_samp * tau0, r0h, r1h)
end

# ── Persist results ──────────────────────────────────────────────────────────
mkpath(OUT_DIR)

open(joinpath(OUT_DIR, "kf_pipeline_summary.csv"), "w") do io
    println(io, "stage,q_wpm,q_wfm,q_rwfm,innov_rms,innov_std,innov_mean,nll,wall_s")
    @printf(io, "fitted,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,NaN,%.6f\n",
            q_wpm0, q_wfm0, q_rwfm0,
            stats0.rms, stats0.std, stats0.mean, t_init)
    @printf(io, "optimal,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.6f\n",
            opt.q_wpm, opt.q_wfm, opt.q_rwfm,
            stats1.rms, stats1.std, stats1.mean, opt.nll, t_final + t_opt)
end

open(joinpath(OUT_DIR, "kf_pipeline_prediction.csv"), "w") do io
    println(io, "horizon_samples,horizon_s,rms_fitted,rms_optimal,n_samples")
    for h in eachindex(res0.rms_error)
        @printf(io, "%d,%.9g,%.9g,%.9g,%d\n",
                h, h * tau0, res0.rms_error[h], res1.rms_error[h], res0.n_samples[h])
    end
end

# NLL surface on (q_wfm, q_rwfm) around the optimum (q_wpm fixed).
say(@sprintf("sampling NLL surface on %d×%d grid around optimum ...",
             SURFACE_NGRID, SURFACE_NGRID))
surf_cfg = OptimizeConfig(
    q_wpm = opt.q_wpm, q_wfm = opt.q_wfm, q_rwfm = opt.q_rwfm,
    nstates = NSTATES, tau = tau0, verbose = false,
)
q_wfm_grid  = 10.0 .^ range(log10(opt.q_wfm)  - SURFACE_DECADES,
                             log10(opt.q_wfm)  + SURFACE_DECADES,
                             length = SURFACE_NGRID)
q_rwfm_grid = 10.0 .^ range(log10(opt.q_rwfm) - SURFACE_DECADES,
                             log10(opt.q_rwfm) + SURFACE_DECADES,
                             length = SURFACE_NGRID)
t_surf = @elapsed begin
    open(joinpath(OUT_DIR, "kf_nll_surface.csv"), "w") do io
        println(io, "q_wfm,q_rwfm,nll")
        for qw in q_wfm_grid, qr in q_rwfm_grid
            nll_val = SigmaTau._kf_nll([log10(qw), log10(qr)], x, surf_cfg)
            @printf(io, "%.9g,%.9g,%.9g\n", qw, qr, nll_val)
        end
    end
end
say(@sprintf("  surface: %.2fs  (%d evals)", t_surf, SURFACE_NGRID^2))

say("wrote kf_pipeline_summary.csv + kf_pipeline_prediction.csv + " *
    "kf_nll_surface.csv to $OUT_DIR")
