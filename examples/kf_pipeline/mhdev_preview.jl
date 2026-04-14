#!/usr/bin/env julia
# mhdev_preview.jl — compute MHDEV on a dataset and write a preview CSV so
# plot_mhdev_preview.py / mhdev_fit_interactive.py can display the curve
# with index labels and fit its components.
#
# Usage (from repo root):
#   julia --threads=auto examples/kf_pipeline/mhdev_preview.jl <dataset>
# where <dataset> is the basename (no .txt) of a file in reference/, e.g.
#   julia --threads=auto examples/kf_pipeline/mhdev_preview.jl 6krb25apr
#   julia --threads=auto examples/kf_pipeline/mhdev_preview.jl 6k27febunsteered

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

function say(msg)
    println(stdout, "[", Dates.format(now(), "HH:MM:SS"), "] ", msg)
    flush(stdout)
end

isempty(ARGS) && error("usage: mhdev_preview.jl <dataset>  (basename in reference/)")
const DATASET  = ARGS[1]
const DATA     = joinpath(REPO, "reference", "$(DATASET).txt")
const OUT_DIR  = joinpath(HERE, "results", DATASET, "kf")
mkpath(OUT_DIR)

isfile(DATA) || error("missing $(DATA); check the dataset name")

say("dataset = $DATASET")
say("reading $DATA ...")
raw   = readdlm(DATA)
N     = size(raw, 1)
mjd   = Float64.(raw[1:N, 1])
x     = Float64.(raw[1:N, 2])
tau0  = median(diff(mjd)) * SEC_PER_DAY
say(@sprintf("  loaded %d samples  tau0=%.6fs  record=%.3f days",
             N, tau0, (N - 1) * tau0 / SEC_PER_DAY))

say("computing MHDEV ...")
t_mh = @elapsed mh = mhdev(x, tau0)
say(@sprintf("  done in %.2fs", t_mh))

tau_m = mh.tau
sig_m = mh.deviation
var_m = sig_m .^ 2
alpha = mh.alpha

println("\n=== MHDEV ($DATASET) ===")
@printf("%5s %12s %14s %14s %10s %8s\n",
        "idx", "tau[s]", "sigma", "sigma^2", "slope(σ)", "α_id")
for i in eachindex(tau_m)
    slope_str = if i < length(tau_m) && sig_m[i] > 0 && sig_m[i+1] > 0
        @sprintf("%+6.2f", log(sig_m[i+1] / sig_m[i]) / log(tau_m[i+1] / tau_m[i]))
    else
        "  -   "
    end
    @printf("%5d %12.4g %14.6g %14.6g %10s %8d\n",
            i, tau_m[i], sig_m[i], var_m[i], slope_str, alpha[i])
end

out_csv = joinpath(OUT_DIR, "mhdev_preview.csv")
open(out_csv, "w") do io
    # ci_lo / ci_hi included so mhdev_fit_interactive.py can use legacy
    # ci2weights-style weighting when fitting (currently defaults to equal).
    println(io, "idx,tau,sigma,sigma_sq,slope,alpha_id,ci_lo,ci_hi")
    for i in eachindex(tau_m)
        slope = (i < length(tau_m) && sig_m[i] > 0 && sig_m[i+1] > 0) ?
                log(sig_m[i+1] / sig_m[i]) / log(tau_m[i+1] / tau_m[i]) :
                NaN
        @printf(io, "%d,%.9g,%.9g,%.9g,%.6f,%d,%.9g,%.9g\n",
                i, tau_m[i], sig_m[i], var_m[i], slope, alpha[i],
                mh.ci[i, 1], mh.ci[i, 2])
    end
end
say("wrote $out_csv")
