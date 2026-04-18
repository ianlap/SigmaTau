"""
gen_crossval_data.jl — Generate cross-validation reference data (stdlib only).

Run from repo root:
    julia --project=julia julia/scripts/gen_crossval_data.jl

Outputs to matlab/tests/:
  crossval_phase_wpm.txt / wfm.txt / rwfm.txt  — N=1024 phase data, 17 digits
  crossval_results.txt                          — tab-delimited, 17-digit deviations
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using SigmaTau
using Printf
using Random

Random.seed!(42)

const N     = 1024
const TAU0  = 1.0
const MLIST = [1, 2, 4, 8, 16, 32, 64]

x_wpm  = randn(N)
x_wfm  = cumsum(randn(N)) .* TAU0
x_rwfm = cumsum(cumsum(randn(N))) .* TAU0^2

outdir = joinpath(@__DIR__, "../../matlab/tests")

# Write phase data at full double precision (17 significant digits)
for (label, x) in [("wpm", x_wpm), ("wfm", x_wfm), ("rwfm", x_rwfm)]
    open(joinpath(outdir, "crossval_phase_$(label).txt"), "w") do io
        for v in x
            @printf(io, "%.17e\n", v)
        end
    end
    println("Wrote crossval_phase_$(label).txt  (N=$(length(x)))")
end

devfns = [
    ("adev",     adev),
    ("mdev",     mdev),
    ("tdev",     tdev),
    ("hdev",     hdev),
    ("mhdev",    mhdev),
    ("ldev",     ldev),
    ("totdev",   totdev),
    ("mtotdev",  mtotdev),
    ("htotdev",  htotdev),
    ("mhtotdev", mhtotdev),
]

outfile = joinpath(outdir, "crossval_results.txt")
open(outfile, "w") do io
    println(io, "devname\tnoise\tm\ttau\tdeviation\tedf\talpha")
    for (devname, fn) in devfns
        for (label, x) in [("wpm", x_wpm), ("wfm", x_wfm), ("rwfm", x_rwfm)]
            r = fn(x, TAU0; m_list=MLIST)
            for k in eachindex(r.tau)
                @printf(io, "%s\t%s\t%d\t%.17e\t%.17e\t%.17e\t%d\n",
                        devname, label, MLIST[k],
                        r.tau[k], r.deviation[k], r.edf[k], r.alpha[k])
            end
        end
    end
end

println("Wrote crossval_results.txt")
println("Done.")
