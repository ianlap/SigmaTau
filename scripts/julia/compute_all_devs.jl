#!/usr/bin/env julia
# compute_all_devs.jl — Computes all 10 SigmaTau deviations from a text file.
#
# Usage:
#   julia --project=julia scripts/julia/compute_all_devs.jl <path_to_data.txt> [tau0]
#
# Arguments:
#   path_to_data.txt - Phase data file (one value per line)
#   tau0 - (optional) sampling interval in seconds, default = 1.0

using SigmaTau
using DelimitedFiles
using Printf
using Dates

if length(ARGS) < 1
    println("Usage: julia compute_all_devs.jl <path_to_data.txt> [tau0]")
    exit(1)
end

filename = ARGS[1]
tau0     = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0

if !isfile(filename)
    println("Error: File '$filename' not found.")
    exit(1)
end

"""Print a timestamped status line and flush immediately."""
function log_msg(msg::AbstractString)
    ts = Dates.format(now(), "HH:MM:SS")
    println(stdout, "[$ts] $msg")
    flush(stdout)
end

log_msg("SigmaTau: Compute All Deviations")
log_msg("-------------------------------")
log_msg("Loading $filename ...")
raw = readdlm(filename)
x   = vec(raw[:, 1])  # Ensure it is a vector
N   = length(x)
log_msg("  Loaded $N phase samples  (tau0 = $tau0 s)")

# ── Compute each deviation ───────────────────────────────────────────────────

dev_funcs = [
    ("ADEV",  adev), ("MDEV",  mdev), ("TDEV",  tdev),
    ("HDEV",  hdev), ("MHDEV", mhdev), ("LDEV",  ldev),
    ("TOT",   totdev), ("MTOT",  mtotdev), ("HTOT",  htotdev),
    ("MHTOT", mhtotdev)
]

# Create output folder
out_folder = filename * "_devs"
mkpath(out_folder)
log_msg("Saving results to: $out_folder")

results = []

for (name, fn) in dev_funcs
    @printf("  %-6s ... ", name)
    
    t0 = time()
    res = fn(x, tau0)
    dt = time() - t0
    
    # Save to CSV
    csv_path = joinpath(out_folder, "$(lowercase(name)).csv")
    open(csv_path, "w") do io
        println(io, "tau,deviation,alpha,edf,ci_lo,ci_hi,neff")
        for i in 1:length(res.tau)
            @printf(io, "%.9g,%.9e,%d,%.9e,%.9e,%.9e,%d\n", 
                    res.tau[i], res.deviation[i], res.alpha[i], 
                    res.edf[i], res.ci[i, 1], res.ci[i, 2], res.neff[i])
        end
    end
    @printf("done (%.2fs)\n", dt)
    push!(results, (name, res))
end

# Create summary CSV
log_msg("Creating summary.csv ...")
open(joinpath(out_folder, "summary.csv"), "w") do io
    # Header: tau, alpha, dev1, dev2, ...
    print(io, "tau,alpha")
    for (name, _) in results
        print(io, ",", lowercase(name))
    end
    println(io)
    
    # Use first result's tau as grid
    first_res = results[1][2]
    for i in 1:length(first_res.tau)
        @printf(io, "%.9g,%d", first_res.tau[i], first_res.alpha[i])
        for (_, res) in results
            if i <= length(res.tau)
                @printf(io, ",%.9e", res.deviation[i])
            else
                print(io, ",NaN")
            end
        end
        println(io)
    end
end

log_msg("\nAll deviations computed successfully.")
