# compute_sigmatau_deviations.jl
# Compute all 10 SigmaTau deviations on stable32gen.DAT phase data and dump CSV.

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "..", "julia"))
using SigmaTau
using Printf

const DAT_PATH = joinpath(@__DIR__, "stable32gen.DAT")
const OUT_PATH = joinpath(@__DIR__, "sigmatau_results.csv")

function load_phase(path)
    x = Float64[]
    in_data = false
    for line in eachline(path)
        if startswith(strip(line), "# Header End")
            in_data = true
            continue
        end
        if in_data
            s = strip(line)
            isempty(s) && continue
            push!(x, parse(Float64, s))
        end
    end
    return x
end

const DEVS = (
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
)

function main()
    x = load_phase(DAT_PATH)
    println("Loaded $(length(x)) phase points from ", basename(DAT_PATH))
    τ₀ = 1.0

    open(OUT_PATH, "w") do io
        println(io, "dev,m,tau,sigma,ci_lo,ci_hi,alpha,edf,neff")
        for (name, fn) in DEVS
            print("  computing $name ... ")
            res = fn(x, τ₀)
            ci_lo = res.ci[:, 1]
            ci_hi = res.ci[:, 2]
            # m = tau / tau0 (integer averaging factor)
            ms = round.(Int, res.tau ./ τ₀)
            for i in eachindex(res.tau)
                @printf(io, "%s,%d,%.6e,%.6e,%.6e,%.6e,%d,%.6e,%d\n",
                        name, ms[i], res.tau[i], res.deviation[i],
                        ci_lo[i], ci_hi[i], res.alpha[i],
                        res.edf[i], res.neff[i])
            end
            println("done ($(length(res.tau)) points)")
        end
    end
    println("\nWrote ", OUT_PATH)
end

main()
