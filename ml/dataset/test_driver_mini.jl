using Pkg; Pkg.activate(@__DIR__)
include("generate_dataset.jl")
using .DatasetGen

out = joinpath(@__DIR__, "tmp_mini.h5")
isfile(out) && rm(out)
DatasetGen.generate_dataset(out; n_samples=32, N=2^13, τ₀=1.0, resume=false)

using HDF5
h5open(out, "r") do f
    @assert size(f["features/X"][])        == (32, 196)
    @assert size(f["labels/q_log10"][])    == (32, 3)
    @assert size(f["labels/h_log10"][])    == (32, 5)
    @assert length(f["labels/fpm_present"][]) == 32
    @assert length(f["meta/taus"][])          == 20
    @assert length(f["meta/feature_names"][]) == 196
end
println("mini driver produced ", out, " with 32 samples.")
rm(out)
