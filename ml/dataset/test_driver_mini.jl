using Pkg; Pkg.activate(@__DIR__)
include("generate_dataset.jl")
using .DatasetGen

out = joinpath(@__DIR__, "tmp_mini.npz")
isfile(out) && rm(out)
DatasetGen.generate_dataset(out; n_samples=32, N=2^13, τ₀=1.0, resume=false)

using NPZ
d = NPZ.npzread(out)
@assert size(d["X"])        == (32, 196)
@assert size(d["y"])        == (32, 3)
@assert size(d["h_coeffs"]) == (32, 5)
@assert length(d["fpm_present"]) == 32
@assert length(d["taus"])        == 20
# feature_names written to companion .txt file (NPZ.jl does not support string arrays)
names_txt = out * ".feature_names.txt"
@assert isfile(names_txt)
fn_count = length(split(strip(read(names_txt, String)), "|"))
@assert fn_count == 196
println("mini driver produced ", out, " with ", size(d["X"],1), " samples.")
rm(out)
rm(names_txt)
