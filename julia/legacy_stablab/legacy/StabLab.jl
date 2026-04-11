module StabLab

using Statistics
using LinearAlgebra

# Export main deviation functions
export adev, mdev, mhdev, hdev, mhtotdev, tdev, ldev, totdev, mtotdev, htotdev

# Export time interval error functions
export tie, mtie, pdev, theo1

# Export helper functions
export noise_id, timmer_koenig_from_psd, clock_noise, compute_ci, unpack_result

# Export plotting functions
export stabplot, load_phase_data, print_results_table, stability_report

# Export data types
export DeviationResult

# Core data structures
"""
    DeviationResult{T}

Result structure containing deviation analysis output.

# Fields
- `tau::Vector{T}`: Averaging times τ = m·τ₀ (seconds)
- `deviation::Vector{T}`: Deviation values
- `edf::Vector{T}`: Equivalent degrees of freedom
- `ci::Matrix{T}`: Confidence intervals [lower, upper]
- `alpha::Vector{T}`: Noise type exponents
- `neff::Vector{Int}`: Effective number of samples
- `tau0::T`: Sampling interval (seconds)
- `N::Int`: Original data length
- `method::String`: Deviation type identifier
- `confidence::T`: Confidence level used
"""
struct DeviationResult{T<:Real}
    tau::Vector{T}
    deviation::Vector{T}
    edf::Vector{T}
    ci::Matrix{T}
    alpha::Vector{Int}
    neff::Vector{Int}
    tau0::T
    N::Int
    method::String
    confidence::T
end

# Include source files
include("core.jl")
include("noise.jl")
include("confidence.jl")
include("deviations.jl")
include("time_error.jl")
include("plotting.jl")

end