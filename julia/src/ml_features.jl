# ml_features.jl — Feature extraction for the ML dataset.
#
# Canonical τ grid shared by generator, feature extractor, and Python loader.
# Computes 196 features per sample: 80 raw σ values + 76 adjacent-τ slopes
# + 40 variance ratios (MVAR/AVAR and MHVAR/HVAR at each τ).

const _N_POINTS_DEFAULT   = 131_072       # 2^17
const _SAFETY_FACTOR       = 10             # τ_max = N / SAFETY_FACTOR

# 20 log-spaced m values from 1 to floor(N/SAFETY_FACTOR).  For N=131072, m_max=13107.
# After integer rounding these 20 values are already unique.
const CANONICAL_M_LIST   = Int[
    1, 2, 3, 4, 7, 12, 19, 31, 51, 83,
    136, 222, 364, 596, 976, 1597, 2614, 4279, 7003, 11461
]
const CANONICAL_TAU_GRID = Float64.(CANONICAL_M_LIST)

# Feature names, ordered as: 80 raw (4 stats × 20 τ), 76 slopes, 40 ratios.
const _STATS  = ("adev", "mdev", "hdev", "mhdev")
const FEATURE_NAMES = let
    names = String[]
    for stat in _STATS, m in CANONICAL_M_LIST
        push!(names, "raw_$(stat)_m$(m)")
    end
    for stat in _STATS, i in 1:19
        push!(names, "slope_$(stat)_m$(CANONICAL_M_LIST[i])_m$(CANONICAL_M_LIST[i+1])")
    end
    for m in CANONICAL_M_LIST
        push!(names, "ratio_mvar_avar_m$(m)")
    end
    for m in CANONICAL_M_LIST
        push!(names, "ratio_mhvar_hvar_m$(m)")
    end
    names
end
@assert length(FEATURE_NAMES) == 196

_safe_log10(σ::Real) = σ > 0 ? log10(σ) : NaN

"""
    compute_feature_vector(x::AbstractVector{<:Real}, τ₀::Real) → Vector{Float64}

Compute the 196-feature vector for a phase time series `x`.  NaN values
propagate through slopes and ratios (column-median imputation happens in Python).
"""
function compute_feature_vector(x::AbstractVector{<:Real}, τ₀::Real)
    m_list = CANONICAL_M_LIST
    # Compute all four deviations on the shared m_list
    r_adev  = adev( x, τ₀; m_list)
    r_mdev  = mdev( x, τ₀; m_list)
    r_hdev  = hdev( x, τ₀; m_list)
    r_mhdev = mhdev(x, τ₀; m_list)

    # --- Raw log10(σ) features: 80 values ---
    σ_adev  = r_adev.deviation
    σ_mdev  = r_mdev.deviation
    σ_hdev  = r_hdev.deviation
    σ_mhdev = r_mhdev.deviation

    v = Float64[]
    append!(v, _safe_log10.(σ_adev))
    append!(v, _safe_log10.(σ_mdev))
    append!(v, _safe_log10.(σ_hdev))
    append!(v, _safe_log10.(σ_mhdev))

    # --- Slope features: 76 values (4 stats × 19 adjacent pairs) ---
    τ = CANONICAL_TAU_GRID
    for σ in (σ_adev, σ_mdev, σ_hdev, σ_mhdev)
        for i in 1:19
            lτ1, lτ2 = log10(τ[i]), log10(τ[i+1])
            lσ1, lσ2 = _safe_log10(σ[i]), _safe_log10(σ[i+1])
            push!(v, (lσ2 - lσ1) / (lτ2 - lτ1))
        end
    end

    # --- Ratio features: 40 values (MVAR/AVAR, MHVAR/HVAR at each τ) ---
    for i in 1:20
        push!(v, (σ_mdev[i]  / σ_adev[i])^2)
    end
    for i in 1:20
        push!(v, (σ_mhdev[i] / σ_hdev[i])^2)
    end

    @assert length(v) == 196
    return v
end
