# validate.jl — Input validation and data preprocessing

"""
    validate_phase_data(x) → Vector{Float64}

Check that phase data is a finite real vector; return as a flat Float64 vector.
"""
function validate_phase_data(x::AbstractVector{<:Real})
    all(isfinite, x) || throw(ArgumentError("Phase data must be finite (no NaN or Inf)."))
    length(x) >= 4 || throw(ArgumentError("Phase data must have at least 4 points."))
    return vec(Float64.(x))
end

"""
    validate_tau0(tau0) → Float64

Check that τ₀ is a positive finite scalar.
"""
function validate_tau0(tau0::Real)
    (isfinite(tau0) && tau0 > 0) ||
        throw(ArgumentError("tau0 must be positive and finite, got $tau0."))
    return Float64(tau0)
end

"""
    detrend_quadratic(x)

Remove quadratic trend from data (used for phase data in noise_id).
"""
function detrend_quadratic(x::AbstractVector{T}) where T<:Real
    N = length(x)
    N < 3 && return copy(x)
    t = collect(T, 1:N)
    A = hcat(ones(T, N), t, t .^ 2)
    coeffs = A \ x
    return x - A * coeffs
end
