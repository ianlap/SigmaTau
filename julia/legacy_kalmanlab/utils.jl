# utils.jl - Utility functions for KalmanLab

"""
    initialize!(kf::KalmanFilter, data::Vector{T}, n_fit::Int=100) where T

Initialize Kalman filter state and covariance using least squares fit to first n_fit data points.

The function fits a polynomial model to the initial data based on the number of states:
- 2-state: Linear fit (phase, frequency)
- 3-state: Quadratic fit (phase, frequency, drift)  
- 5-state: Quadratic + sinusoidal fit (phase, frequency, drift, diurnal terms)

# Arguments
- `kf::KalmanFilter`: Filter to initialize (x0 and P0 are modified in-place)
- `data::Vector{T}`: Raw phase error data (deviation from reference)
- `n_fit::Int=100`: Number of initial points to use for fitting

# Returns
- `Vector{T}`: Trimmed data with first n_fit points removed

# Example
```julia
kf = KalmanFilter(g_p=0.1, ..., nstates=3, tau=1.0, x0=zeros(3), P0=1e6)
data_trimmed = initialize!(kf, phase_data, 100)  # Fit first 100 points
results = run(kf, data_trimmed)  # Filter remaining data
```

# Mathematical Model
For 3-state case: `phase[k] = x₀ + x₁*t + 0.5*x₂*t²`
where `t = (k-1) * tau` and k = 1, 2, ..., n_fit
"""
function initialize!(kf::KalmanFilter{T}, data::Vector{T}, n_fit::Int=100) where T
    
    N = length(data)
    
    # Validate inputs
    if n_fit >= N
        error("n_fit ($n_fit) must be less than data length ($N)")
    end
    
    if n_fit < kf.nstates
        error("n_fit ($n_fit) must be >= nstates ($(kf.nstates)) for unique solution")
    end
    
    # Time vector for fitting (relative to first sample)
    t = T.((0:n_fit-1) * kf.tau)
    y = data[1:n_fit]
    
    # Build design matrix A based on number of states
    A = build_design_matrix(t, kf.nstates, T(86400.0))  # Default diurnal period
    
    # Least squares solution: x = (A'A)⁻¹A'y
    try
        coeffs = A \ y  # More numerically stable than (A'A)⁻¹A'y
        
        # Update initial state
        kf.x0 = coeffs[1:kf.nstates]
        
        # Compute covariance from inverse Hessian: P = σ²(A'A)⁻¹
        y_pred = A * coeffs
        residuals = y - y_pred
        P_matrix = var(residuals) * inv(A' * A)
        
        # Always set P0 to the full covariance matrix
        kf.P0 = P_matrix
        
        # Return trimmed data (exclude fitted points)
        return data[n_fit+1:end]
        
    catch e
        @warn "Least squares initialization failed: $e. Using original x0 and P0."
        return data[n_fit+1:end]
    end
end

"""
    build_design_matrix(t::Vector{T}, nstates::Int, period::T=T(86400.0)) where T

Build design matrix A for least squares fitting based on clock model.

# Arguments
- `t::Vector{T}`: Time vector (seconds relative to start)
- `nstates::Int`: Number of states (2, 3, or 5)
- `period::T`: Period for diurnal terms (seconds, default: 1 day)

# Returns
- `Matrix{T}`: Design matrix where each row corresponds to one time point

# Models
- 2-state: A = [1 t]
- 3-state: A = [1 t t²/2]  
- 5-state: A = [1 t t²/2 sin(2πt/T) cos(2πt/T)]
"""
function build_design_matrix(t::Vector{T}, nstates::Int, period::T=T(86400.0)) where T
    
    n = length(t)
    A = Matrix{T}(undef, n, nstates)
    
    # Column 1: Phase (constant term)
    A[:, 1] .= one(T)
    
    if nstates >= 2
        # Column 2: Frequency (linear term)
        A[:, 2] = t
    end
    
    if nstates >= 3
        # Column 3: Drift (quadratic term / 2)
        A[:, 3] = t.^2 ./ 2
    end
    
    if nstates == 5
        # Columns 4-5: Diurnal terms
        ω = 2π / period  # Angular frequency for diurnal variation
        A[:, 4] = sin.(ω .* t)  # Sine component
        A[:, 5] = cos.(ω .* t)  # Cosine component
    elseif nstates != 2 && nstates != 3 && nstates != 5
        error("nstates must be 2, 3, or 5, got $nstates")
    end
    
    return A
end

"""
    validate_filter(kf::KalmanFilter)

Validate KalmanFilter configuration and return any warnings.

# Arguments
- `kf::KalmanFilter`: Filter to validate

# Returns
- `Vector{String}`: List of validation warnings (empty if all OK)
"""
function validate_filter(kf::KalmanFilter{T}) where T
    warnings = String[]
    
    # Check state dimensions
    if !(kf.nstates in [2, 3, 5])
        push!(warnings, "nstates must be 2, 3, or 5, got $(kf.nstates)")
    end
    
    # Check initial state dimension
    if length(kf.x0) != kf.nstates
        push!(warnings, "x0 length ($(length(kf.x0))) must equal nstates ($(kf.nstates))")
    end
    
    # Check P0 dimensions
    if isa(kf.P0, Matrix)
        if size(kf.P0) != (kf.nstates, kf.nstates)
            push!(warnings, "P0 matrix size $(size(kf.P0)) must be ($(kf.nstates), $(kf.nstates))")
        end
    elseif kf.P0 <= 0
        push!(warnings, "Scalar P0 must be positive, got $(kf.P0)")
    end
    
    # Check noise parameters
    if kf.q_wpm < 0
        push!(warnings, "q_wpm must be non-negative, got $(kf.q_wpm)")
    end
    
    if kf.q_wfm < 0  
        push!(warnings, "q_wfm must be non-negative, got $(kf.q_wfm)")
    end
    
    if kf.q_rwfm < 0
        push!(warnings, "q_rwfm must be non-negative, got $(kf.q_rwfm)")
    end
    
    if kf.q_irwfm < 0
        push!(warnings, "q_irwfm must be non-negative, got $(kf.q_irwfm)")
    end
    
    if kf.q_diurnal < 0
        push!(warnings, "q_diurnal must be non-negative, got $(kf.q_diurnal)")
    end
    
    # Check diurnal consistency
    if kf.q_diurnal > 0 && kf.nstates != 5
        push!(warnings, "q_diurnal > 0 requires nstates = 5, got nstates = $(kf.nstates)")
    end
    
    # Check tau
    if kf.tau <= 0
        push!(warnings, "tau must be positive, got $(kf.tau)")
    end
    
    return warnings
end

"""
    default_filter(::Type{T}=Float64; nstates::Int=3, tau::Real=1.0) where T

Create a KalmanFilter with reasonable default parameters for typical applications.

# Arguments
- `T`: Number type (default: Float64)
- `nstates::Int`: Number of states (default: 3)
- `tau::Real`: Sampling interval in seconds (default: 1.0)

# Returns  
- `KalmanFilter{T}`: Filter with default parameters

# Example
```julia
kf = default_filter(Float64, nstates=3, tau=1.0)  # Standard 3-state filter
```
"""
function default_filter(::Type{T}=Float64; nstates::Int=3, tau::Real=1.0) where T
    
    return KalmanFilter{T}(
        # Default PID gains (conservative)
        T(0.1),      # g_p
        T(0.01),     # g_i
        T(0.05),     # g_d
        
        # Default noise parameters (typical for oscillators)
        T(100.0),    # q_wpm - 100 ns² measurement noise
        T(0.01),     # q_wfm - White frequency noise
        T(1e-6),     # q_rwfm - Random walk frequency noise
        T(0.0),      # q_irwfm - No integrated RWFM by default
        T(0.0),      # q_diurnal - No diurnal by default
        
        # Model configuration
        nstates,     # nstates
        T(tau),      # tau
        
        # Initial conditions (will be updated by initialize!)
        zeros(T, nstates),  # x0
        T(1e6)              # P0 - Large initial uncertainty
    )
end