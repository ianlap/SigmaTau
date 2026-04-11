module KalmanLab

using LinearAlgebra
using Statistics

# Export main types
export KalmanFilter, KalmanResults, PredictionResults

# Export main functions
export run, holdoverpredict, optimize!, initialize!, reset!

# Export utility functions
export default_filter, validate_filter

# Core data structures
"""
    KalmanFilter{T<:Real}

Kalman filter configuration for oscillator phase/frequency tracking with PID steering.

# Fields
- `g_p::T`: Proportional gain for PID control
- `g_i::T`: Integral gain for PID control  
- `g_d::T`: Derivative gain for PID control
- `q_wpm::T`: White phase modulation (measurement noise R)
- `q_wfm::T`: White frequency modulation
- `q_rwfm::T`: Random walk frequency modulation
- `q_irwfm::T`: Integrated random walk frequency modulation (optional)
- `q_diurnal::T`: Diurnal variation (optional, requires nstates=5)
- `nstates::Int`: Number of states (2, 3, or 5)
- `tau::T`: Sampling interval (seconds)
- `x0::Vector{T}`: Initial state estimate [phase, freq, drift, ...]
- `P0::Union{T, Matrix{T}}`: Initial covariance (scalar or matrix from initialize!)

# Usage
```julia
# Create 3-state filter with PID control
kf = KalmanFilter(
    g_p=0.1, g_i=0.01, g_d=0.05,           # PID gains
    q_wpm=100.0, q_wfm=0.01, q_rwfm=1e-6,  # Noise parameters
    q_irwfm=0.0, q_diurnal=0.0,            # Optional terms
    nstates=3, tau=1.0,                     # 3 states, 1 sec sampling
    x0=zeros(3), P0=1e6                     # Initial conditions
)

# Run filter
results = run!(kf, phase_data)
```
"""
mutable struct KalmanFilter{T<:Real}
    # PID controller gains (mutable for holdover scenarios)
    g_p::T
    g_i::T
    g_d::T
    
    # Noise parameters (Q matrix components)
    q_wpm::T     # White phase modulation (measurement noise R)
    q_wfm::T     # White frequency modulation
    q_rwfm::T    # Random walk frequency modulation
    q_irwfm::T   # Integrated RWFM (optional)
    q_diurnal::T # Diurnal variation (optional)
    
    # Model configuration
    nstates::Int        # 2, 3, or 5 states
    tau::T             # Sampling interval
    
    # Initial conditions
    x0::Vector{T}      # Initial state estimate
    P0::Union{T, Matrix{T}}  # Initial covariance (scalar or matrix)
end

"""
    KalmanResults{T<:Real}

Results returned by running a Kalman filter on data.

# Fields
- `phase_est::Vector{T}`: Estimated phase error
- `freq_est::Vector{T}`: Estimated frequency error
- `drift_est::Vector{T}`: Estimated frequency drift (zeros if nstates < 3)
- `diurnal_cos::Vector{T}`: Cosine diurnal component (zeros if nstates < 5)
- `diurnal_sin::Vector{T}`: Sine diurnal component (zeros if nstates < 5)
- `residuals::Vector{T}`: Measurement residuals (measurement - estimate)
- `innovations::Vector{T}`: Kalman filter innovations (prediction errors)
- `steers::Vector{T}`: PID steering corrections applied
- `sumsteers::Vector{T}`: Cumulative frequency steering
- `sum2steers::Vector{T}`: Cumulative phase steering
- `covariances::Vector{Matrix{T}}`: Covariance matrices over time
- `final_state::Vector{T}`: Final state estimate
- `final_P::Matrix{T}`: Final covariance matrix
- `config::KalmanFilter{T}`: Filter configuration used
- `converged::Bool`: Filter convergence status
- `n_samples::Int`: Number of samples processed

# Usage
```julia
# Access results
plot(results.phase_est)      # Plot phase estimates
plot(results.residuals)      # Plot residuals
rms_error = sqrt(mean(results.residuals.^2))  # Compute RMS error
```
"""
struct KalmanResults{T<:Real}
    # State estimates
    phase_est::Vector{T}
    freq_est::Vector{T}
    drift_est::Vector{T}
    diurnal_cos::Vector{T}
    diurnal_sin::Vector{T}
    
    # Filter diagnostics
    residuals::Vector{T}
    innovations::Vector{T}
    
    # Steering outputs
    steers::Vector{T}
    sumsteers::Vector{T}
    sum2steers::Vector{T}
    
    # Covariance and final state
    covariances::Vector{Matrix{T}}
    final_state::Vector{T}
    final_P::Matrix{T}
    
    # Metadata
    config::KalmanFilter{T}
    converged::Bool
    n_samples::Int
end

"""
    PredictionResults{T<:Real}

Results from holdover prediction analysis.

# Fields
- `horizons::Vector{Int}`: Prediction horizons (samples)
- `predictions::Vector{T}`: Predicted phase values
- `rms_errors::Vector{T}`: RMS prediction errors at each horizon
- `n_predictions::Vector{Int}`: Number of predictions at each horizon
- `initial_state::Vector{T}`: State used for predictions
- `config::KalmanFilter{T}`: Filter configuration

# Usage
```julia
predictions = holdoverpredict(kf, [10, 100, 1000])
plot(predictions.horizons, predictions.rms_errors)  # Plot RMS vs horizon
```
"""
struct PredictionResults{T<:Real}
    horizons::Vector{Int}
    predictions::Vector{T}
    rms_errors::Vector{T}
    n_predictions::Vector{Int}
    initial_state::Vector{T}
    config::KalmanFilter{T}
end

# Include source files
include("types.jl")
include("filter.jl")
include("prediction.jl")
include("optimization.jl")
include("utils.jl")

end