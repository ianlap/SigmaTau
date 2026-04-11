# filter.jl - Core Kalman filter implementation

"""
    run(kf::KalmanFilter, data::Vector{T}) -> KalmanResults{T} where T

Execute Kalman filter on phase error data using the configured parameters.

This function implements the complete Kalman filter algorithm with PID steering control,
translating the MATLAB implementation with careful attention to linear algebra operations.

# Arguments
- `kf::KalmanFilter{T}`: Configured filter (must have been initialized with initialize!)
- `data::Vector{T}`: Phase error data (trimmed by initialize!)

# Returns
- `KalmanResults{T}`: Complete filter results including state estimates and diagnostics

# Algorithm
Implements the standard Kalman filter predict/update cycle:
1. Predict: x⁻ = Φx⁺, P⁻ = ΦP⁺Φᵀ + Q
2. Update: K = P⁻Hᵀ(HP⁻Hᵀ + R)⁻¹, x⁺ = x⁻ + Kz, P⁺ = (I-KH)P⁻
3. PID Steering: Apply proportional-integral-derivative control
4. Store results and covariance information
"""
function run(kf::KalmanFilter{T}, data::Vector{T}) where T
    
    # Validate inputs
    warnings = validate_filter(kf)
    if !isempty(warnings)
        for w in warnings
            @warn w
        end
        error("Filter validation failed - see warnings above")
    end
    
    N = length(data)
    if N == 0
        error("Data vector is empty")
    end
    
    # Initialize filter state and matrices
    x, P, Φ, Q, H, results, pid_state = initialize_filter_state(kf, N)
    
    # Extract configuration for convenience
    nstates = kf.nstates
    τ = kf.tau
    g_p, g_i, g_d = kf.g_p, kf.g_i, kf.g_d
    
    # Process noise parameters
    q_wpm, q_wfm, q_rwfm = kf.q_wpm, kf.q_wfm, kf.q_rwfm
    q_irwfm, q_diurnal = kf.q_irwfm, kf.q_diurnal
    R = q_wpm  # Measurement noise
    
    # Diurnal parameters (if applicable)
    period = T(86400.0)  # 24 hour period in seconds
    twopi = T(2π)
    
    # Working copy of phase data (modified for steering)
    phase = copy(data)
    
    # Main Kalman filter loop
    for k in 1:N
        
        # --- Update time-varying parameters ---
        τₖ = τ  # Could be time-varying if needed
        absτ = abs(τₖ)
        
        # --- Update state transition matrix Φ ---
        update_state_transition!(Φ, nstates, τₖ)
        
        # --- Update process noise covariance Q ---
        update_process_noise!(Q, nstates, q_wfm, q_rwfm, q_irwfm, q_diurnal, absτ)
        
        # --- Update measurement matrix H (for diurnal terms) ---
        if nstates == 5
            # Diurnal measurement terms: H[4] = sin(2πk/period), H[5] = cos(2πk/period)
            H[4] = sin(twopi * k / period)
            H[5] = cos(twopi * k / period)
        end
        
        # --- Prediction step ---
        if k > 1
            # Predict state: x⁻ = Φx⁺
            x = Φ * x
            
            # Apply steering correction to predicted state
            x[1] += pid_state[2] * τₖ  # Phase correction (last_steer)
            if nstates >= 2
                x[2] += pid_state[2]    # Frequency correction (last_steer)
            end
            
            # Predict covariance: P⁻ = ΦP⁺Φᵀ + Q
            P = Φ * P * Φ' + Q
            
            # Ensure symmetry after prediction step
            P = Symmetric(P)
        end
        
        # --- Update phase measurement with steering effects ---
        if k > 1
            # MATLAB does this (line 152 overwrites line 151):
            # phase(k) = phase(k-1) + rawphase(k) - rawphase(k-1) + sumsteers(k-1);
            # In MATLAB: phase is working copy, rawphase is original data
            phase[k] = phase[k-1] + data[k] - data[k-1] + results.sumsteers[k-1]
        end
        
        # --- Innovation (prediction error) ---
        innovation = phase[k] - dot(H, x)  # z = y - Hx⁻
        
        # --- Update step ---
        # Innovation covariance: S = HP⁻Hᵀ + R (scalar for single measurement)
        S = (H * P * H')[1,1] + R  # Explicit matrix computation, extract scalar
        
        # Kalman gain: K = P⁻Hᵀ S⁻¹ (need column vector)
        K = (P * H') / S  # H' converts row vector to column vector
        
        # State update: x⁺ = x⁻ + Kz
        x = x + K * innovation
        
        # Covariance update - MATLAB exact match
        # Use simple form exactly as MATLAB (line 167: P = I_KH * P)
        I_KH = I - K * H  # K is column vector, H is row vector
        P = I_KH * P      # Simple form - matches MATLAB exactly
        
        # Force symmetry to ensure numerical stability (like MATLAB line 170)
        P = Symmetric(P)
        
        # Stabilized form (more stable but doesn't match MATLAB):
        # P = I_KH * P * I_KH'
        
        # Full Joseph form (most stable but doesn't match MATLAB):
        # P = I_KH * P * I_KH' + K * K' * R
        
        # --- Calculate residual ---
        residual = phase[k] - x[1]
        
        # --- PID steering control ---
        pid_state[1] += x[1]  # Accumulate for integral term (sumx)
        
        if nstates >= 2
            steer = -g_p * x[1] - g_i * pid_state[1] - g_d * x[2]
        else
            steer = -g_p * x[1] - g_i * pid_state[1]  # No derivative term
        end
        
        pid_state[2] = steer  # Update last_steer
        
        # Update cumulative steering
        if k == 1
            results.sumsteers[k] = pid_state[2]  # last_steer
            results.sum2steers[k] = results.sumsteers[k]
        else
            results.sumsteers[k] = results.sumsteers[k-1] + pid_state[2]  # last_steer
            results.sum2steers[k] = results.sum2steers[k-1] + results.sumsteers[k]
        end
        
        # --- Store results ---
        results.phase_est[k] = x[1]
        results.freq_est[k] = nstates >= 2 ? x[2] : zero(T)
        results.drift_est[k] = nstates >= 3 ? x[3] : zero(T)
        results.diurnal_cos[k] = nstates == 5 ? x[4] : zero(T)
        results.diurnal_sin[k] = nstates == 5 ? x[5] : zero(T)
        
        results.residuals[k] = residual
        results.innovations[k] = innovation
        results.steers[k] = steer
        
        # Store covariance matrix (full matrix for analysis)
        results.covariances[k] = copy(P)
    end
    
    # Package final results
    return KalmanResults{T}(
        results.phase_est,
        results.freq_est, 
        results.drift_est,
        results.diurnal_cos,
        results.diurnal_sin,
        results.residuals,
        results.innovations,
        results.steers,
        results.sumsteers,
        results.sum2steers,
        results.covariances,
        copy(x),      # final_state
        copy(P),      # final_P
        kf,           # config
        true,         # converged (could add convergence check)
        N             # n_samples
    )
end

"""
    initialize_filter_state(kf::KalmanFilter{T}, N::Int) where T

Initialize all filter variables and pre-allocate result arrays.

Returns tuple of (x, P, Φ, Q, H, results, pid_state) where:
- x: Initial state vector
- P: Initial covariance matrix  
- Φ: State transition matrix (updated each step)
- Q: Process noise covariance (updated each step)
- H: Measurement matrix
- results: Pre-allocated result arrays
- pid_state: PID controller state
"""
function initialize_filter_state(kf::KalmanFilter{T}, N::Int) where T
    
    nstates = kf.nstates
    
    # Initial state and covariance
    x = copy(kf.x0)
    P_init = isa(kf.P0, Matrix) ? copy(kf.P0) : Matrix{T}(kf.P0 * I, nstates, nstates)
    P = Symmetric(P_init)  # Ensure initial covariance is symmetric
    
    # System matrices
    Φ = Matrix{T}(I, nstates, nstates)  # State transition (updated each step)
    Q = zeros(T, nstates, nstates)      # Process noise (updated each step)
    H = zeros(T, nstates)'              # Measurement matrix (row vector)
    H[1] = one(T)                       # Measure phase only
    
    # Pre-allocate result arrays
    results = (
        phase_est = zeros(T, N),
        freq_est = zeros(T, N),
        drift_est = zeros(T, N),
        diurnal_cos = zeros(T, N),
        diurnal_sin = zeros(T, N),
        residuals = zeros(T, N),
        innovations = zeros(T, N),
        steers = zeros(T, N),
        sumsteers = zeros(T, N),
        sum2steers = zeros(T, N),
        covariances = Vector{Matrix{T}}(undef, N)
    )
    
    # Initialize covariance storage
    for i in 1:N
        results.covariances[i] = zeros(T, nstates, nstates)
    end
    
    # PID controller state (mutable)
    pid_state = [zero(T), zero(T)]  # [sumx, last_steer] as mutable array
    
    return x, P, Φ, Q, H, results, pid_state
end

"""
    update_state_transition!(Φ::Matrix{T}, nstates::Int, τ::T) where T

Update state transition matrix Φ for current time step.

State models:
- 2-state: [1 τ; 0 1] (phase, frequency)
- 3-state: [1 τ τ²/2; 0 1 τ; 0 0 1] (phase, frequency, drift)
- 5-state: 3-state + identity for diurnal terms
"""
function update_state_transition!(Φ::Matrix{T}, nstates::Int, τ::T) where T
    
    # Reset to identity
    Φ .= zero(T)
    for i in 1:nstates
        Φ[i,i] = one(T)
    end
    
    # Phase from frequency
    if nstates >= 2
        Φ[1,2] = τ
    end
    
    # Phase and frequency from drift
    if nstates >= 3
        Φ[1,3] = τ^2 / 2   # Phase from drift
        Φ[2,3] = τ         # Frequency from drift  
    end
    
    # Diurnal states (4,5) remain as identity (nstates == 5)
end

"""
    update_process_noise!(Q::Matrix{T}, nstates::Int, q_wfm::T, q_rwfm::T, 
                         q_irwfm::T, q_diurnal::T, τ::T) where T

Update process noise covariance matrix Q for current time step.

Implements the exact noise model from the MATLAB version with proper
integration of white FM, random walk FM, and integrated RWFM processes.
"""
function update_process_noise!(Q::Matrix{T}, nstates::Int, q_wfm::T, q_rwfm::T, 
                              q_irwfm::T, q_diurnal::T, τ::T) where T
    
    # Clear matrix
    Q .= zero(T)
    
    τ² = τ^2
    τ³ = τ^3
    τ⁴ = τ^4
    τ⁵ = τ^5
    
    # Phase variance (1,1 element)
    Q[1,1] = q_wfm*τ + q_rwfm*τ³/3 + q_irwfm*τ⁵/20
    
    if nstates >= 2
        # Phase-frequency covariance (1,2 and 2,1 elements)
        Q[1,2] = q_rwfm*τ²/2 + q_irwfm*τ⁴/8
        Q[2,1] = Q[1,2]  # Symmetric
        
        # Frequency variance (2,2 element)
        Q[2,2] = q_rwfm*τ + q_irwfm*τ³/3
    end
    
    if nstates >= 3
        # Phase-drift covariances (1,3 and 3,1 elements) 
        Q[1,3] = q_irwfm*τ³/6
        Q[3,1] = Q[1,3]  # Symmetric
        
        # Frequency-drift covariances (2,3 and 3,2 elements)
        Q[2,3] = q_irwfm*τ²/2  
        Q[3,2] = Q[2,3]  # Symmetric
        
        # Drift variance (3,3 element)
        Q[3,3] = q_irwfm*τ
    end
    
    if nstates == 5
        # Diurnal noise terms (independent)
        Q[4,4] = q_diurnal
        Q[5,5] = q_diurnal
    end
end