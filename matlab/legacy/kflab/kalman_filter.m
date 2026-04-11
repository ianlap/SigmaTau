function [phase_est, freq_est, drift_est, residuals, innovations, steers, ...
          rtP00, rtP11, rtP22, rtP01, rtP02, rtP12, sumsteers, sumsumsteers] = ...
          kalman_filter(rawphase, q_wfm, q_rwfm, R, g_p, g_i, g_d, ...
                       nparams, tau, start_cov, init_state, q_irwfm, q_diurnal, period)
% KALMAN_FILTER - Kalman filter for oscillator phase/frequency tracking
%
% Implements a Kalman filter with optional PID steering control for tracking
% oscillator phase error, frequency, and drift. Supports 2, 3, or 5 state models
% with various noise processes including diurnal variations.
%
% Syntax:
%   [phase_est, freq_est, drift_est, ...] = kalman_filter(rawphase, q_wfm, q_rwfm, R, ...)
%
% Inputs:
%   rawphase   - Vector of raw phase error measurements [ns or cycles]
%   q_wfm      - White frequency modulation variance
%   q_rwfm     - Random walk frequency modulation variance  
%   R          - Measurement noise variance
%   g_p        - Proportional gain for PID control
%   g_i        - Integral gain for PID control
%   g_d        - Derivative gain for PID control
%   nparams    - Number of states: 2 (phase/freq), 3 (+drift), or 5 (+diurnal)
%   tau        - Sampling interval [s]
%   start_cov  - Initial covariance (scalar or matrix)
%   init_state - Initial state vector (optional, default: auto-initialize)
%   q_irwfm    - Integrated RW frequency modulation variance (optional, default: 0)
%   q_diurnal  - Diurnal variation variance (optional, default: 0, requires nparams=5)
%   period     - Period for diurnal terms [s] (optional, default: 86400)
%
% Outputs:
%   phase_est    - Estimated phase error [ns or cycles]
%   freq_est     - Estimated frequency error [ns/s or cycles/s]
%   drift_est    - Estimated frequency drift [ns/s² or cycles/s²]
%   residuals    - Measurement residuals after update
%   innovations  - Kalman filter innovations (prediction errors)
%   steers       - PID steering corrections applied
%   rtP00-rtP12  - Square root of covariance matrix elements
%   sumsteers    - Cumulative frequency steering
%   sumsumsteers - Cumulative phase steering
%
% Example:
%   % Basic 3-state filter with PID control
%   % phase_error_data is the measured phase deviation from reference
%   [phase_est, freq_est, drift_est] = kalman_filter(phase_error_data, ...
%       0.1, 1e-6, 100, 0.1, 0.01, 0.05, 3, 1.0, 1e6);
%
% See also: KF_PREDICT, OPTIMIZE_KF
%% Handle optional parameters
if nargin < 12 || isempty(q_irwfm), q_irwfm = 0; end
if nargin < 13 || isempty(q_diurnal), q_diurnal = 0; end
if nargin < 14 || isempty(period), period = 86400; end  % Daily period

%% Validate inputs
if q_diurnal > 0 && nparams ~= 5
    error('KF:InvalidParams', 'Diurnal terms (q_diurnal > 0) require nparams = 5');
end

if ~ismember(nparams, [2, 3, 5])
    error('KF:InvalidParams', 'nparams must be 2, 3, or 5');
end

%% Initialize Kalman filter
[N, x, P, Phi, Q, H, outputs, phase, pid] = ...
    initialize_kf(rawphase, nparams, tau, start_cov, init_state);

% Unpack output structures for convenience
phase_est = outputs.phase_est;
freq_est = outputs.freq_est;
drift_est = outputs.drift_est;
residuals = outputs.residuals;
innovations = outputs.innovations;
steers = outputs.steers;
sumsteers = outputs.sumsteers;
sumsumsteers = outputs.sumsumsteers;
rtP00 = outputs.rtP00;
rtP11 = outputs.rtP11;
rtP22 = outputs.rtP22;
rtP01 = outputs.rtP01;
rtP02 = outputs.rtP02;
rtP12 = outputs.rtP12;

% For diurnal terms
twopi = 2 * pi;

%% Main Kalman filter loop
for k = 1:N
    % --- Update time-varying parameters ---
    tau_k = tau;  % Could be time-varying if needed
    abstau = abs(tau_k);
    
    % --- Update state transition matrix ---
    if nparams >= 2
        Phi(1,2) = tau_k;  % Phase from frequency
    end
    if nparams >= 3
        Phi(1,3) = 0.5 * tau_k^2;  % Phase from drift
        Phi(2,3) = tau_k;          % Frequency from drift
    end
    % Diurnal states (4,5) remain as identity
    
    % --- Update process noise covariance Q ---
    % Phase variance
    Q(1,1) = q_wfm*abstau + q_rwfm*abstau^3/3 + q_irwfm*abstau^5/20;
    
    if nparams >= 2
        % Phase-frequency covariance
        Q(1,2) = q_rwfm*abstau^2/2 + q_irwfm*abstau^4/8;
        Q(2,1) = Q(1,2);
        % Frequency variance
        Q(2,2) = q_rwfm*abstau + q_irwfm*abstau^3/3;
    end
    
    if nparams >= 3
        % Phase-drift covariance
        Q(1,3) = q_irwfm*abstau^3/6;
        Q(3,1) = Q(1,3);
        % Frequency-drift covariance
        Q(2,3) = q_irwfm*abstau^2/2;
        Q(3,2) = Q(2,3);
        % Drift variance
        Q(3,3) = q_irwfm*abstau;
    end
    
    if nparams == 5
        % Diurnal noise terms
        Q(4,4) = q_diurnal;
        Q(5,5) = q_diurnal;
        % Update measurement matrix for diurnal terms
        H(1,4) = sin(twopi * k / period);
        H(1,5) = cos(twopi * k / period);
    end
    
    % --- Prediction step ---
    if k > 1
        % Predict state
        x = Phi * x;
        
        % Apply steering correction to state (after Phi, before P update)
        x(1) = x(1) + pid.last_steer * tau_k;  % Phase correction
        if nparams >= 2
            x(2) = x(2) + pid.last_steer;      % Frequency correction
        end
        
        % Predict covariance
        P = Phi * P * Phi' + Q;
    end
    
    % --- Update phase measurement with steering ---
    if k > 1
        % Incorporate cumulative steering effects
        phase(k) = rawphase(k) + sumsumsteers(k-1);
        phase(k) = phase(k-1) + rawphase(k) - rawphase(k-1) + sumsteers(k-1);
    end
    
    % --- Innovation ---
    z = phase(k) - (H * x);
    
    % --- Update step ---
    S = H * P * H' + R;     % Innovation covariance (scalar for single measurement)
    K = (P * H') / S;       % Kalman gain vector
    
    % State update
    x = x + K * z;
    
    % Covariance update (Joseph form for numerical stability)
    I_KH = eye(nparams) - K * H;
    P = I_KH * P;
    
    % Ensure P remains symmetric (numerical stability)
    %P = make_symmetric(P, nparams);
    
    % --- Calculate residual ---
    residual = phase(k) - x(1);
    
    % --- PID steering control ---
    pid.sumx = pid.sumx + x(1);  % Accumulate for integral term
    if nparams >= 2
        steer = -g_p * x(1) - g_i * pid.sumx - g_d * x(2);
    else
        steer = -g_p * x(1) - g_i * pid.sumx;  % No derivative term without frequency
    end
    pid.last_steer = steer;
    
    % Update cumulative steering
    if k == 1
        sumsteers(k) = pid.last_steer;
        sumsumsteers(k) = sumsteers(k);
    else
        sumsteers(k) = sumsteers(k-1) + pid.last_steer;
        sumsumsteers(k) = sumsumsteers(k-1) + sumsteers(k);
    end
    
    % --- Store results ---
    phase_est(k) = x(1);
    if nparams >= 2
        freq_est(k) = x(2);
    else
        freq_est(k) = 0.0;
    end
    if nparams >= 3
        drift_est(k) = x(3);
    else
        drift_est(k) = 0.0;
    end
    
    residuals(k) = residual;
    innovations(k) = z;
    steers(k) = steer;
    
    % Store covariance elements (as square roots)
    rtP00(k) = safe_sqrt(P(1, 1));
    if nparams >= 2
        rtP11(k) = safe_sqrt(P(2, 2));
        rtP01(k) = safe_sqrt(P(1, 2));
    else
        rtP11(k) = 0.0;
        rtP01(k) = 0.0;
    end
    if nparams >= 3
        rtP22(k) = safe_sqrt(P(3, 3));
        rtP02(k) = safe_sqrt(P(1, 3));
        rtP12(k) = safe_sqrt(P(2, 3));
    else
        rtP22(k) = 0.0;
        rtP02(k) = 0.0;
        rtP12(k) = 0.0;
    end
end

end

%% ===================== Helper Functions =====================

function [N, x, P, Phi, Q, H, outputs, phase, pid] = initialize_kf(rawphase, nparams, tau, start_cov, init_state)
% Initialize all Kalman filter variables and pre-allocate arrays

N = length(rawphase);
phase = rawphase;  % Working copy - IMPORTANT: modified in-place for steering

% --- Pre-allocate output arrays ---
outputs = struct();
outputs.phase_est = zeros(N, 1);
outputs.freq_est = zeros(N, 1);
outputs.drift_est = zeros(N, 1);
outputs.residuals = zeros(N, 1);
outputs.innovations = zeros(N, 1);
outputs.steers = zeros(N, 1);
outputs.sumsteers = zeros(N, 1);
outputs.sumsumsteers = zeros(N, 1);

% Covariance elements (stored as square roots for numerical stability)
outputs.rtP00 = zeros(N, 1);
outputs.rtP11 = zeros(N, 1);
outputs.rtP22 = zeros(N, 1);
outputs.rtP01 = zeros(N, 1);
outputs.rtP02 = zeros(N, 1);
outputs.rtP12 = zeros(N, 1);

% --- Initialize state vector ---
if isempty(init_state) || any(isnan(init_state))
    x = zeros(nparams, 1);
    % Auto-initialize from first few samples
    if N >= 3
        if nparams >= 2
            x(1) = rawphase(3);                           % Phase estimate
            x(2) = (rawphase(3) - rawphase(1)) / (2*tau); % Frequency estimate
        end
        % Higher states start at zero
    else
        x(1) = rawphase(1);
    end
else
    x = init_state(:);  % Ensure column vector
end

% --- Initialize covariance matrix ---
if isscalar(start_cov)
    P = eye(nparams) * start_cov;
else
    P = start_cov;
end

% --- Setup system matrices ---
% State transition matrix Phi - base structure
Phi = eye(nparams);

% Process noise covariance matrix Q
Q = zeros(nparams);

% Measurement matrix H
H = zeros(1, nparams);
H(1) = 1.0;  % Measure phase only

% --- Initialize PID controller state ---
pid = struct();
pid.sumx = 0.0;         % Integral term accumulator
pid.last_steer = 0.0;   % Previous steering value

% Initialize first steering values
outputs.sumsteers(1) = 0;
outputs.sumsumsteers(1) = 0;

end

function y = safe_sqrt(x)
% Safe square root that handles negative values
% Returns signed square root: sqrt(|x|) * sign(x)

if abs(x) < 1e-10
    y = 0.0;
elseif x >= 0
    y = sqrt(x);
else
    y = -sqrt(-x);  % Preserve sign information
end
end

function X = make_symmetric(X, nstates)
% Force matrix to be symmetric by averaging off-diagonal elements
% Helps maintain numerical stability in covariance matrices

if nstates > 1
    xave = (X(1,2) + X(2,1)) / 2;
    X(1,2) = xave;
    X(2,1) = xave;
end
if nstates >= 3
    xave = (X(1,3) + X(3,1)) / 2;
    X(1,3) = xave;
    X(3,1) = xave;
    xave = (X(2,3) + X(3,2)) / 2;
    X(3,2) = xave;
    X(2,3) = xave;
end
if nstates == 5
    % Make all off-diagonal elements symmetric
    for i = 1:5
        for j = i+1:5
            xave = (X(i,j) + X(j,i)) / 2;
            X(i,j) = xave;
            X(j,i) = xave;
        end
    end
end
end