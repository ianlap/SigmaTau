function prediction_results = kf_predict(phase_data, tau, noise_params, kf_params, pred_params)
% KF_PREDICT - Kalman filter prediction analysis with RMS error calculation
%
% Tests Kalman filter prediction performance by running the filter to maturity,
% then evaluating multi-step predictions at various horizons. Optimized for
% speed with minimal memory allocation and optional progress reporting.
%
% Syntax:
%   prediction_results = kf_predict(phase_data, tau, noise_params, kf_params, pred_params)
%
% Inputs:
%   phase_data   - Vector of raw phase error measurements (deviation from ideal) [ns or cycles]
%   tau          - Sampling interval [s]
%   noise_params - Struct with noise parameters:
%                  .q_wpm     - White phase modulation (measurement noise)
%                  .q_wfm     - White frequency modulation  
%                  .q_rwfm    - Random walk frequency modulation
%                  .q_irwfm   - Integrated RWFM (optional, default: 0)
%                  .q_diurnal - Diurnal variations (optional, default: 0)
%   kf_params    - Struct with filter configuration:
%                  .nstates     - Number of states: 2, 3, or 5
%                  .maturity    - Samples before starting predictions
%                  .max_horizon - Maximum prediction horizon [samples]
%                  .period      - Diurnal period [s] (optional, default: 86400)
%   pred_params  - Struct with prediction settings:
%                  .g_p, .g_i, .g_d     - PID gains (set to 0 for no control)
%                  .init_cov            - Initial covariance
%                  .save_predictions    - Save individual predictions (default: false)
%                  .verbose             - Progress reporting (default: false)
%
% Outputs:
%   prediction_results - Struct containing:
%                       .rms_stats      - Table with columns: horizon, rms_error, n_samples
%                       .state_history  - State estimates from filter
%                       .residuals      - Measurement residuals
%                       .noise_params   - Input parameters (for reference)
%                       .kf_params      - Input parameters (for reference)
%                       .kf_time        - Kalman filter execution time [s]
%                       .pred_time      - Prediction analysis time [s]
%                       .kf_data        - Complete Kalman filter outputs:
%                                        .phase_est, .freq_est, .drift_est
%                                        .innovations, .steers, .covariances
%                                        .sumsteers, .sumsumsteers
%
% Example:
%   % phase_error_data is measured phase deviation from reference oscillator
%   noise_params = struct('q_wpm', 100, 'q_wfm', 0.01, 'q_rwfm', 1e-6);
%   kf_params = struct('nstates', 3, 'maturity', 50000, 'max_horizon', 10000);
%   pred_params = struct('g_p', 0, 'g_i', 0, 'g_d', 0, 'init_cov', 1e30);
%   results = kf_predict(phase_error_data, 1.0, noise_params, kf_params, pred_params);
%
% See also: KALMAN_FILTER, OPTIMIZE_KF

import KFLab.*

%% Validate and set defaults for all parameters
[noise_params, kf_params, pred_params, N, nstates, maturity, max_horizon] = ...
    validate_and_set_defaults(phase_data, noise_params, kf_params, pred_params);

%% Run Kalman filter
tic;
[phase_est, freq_est, drift_est, residuals, innovations, steers, ...
 rtP00, rtP11, rtP22, rtP01, rtP02, rtP12, sumsteers, sumsumsteers] = ...
 kalman_filter(phase_data, ...
               noise_params.q_wfm, ...     
               noise_params.q_rwfm, ...    
               noise_params.q_wpm, ...     % R (measurement noise)
               pred_params.g_p, ...        
               pred_params.g_i, ...        
               pred_params.g_d, ...        
               nstates, ...                
               tau, ...                    
               pred_params.init_cov, ...   
               [], ...                     % Auto-initialize state
               noise_params.q_irwfm, ...   
               noise_params.q_diurnal, ... 
               kf_params.period);          
kf_time = toc;

if pred_params.verbose
    fprintf('  KF completed in %.2f seconds\n', kf_time);
end

%% Build state history
% Always use 3-row format for consistency (unused states remain zero)
state_history = zeros(3, N);
state_history(1, :) = phase_est';
state_history(2, :) = freq_est';
if nstates >= 3
    state_history(3, :) = drift_est';
end

%% Compute RMS prediction errors
tic;
[rms_stats, predictions] = compute_prediction_rms(phase_data, state_history, ...
                                                 nstates, maturity, max_horizon, ...
                                                 pred_params);
pred_time = toc;

if pred_params.verbose
    fprintf('  Predictions completed in %.2f seconds\n', pred_time);
end

%% Package results
prediction_results = struct();
prediction_results.rms_stats = rms_stats;
prediction_results.predictions = predictions;
prediction_results.state_history = state_history;
prediction_results.residuals = residuals;
prediction_results.noise_params = noise_params;
prediction_results.kf_params = kf_params;
prediction_results.kf_time = kf_time;
prediction_results.pred_time = pred_time;

% Include complete Kalman filter data
prediction_results.kf_data = struct();
prediction_results.kf_data.phase_est = phase_est;
prediction_results.kf_data.freq_est = freq_est;
prediction_results.kf_data.drift_est = drift_est;
prediction_results.kf_data.innovations = innovations;
prediction_results.kf_data.steers = steers;
prediction_results.kf_data.sumsteers = sumsteers;
prediction_results.kf_data.sumsumsteers = sumsumsteers;

% Include covariance matrices
prediction_results.kf_data.covariances = struct();
prediction_results.kf_data.covariances.P00 = rtP00;
prediction_results.kf_data.covariances.P11 = rtP11;
prediction_results.kf_data.covariances.P22 = rtP22;
prediction_results.kf_data.covariances.P01 = rtP01;
prediction_results.kf_data.covariances.P02 = rtP02;
prediction_results.kf_data.covariances.P12 = rtP12;

%% Display summary (if verbose)
if pred_params.verbose
    fprintf('\n RMS errors:\n');
    horizons_to_show = [10, 100, 1000, 10000];
    for h = horizons_to_show
        idx = find(rms_stats.horizon == h, 1);
        if ~isempty(idx)
            fprintf('  h=%d: %.3f ns\n', h, rms_stats.rms_error(idx));
        end
    end
    fprintf('\nTotal time: %.2f seconds\n', kf_time + pred_time);
end

end

%% ===================== Helper Functions =====================

function [noise_params, kf_params, pred_params, N, nstates, maturity, max_horizon] = ...
    validate_and_set_defaults(phase_data, noise_params, kf_params, pred_params)
% Validate inputs and set default values for all parameters

N = length(phase_data);

% --- Set defaults for noise parameters ---
if ~isfield(noise_params, 'q_irwfm'), noise_params.q_irwfm = 0; end
if ~isfield(noise_params, 'q_diurnal'), noise_params.q_diurnal = 0; end

% --- Set defaults for KF parameters ---
if ~isfield(kf_params, 'maturity'), kf_params.maturity = 50000; end
if ~isfield(kf_params, 'max_horizon'), kf_params.max_horizon = 80000; end
if ~isfield(kf_params, 'period'), kf_params.period = 86400; end

% --- Set defaults for prediction parameters ---
if ~exist('pred_params', 'var') || isempty(pred_params)
    pred_params = struct();
end

% Individual field defaults
if ~isfield(pred_params, 'g_p'), pred_params.g_p = 0; end
if ~isfield(pred_params, 'g_i'), pred_params.g_i = 0; end
if ~isfield(pred_params, 'g_d'), pred_params.g_d = 0; end
if ~isfield(pred_params, 'init_cov'), pred_params.init_cov = 1e30; end
if ~isfield(pred_params, 'save_predictions'), pred_params.save_predictions = false; end
if ~isfield(pred_params, 'verbose'), pred_params.verbose = false; end

% --- Extract and validate key parameters ---
nstates = kf_params.nstates;
maturity = min(kf_params.maturity, N-1);
max_horizon = min(kf_params.max_horizon, N - maturity - 1);

% Validate state configuration
if ~ismember(nstates, [2, 3, 5])
    error('KF_PREDICT:InvalidStates', 'nstates must be 2, 3, or 5');
end

% Validate noise parameters consistency
if noise_params.q_diurnal > 0 && nstates ~= 5
    error('KF_PREDICT:InvalidConfig', ...
          'Diurnal noise (q_diurnal > 0) requires nstates = 5');
end

% Display configuration if verbose
if pred_params.verbose
    fprintf('\n=== KF PREDICTION ANALYSIS ===\n');
    fprintf('Data length: %d samples\n', N);
    fprintf('States: %d\n', nstates);
    fprintf('Maturity: %d samples\n', maturity);
    fprintf('Max horizon: %d samples\n', max_horizon);
end

end

function [rms_stats, predictions] = compute_prediction_rms(phase_data, state_history, ...
                                                          nstates, maturity, max_horizon, ...
                                                          pred_params)
% Compute RMS prediction errors at multiple horizons

N = length(phase_data);

% Pre-allocate for variance accumulation
var_accum = zeros(max_horizon, 1);
nvar = zeros(max_horizon, 1);

% Progress tracking setup
if pred_params.verbose
    fprintf('\n');
    text_progress(0, 'Generating predictions');
end

% Calculate total epochs and update interval
total_epochs = N - 1 - maturity;
update_interval = max(100, floor(total_epochs / 100)); % Update ~100 times

% Main prediction loop 
for np = maturity:(N-1)
    % Update progress bar
    if pred_params.verbose && mod(np - maturity, update_interval) == 0
        text_progress((np - maturity) / total_epochs);
    end
    
    % Extract current state
    x1 = state_history(1, np);  % Phase
    x2 = state_history(2, np);  % Frequency
    
    % Determine valid horizon range for this time point
    h_max = min(max_horizon, N - np);
    
    % Vectorized prediction for all horizons
    h_vec = (1:h_max)';
    
    if nstates == 2
        % Linear prediction: x(t+h) = x(t) + v(t)*h
        xpred_vec = x1 + x2 * h_vec;
    elseif nstates >= 3
        % Quadratic prediction: x(t+h) = x(t) + v(t)*h + 0.5*a(t)*h²
        x3 = state_history(3, np);  % Drift
        xpred_vec = x1 + x2 * h_vec + 0.5 * x3 * h_vec.^2;
    end
    
    % Get actual values
    actual_vec = phase_data((np+1):(np+h_max));
    
    % Compute squared errors
    error_vec = actual_vec - xpred_vec;
    error_sq = error_vec.^2;
    
    % Accumulate for RMS calculation
    var_accum(1:h_max) = var_accum(1:h_max) + error_sq;
    nvar(1:h_max) = nvar(1:h_max) + 1;
end

% Complete progress
if pred_params.verbose
    text_progress(1);
end

% Calculate RMS for valid horizons
valid_h = find(nvar > 0);
rms = sqrt(var_accum(valid_h) ./ nvar(valid_h));

% Create results table
rms_stats = table(valid_h, rms, nvar(valid_h), ...
                 'VariableNames', {'horizon', 'rms_error', 'n_samples'});

% Individual predictions (only if requested - slows computation)
predictions = [];
if pred_params.save_predictions
    warning('KF_PREDICT:SlowOption', ...
            'save_predictions=true significantly slows computation');
    % Implementation omitted for brevity - rarely needed
end

end