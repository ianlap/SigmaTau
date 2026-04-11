function [q_opt, results] = optimize_kf(phase_data, tau, q_initial, opt_config)
% OPTIMIZE_KF - Optimize Kalman filter Q parameters via grid search or fmincon
%
% Finds optimal process noise parameters (Q matrix) for a Kalman filter by
% minimizing weighted RMS prediction errors at specified horizons. Currently
% implements grid search with fmincon planned for future development.
%
% Syntax:
%   [q_opt, results] = optimize_kf(phase_data, tau, q_initial, opt_config)
%
% Inputs:
%   phase_data - Vector of raw phase error measurements [ns or cycles]
%   tau        - Sampling interval [s]
%   q_initial  - Struct with initial Q parameters:
%                .q_wpm  - White phase modulation (fixed during optimization)
%                .q_wfm  - White frequency modulation (initial guess)
%                .q_rwfm - Random walk frequency modulation (initial guess)
%                .q_irwfm - Integrated RWFM (optional, default: 0)
%   opt_config - Struct with optimization settings (all fields optional):
%                .search_range      - Search ±N decades around initial (default: 2)
%                .n_grid_per_decade - Grid points per decade (default: 5)
%                .nstates           - Number of KF states (default: 2 or 3 based on q_irwfm)
%                .target_horizons   - Prediction horizons to optimize (default: [10 100 1000])
%                .horizon_weights   - Weights for each horizon (default: equal weights)
%                .maturity          - KF maturity before predictions (default: 50000)
%                .method            - 'grid' or 'fmincon' (default: 'grid')
%                .verbose           - Display progress (default: true)
%
% Outputs:
%   q_opt   - Struct with optimal Q parameters
%   results - Struct containing:
%             .rms_opt         - RMS errors at each target horizon
%             .weighted_rms    - Weighted cost function value
%             .search_history  - [params, cost] matrix for all evaluations
%             .method          - Optimization method used
%             .n_evaluations   - Total function evaluations
%             .optimisation_time - Total execution time [s]
%
% Notes:
%   - WPM (q_wpm) is held fixed during optimization as it represents measurement noise
%   - Grid search uses logarithmic spacing for Q parameters
%   - Parallel processing is used for grid search when available
%
% Example:
%   q_initial = struct('q_wpm', 100, 'q_wfm', 0.01, 'q_rwfm', 1e-6);
%   opt_config = struct('search_range', 2, 'target_horizons', [10 100 1000]);
%   [q_opt, results] = optimize_kf(phase_error_data, 1.0, q_initial, opt_config);
%
% See also: KF_PREDICT, KALMAN_FILTER

%% Initialize configuration with defaults
cfg = set_optimization_defaults(opt_config, q_initial);

if cfg.verbose
    fprintf('\n=== KF Q PARAMETER OPTIMIZATION (Fixed WPM) ===\n');
    fprintf('Search ±%d decades ; method = %s\n', cfg.search_range, cfg.method);
    fprintf('WPM fixed at: %.3e\n', q_initial.q_wpm);
end

%% Run optimization
tic;
switch lower(cfg.method)
    case 'grid'
        [q_opt, results] = optimize_grid(phase_data, tau, q_initial, cfg);
    case 'fmincon'
        [q_opt, results] = optimize_fmincon(phase_data, tau, q_initial, cfg);
    otherwise
        error('OPTIMIZE_KF:UnknownMethod', 'Unknown optimization method "%s".', cfg.method);
end
results.optimisation_time = toc;

end

%% ===================== Helper Functions =====================

function cfg = set_optimization_defaults(user_config, q0)
% Set default configuration values and merge with user settings

% Start with default configuration
defaults = struct();
defaults.search_range      = 2;
defaults.n_grid_per_decade = 5;
defaults.nstates           = 2 + (isfield(q0, 'q_irwfm') && q0.q_irwfm > 0);
defaults.target_horizons   = [10 100 1000];
defaults.horizon_weights   = [];  % Will be set to equal weights below
defaults.maturity          = 5e4;
defaults.method            = 'grid';
defaults.verbose           = true;

% Initialize output with defaults
cfg = defaults;

% Override with user-specified values
if nargin >= 1 && ~isempty(user_config)
    fields = fieldnames(user_config);
    for i = 1:length(fields)
        cfg.(fields{i}) = user_config.(fields{i});
    end
end

% Ensure q_irwfm exists in q0
if ~isfield(q0, 'q_irwfm')
    q0.q_irwfm = 0;
end

% Set equal weights if not specified
if isempty(cfg.horizon_weights)
    cfg.horizon_weights = ones(1, numel(cfg.target_horizons));
end

% Normalize weights to sum to 1
cfg.horizon_weights = cfg.horizon_weights / sum(cfg.horizon_weights);

% Calculate search bounds
factor = 10^cfg.search_range;

% WPM is fixed (measurement noise shouldn't change)
cfg.bounds.q_wpm_range = [q0.q_wpm, q0.q_wpm];

% WFM and RWFM search ranges
cfg.bounds.q_wfm_range  = [q0.q_wfm/factor, q0.q_wfm*factor];
cfg.bounds.q_rwfm_range = [q0.q_rwfm/factor, q0.q_rwfm*factor];

% IRWFM search range (only if used)
cfg.bounds.q_irwfm_range = [0, 0];
if cfg.nstates >= 3 && q0.q_irwfm > 0
    cfg.bounds.q_irwfm_range = [q0.q_irwfm/factor, q0.q_irwfm*factor];
end

end

function [q_opt, results] = optimize_grid(x, tau, q0, cfg)
% Grid search optimization over Q parameters

%% Build parameter grids
% Points per parameter: ensures odd number (includes center point)
nPts = 2 * cfg.search_range * cfg.n_grid_per_decade + 1;

% WPM is fixed at initial value
q_wpm = q0.q_wpm;

% Create log-spaced grids for parameters to optimize
q_wfm  = logspace(log10(cfg.bounds.q_wfm_range(1)), ...
                  log10(cfg.bounds.q_wfm_range(2)), nPts);
q_rwfm = logspace(log10(cfg.bounds.q_rwfm_range(1)), ...
                  log10(cfg.bounds.q_rwfm_range(2)), nPts);

% Handle IRWFM based on number of states
if cfg.nstates >= 3 && q0.q_irwfm > 0
    q_irwfm = logspace(log10(cfg.bounds.q_irwfm_range(1)), ...
                       log10(cfg.bounds.q_irwfm_range(2)), nPts);
    % Create 3D grid
    [QWFM, QRWFM, QIRWFM] = ndgrid(q_wfm, q_rwfm, q_irwfm);
    params = [repmat(q_wpm, numel(QWFM), 1), QWFM(:), QRWFM(:), QIRWFM(:)];
else
    % Create 2D grid
    [QWFM, QRWFM] = ndgrid(q_wfm, q_rwfm);
    params = [repmat(q_wpm, numel(QWFM), 1), QWFM(:), QRWFM(:), ...
              zeros(numel(QWFM), 1)];
end

nTotal = size(params, 1);

if cfg.verbose
    fprintf('Grid search: %d points, %d total evaluations\n', nPts, nTotal);
    fprintf('Using parallel processing...\n');
end

%% Evaluate all grid points
costs = zeros(nTotal, 1);
rmsAll = nan(nTotal, numel(cfg.target_horizons));

% Parallel evaluation of cost function
parfor k = 1:nTotal
    [costs(k), rmsAll(k, :)] = evaluate_prediction_cost(params(k, :), x, tau, cfg);
end

%% Find optimal parameters
[best_cost, idx] = min(costs);
q_opt = struct('q_wpm',  params(idx, 1), ...
               'q_wfm',  params(idx, 2), ...
               'q_rwfm', params(idx, 3), ...
               'q_irwfm', params(idx, 4));

%% Package results
results = struct('rms_opt',        rmsAll(idx, :), ...
                 'weighted_rms',   best_cost, ...
                 'search_history', [params, costs], ...
                 'method',         'grid', ...
                 'n_evaluations',  nTotal);

if cfg.verbose
    fprintf('\nOptimization complete. Best weighted RMS: %.3f\n', best_cost);
end

end

% function [q_opt, results] = optimize_fmincon(x, tau, q0, cfg)
% % Gradient-based optimization using fmincon
% 
% error('OPTIMIZE_KF:NotImplemented', ...
%       'fmincon optimization not yet implemented. Use method="grid".');
% 
% end

function [weighted_cost, rms_at_horizons] = evaluate_prediction_cost(q, x, tau, cfg)
% Evaluate prediction performance for given Q parameters
% Returns weighted RMS cost and individual RMS values at target horizons

% Initialize outputs with high penalty values (in case of failure)
nH = numel(cfg.target_horizons);
rms_at_horizons = 1e10 * ones(1, nH);
weighted_cost = 1e10;

% Setup parameters for KF prediction
noise = struct('q_wpm', q(1), 'q_wfm', q(2), 'q_rwfm', q(3), 'q_irwfm', q(4));
kf = struct('nstates', cfg.nstates, ...
            'maturity', cfg.maturity, ...
            'max_horizon', max(cfg.target_horizons));
pred = struct('g_p', 0, 'g_i', 0, 'g_d', 0, ...  % No steering for optimization
              'init_cov', 1e30, ...
              'save_predictions', false, ...
              'verbose', false);

% Run prediction analysis
try
    out = kf_predict(x, tau, noise, kf, pred);
    
    % Extract RMS at target horizons
    for k = 1:nH
        h = cfg.target_horizons(k);
        idx = find(out.rms_stats.horizon == h, 1);
        if ~isempty(idx)
            rms_at_horizons(k) = out.rms_stats.rms_error(idx);
        end
    end
    
    % Calculate weighted cost
    weighted_cost = sum(cfg.horizon_weights(:) .* rms_at_horizons(:));
    
catch ME
    % Keep default penalty values if KF fails
    if cfg.verbose
        warning('OPTIMIZE_KF:EvaluationFailed', ...
                'KF failed for q=[%.2e, %.2e, %.2e, %.2e]: %s', ...
                q(1), q(2), q(3), q(4), ME.message);
    end
end

end

%% ===================== fmincon Optimization =====================

function [q_opt, results] = optimize_fmincon(phase_data, tau, q_initial, cfg)
% Gradient-based optimization using fmincon

%% Determine which parameters to optimize
all_param_names = {'q_wfm', 'q_rwfm', 'q_irwfm'};
param_names = {};
factor = 10^cfg.search_range;

for i = 1:length(all_param_names)
    name = all_param_names{i};
    val = q_initial.(name);
    
    if val > 1e-30
        param_names{end+1} = name;
    else
        if cfg.verbose
            fprintf('Skipping optimization of %s (value = %.3e)\n', name, val);
        end
    end
end

%% Set up optimization variables  
x0 = [];
lb = [];
ub = [];

for i = 1:length(param_names)
    name = param_names{i};
    val = q_initial.(name);
    
    x0(i) = log10(val);
    lb(i) = log10(val / factor);
    ub(i) = log10(val * factor);
    
    if cfg.verbose
        fprintf('Optimizing %s: initial=%.3e, bounds=[%.3e, %.3e]\n', ...
                name, val, val/factor, val*factor);
    end
end

%% Define cost function
cost_function = @(x) evaluate_cost_fmincon(x, phase_data, tau, q_initial, param_names, cfg);

%% Run optimization
options = optimoptions('fmincon', ...
    'Display', 'iter-detailed', ...
    'MaxIterations', 200, ...
    'MaxFunctionEvaluations', 1000, ...
    'TolX', 1e-6, ...
    'TolFun', 1e-6, ...
    'Algorithm', 'interior-point', ...
    'FiniteDifferenceType', 'central', ...
    'FiniteDifferenceStepSize', 1e-6);

if cfg.verbose
    fprintf('\nStarting fmincon optimization with %d parameters...\n', length(param_names));
end

[x_opt, fval, exitflag, output] = fmincon(cost_function, x0, [], [], [], [], lb, ub, [], options);

%% Convert back to linear scale
q_opt = q_initial;  % Start with original values
for i = 1:length(param_names)
    name = param_names{i};
    q_opt.(name) = 10^x_opt(i);
end

%% Test optimized parameters to get RMS values
q_array = [q_opt.q_wpm, q_opt.q_wfm, q_opt.q_rwfm, q_opt.q_irwfm];
[final_cost, rms_at_horizons] = evaluate_prediction_cost(q_array, phase_data, tau, cfg);

%% Package results
results = struct('rms_opt', rms_at_horizons, ...
                 'weighted_rms', final_cost, ...
                 'fval', fval, ...
                 'exitflag', exitflag, ...
                 'output', output, ...
                 'method', 'fmincon', ...
                 'n_evaluations', output.funcCount, ...
                 'search_history', []);  % fmincon doesn't provide search history

if cfg.verbose
    fprintf('\nfmincon optimization complete:\n');
    fprintf('  Exit flag: %d\n', exitflag);
    fprintf('  Function evaluations: %d\n', output.funcCount);
    fprintf('  Final cost: %.6f\n', fval);
end

end

function cost = evaluate_cost_fmincon(x_log, phase_data, tau, q_initial, param_names, cfg)
% Cost function wrapper for fmincon

try
    % Build noise parameters  
    noise_params = q_initial;
    for i = 1:length(param_names)
        noise_params.(param_names{i}) = 10^x_log(i);
    end
    
    % Convert to parameter array format for cost function
    q_array = [noise_params.q_wpm, noise_params.q_wfm, noise_params.q_rwfm, noise_params.q_irwfm];
    
    % Evaluate cost
    [cost, ~] = evaluate_prediction_cost(q_array, phase_data, tau, cfg);
    
catch ME
    cost = 1e10;  % High penalty for failed evaluations
    if cfg.verbose > 1
        warning('Cost evaluation failed');
    end
end

end