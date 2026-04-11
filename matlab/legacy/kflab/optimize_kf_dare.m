function [q_opt, results] = optimize_kf_dare(phase_data, tau, q_initial, opt_config)
% OPTIMIZE_KF_DARE - Optimize Kalman filter Q parameters using DARE approach
%
% Uses kf_predict_dare for fast steady-state predictions while maintaining
% the same cost function methodology as the original optimize_kf.

%% Initialize configuration with defaults (same as original optimize_kf)
cfg = set_optimization_defaults(opt_config, q_initial);

if cfg.verbose
    fprintf('\n=== KF Q PARAMETER OPTIMIZATION (DARE Method) ===\n');
    fprintf('Search ±%d decades ; method = %s\n', cfg.search_range, cfg.method);
    fprintf('WPM fixed at: %.3e\n', q_initial.q_wpm);
end

%% Run optimization
tic;
switch lower(cfg.method)
    case 'grid'
        [q_opt, results] = optimize_grid_dare(phase_data, tau, q_initial, cfg);
    case 'fmincon'
        [q_opt, results] = optimize_fmincon_dare(phase_data, tau, q_initial, cfg);
    otherwise
        error('OPTIMIZE_KF_DARE:UnknownMethod', 'Unknown optimization method "%s".', cfg.method);
end
results.optimisation_time = toc;

end

%% ===================== Grid Search with DARE =====================

function [q_opt, results] = optimize_grid_dare(x, tau, q0, cfg)
% Grid search optimization using kf_predict_dare

%% Build parameter grids (same as original)
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
    fprintf('Grid search: %d points, %d total evaluations (DARE method)\n', nPts, nTotal);
    fprintf('Using parallel processing...\n');
end

%% Evaluate all grid points using DARE
costs = zeros(nTotal, 1);
rmsAll = nan(nTotal, numel(cfg.target_horizons));

% Parallel evaluation of cost function
parfor k = 1:nTotal
    [costs(k), rmsAll(k, :)] = evaluate_prediction_cost_dare(params(k, :), x, tau, cfg);
end

%% Find optimal parameters (same as original)
[best_cost, idx] = min(costs);
q_opt = struct('q_wpm',  params(idx, 1), ...
               'q_wfm',  params(idx, 2), ...
               'q_rwfm', params(idx, 3), ...
               'q_irwfm', params(idx, 4));

%% Package results (same as original)
results = struct('rms_opt',        rmsAll(idx, :), ...
                 'weighted_rms',   best_cost, ...
                 'search_history', [params, costs], ...
                 'method',         'grid-dare', ...
                 'n_evaluations',  nTotal);

if cfg.verbose
    fprintf('\nOptimization complete. Best weighted RMS: %.3f\n', best_cost);
end

end

%% ===================== fmincon with DARE =====================

function [q_opt, results] = optimize_fmincon_dare(phase_data, tau, q_initial, cfg)
% Gradient-based optimization using fmincon with DARE

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

%% Define cost function using DARE
cost_function = @(x) evaluate_cost_fmincon_dare(x, phase_data, tau, q_initial, param_names, cfg);

%% Run optimization
options = optimoptions('fmincon', ...
    'Display', 'iter-detailed', ...
    'MaxIterations', 200, ...
    'TolX', 1e-6, ...
    'TolFun', 1e-6);

fprintf('\nStarting fmincon optimization with %d parameters (DARE method)...\n', length(param_names));
[x_opt, fval, exitflag, output] = fmincon(cost_function, x0, [], [], [], [], lb, ub, [], options);

%% Convert back to linear scale
q_opt = q_initial;  % Start with original values
for i = 1:length(param_names)
    name = param_names{i};
    q_opt.(name) = 10^x_opt(i);
end

%% Package results
results = struct('fval', fval, ...
                 'exitflag', exitflag, ...
                 'output', output, ...
                 'method', 'fmincon-dare', ...
                 'n_evaluations', output.funcCount);

end

%% ===================== Cost Functions =====================

function [weighted_cost, rms_at_horizons] = evaluate_prediction_cost_dare(q, x, tau, cfg)
% Evaluate prediction performance using kf_predict_dare (same interface as original)

% Initialize outputs with high penalty values (same as original)
nH = numel(cfg.target_horizons);
rms_at_horizons = 1e10 * ones(1, nH);
weighted_cost = 1e10;

% Setup parameters for KF prediction (same as original)
noise = struct('q_wpm', q(1), 'q_wfm', q(2), 'q_rwfm', q(3), 'q_irwfm', q(4));
kf = struct('nstates', cfg.nstates, ...
            'maturity', cfg.maturity, ...
            'max_horizon', max(cfg.target_horizons));
pred = struct('g_p', 0, 'g_i', 0, 'g_d', 0, ...  % No steering for optimization
              'init_cov', 1e30, ...
              'save_predictions', false, ...
              'verbose', false);

% Run prediction analysis using DARE method
try
    out = kf_predict_dare(x, tau, noise, kf, pred);  % ← Use DARE version
    
    % Extract RMS at target horizons (same as original)
    for k = 1:nH
        h = cfg.target_horizons(k);
        idx = find(out.rms_stats.horizon == h, 1);
        if ~isempty(idx)
            rms_at_horizons(k) = out.rms_stats.rms_error(idx);
        end
    end
    
    % Calculate weighted cost (same as original)
    weighted_cost = sum(cfg.horizon_weights(:) .* rms_at_horizons(:));
    
catch ME
    % Keep default penalty values if KF fails (same as original)
    if cfg.verbose
        warning('OPTIMIZE_KF_DARE:EvaluationFailed', ...
                'KF failed for q=[%.2e, %.2e, %.2e, %.2e]: %s', ...
                q(1), q(2), q(3), q(4), ME.message);
    end
end

end

function cost = evaluate_cost_fmincon_dare(x_log, phase_data, tau, q_initial, param_names, cfg)
% Cost function wrapper for fmincon

try
    % Build noise parameters
    noise_params = q_initial;
    for i = 1:length(param_names)
        noise_params.(param_names{i}) = 10^x_log(i);
    end
    
    % Convert to parameter array format for cost function
    q_array = [noise_params.q_wpm, noise_params.q_wfm, noise_params.q_rwfm, noise_params.q_irwfm];
    
    % Evaluate cost using DARE method
    [cost, ~] = evaluate_prediction_cost_dare(q_array, phase_data, tau, cfg);
    
catch ME
    cost = 1e10;
    if cfg.verbose
        warning('Cost evaluation failed: %s', ME.message);
    end
end

end

%% ===================== Configuration (Same as Original) =====================

function cfg = set_optimization_defaults(user_config, q0)
% Set default configuration values and merge with user settings (SAME AS ORIGINAL)

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