function results = main_kf_pipeline_unified(pipeline_mode, data_config, opt_config, output_config)
% MAIN_KF_PIPELINE_UNIFIED - Unified Kalman filter pipeline for all data types
%
% This consolidates the 4 separate pipeline functions into one simple function
% with a mode parameter to select the data type.
%
% Usage:
%   results = main_kf_pipeline_unified(mode)  % Interactive mode
%   results = main_kf_pipeline_unified(mode, data_config, opt_config, output_config)
%
% Modes:
%   'raw'              - Raw phase data, compute MHDEV (default, faster)
%   'raw_mhtotdev'     - Raw phase data, compute MHTOTDEV (more intensive)
%   'precomputed'      - Pre-computed MHDEV data
%   'precomputed_mhtotdev'- Pre-computed MHTOTDEV data
%   'noise'            - Skip deviation computation, use provided noise parameters
%
% Inputs: Same as original functions
%   data_config  - Data configuration struct
%   opt_config   - Optimization configuration struct  
%   output_config - Output configuration struct
%
% Output:
%   results - Complete results structure

    % Validate pipeline mode
    valid_modes = {'raw', 'raw_mhtotdev', 'precomputed', 'precomputed_mhtotdev', 'noise'};
    if ~ismember(pipeline_mode, valid_modes)
        error('Invalid pipeline mode. Must be one of: %s', strjoin(valid_modes, ', '));
    end
    
    % Handle interactive vs programmatic mode
    if nargin == 1
        % Interactive mode - prompt for configurations
        [data_config, opt_config, output_config] = prompt_user_config(pipeline_mode);
    elseif nargin ~= 4
        error('Must provide either 1 argument (interactive) or 4 arguments (programmatic)');
    end
    
    % Create unified config - no defaults, user must specify required fields
    config = struct();
    
    % Required fields that must be provided by user - set empty as placeholders
    % Data fields
    config.data_file = '';
    config.data_column = 1;  % Column 1 unless specified
    config.tau0 = 1.0;  % Default sampling interval
    config.data_name = '';  % REQUIRED
    config.conv_factor = 1;
    config.fit_max_iterations = 6;
    config.mhtotdev_m_list = [];
    
    % Optimization fields
    config.opt = struct();
    config.opt.search_range = [];  % REQUIRED
    config.opt.n_grid_per_decade = 5;
    config.opt.nstates = [];  % REQUIRED
    config.opt.target_horizons = [];  % REQUIRED
    config.opt.horizon_weights = [];  % Will be set to equal weights
    config.opt.method = 'grid';
    config.opt.maturity = [];  % REQUIRED
    
    % Output fields
    config.save_results = false;  % Don't save by default
    config.save_plots = false;    % Don't save by default
    config.results_dir = '';
    config.verbose = false;
    
    % Merge user data_config
    if ~isempty(data_config)
        fields = fieldnames(data_config);
        for i = 1:length(fields)
            config.(fields{i}) = data_config.(fields{i});
        end
    end
    
    % Merge user opt_config
    if ~isempty(opt_config)
        fields = fieldnames(opt_config);
        for i = 1:length(fields)
            config.opt.(fields{i}) = opt_config.(fields{i});
        end
    end
    
    % Merge user output_config
    if ~isempty(output_config)
        fields = fieldnames(output_config);
        for i = 1:length(fields)
            config.(fields{i}) = output_config.(fields{i});
        end
    end
    
    % Validation - check required fields are provided
    % tau0 now has default of 1.0, so no validation needed
    if isempty(config.data_name)
        error('data_name is required');
    end
    if isempty(config.opt.nstates)
        error('opt.nstates is required (2, 3, or 5)');
    end
    if isempty(config.opt.target_horizons)
        error('opt.target_horizons is required');
    end
    if isempty(config.opt.maturity)
        error('opt.maturity is required');
    end
    
    % Special validation for noise mode
    if strcmp(pipeline_mode, 'noise')
        if ~isfield(config, 'q_wpm') || ~isfield(config, 'q_wfm') || ~isfield(config, 'q_rwfm')
            error('For noise mode, q_wpm, q_wfm, and q_rwfm are required in data_config');
        end
        if isempty(config.data_file)
            error('For noise mode, data_file is required to load phase data');
        end
        % search_range IS needed for noise mode optimization bounds
        if isempty(config.opt.search_range)
            error('opt.search_range is required for optimization bounds');
        end
    else
        if isempty(config.opt.search_range)
            error('opt.search_range is required');
        end
    end
    
    % Value validation
    if config.tau0 <= 0
        error('tau0 must be positive');
    end
    if ~ismember(config.opt.nstates, [2, 3, 5])
        error('nstates must be 2, 3, or 5');
    end
    if any(config.opt.target_horizons <= 0)
        error('target_horizons must be positive');
    end
    
    % Ensure horizon weights match horizons
    if length(config.opt.horizon_weights) ~= length(config.opt.target_horizons)
        config.opt.horizon_weights = ones(1, length(config.opt.target_horizons));
    end
    
    % Setup results directory if saving
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    if config.save_results && isempty(config.results_dir)
        % Create two-level directory structure: name+date/timestamp/
        base_dir = fullfile('results', ...
            sprintf('%s_%s', config.data_name, datestr(now, 'yyyy-mm-dd')));
        config.results_dir = fullfile(base_dir, timestamp);
        if ~exist(config.results_dir, 'dir')
            mkdir(config.results_dir);
        end
    end
    
    %% Step 1: Load data based on mode
    fprintf('\n=== Kalman Filter Pipeline (%s mode) ===\n', pipeline_mode);
    
    if strcmp(pipeline_mode, 'noise')
        fprintf('Step 1: Loading phase data (skipping deviation computation)...\n');
        [phase_data, tau, deviation_vals, ci, deviation_type] = load_noise_mode_data(config);
    else
        fprintf('Step 1: Loading data...\n');
        switch pipeline_mode
            case 'raw'
                [phase_data, tau, deviation_vals, ci, deviation_type] = load_raw_data(config, 'mhdev');
            case 'raw_mhtotdev'
                [phase_data, tau, deviation_vals, ci, deviation_type] = load_raw_data(config, 'mhtotdev');
            case 'precomputed'
                [phase_data, tau, deviation_vals, ci, deviation_type] = load_precomputed_data(config, 'mhdev');
            case 'precomputed_mhtotdev'
                [phase_data, tau, deviation_vals, ci, deviation_type] = load_precomputed_data(config, 'mhtotdev');
        end
    end
    
    % Extract CI bounds and prepare for fitting functions
    % Ensure all vectors are column vectors first
    tau = tau(:);
    deviation_vals = deviation_vals(:);
    
    if ~isempty(ci) && size(ci, 2) == 2
        ci_low = ci(:, 1);
        ci_high = ci(:, 2);
    elseif ~isempty(ci) && size(ci, 1) == 2
        % Handle case where ci is [2 x N] instead of [N x 2]
        ci_low = ci(1, :)';
        ci_high = ci(2, :)';
    else
        % Default CI if none provided
        ci_low = deviation_vals * 0.9;
        ci_high = deviation_vals * 1.1;
    end
    
    % Ensure CI vectors are column vectors and package for fitting
    ci_low = ci_low(:);
    ci_high = ci_high(:);
    ci_for_fit = [ci_low, ci_high];  % [lower, upper] format for fitting functions
    
    %% Step 2: Fit noise components or use provided values
    if strcmp(pipeline_mode, 'noise')
        fprintf('\nStep 2: Using provided noise parameters...\n');
        
        % Validate that noise parameters are provided
        if ~isfield(config, 'q_wpm') || ~isfield(config, 'q_wfm') || ~isfield(config, 'q_rwfm')
            error('For noise mode, must provide q_wpm, q_wfm, and q_rwfm in data_config');
        end
        
        q0_fit = config.q_wpm;
        q1_fit = config.q_wfm;
        q2_fit = config.q_rwfm;
        
        if isfield(config, 'q_irwfm')
            q_irwfm = config.q_irwfm;
        else
            q_irwfm = 0;
        end
        
        fprintf('  q_wpm:  %.3e\n', q0_fit);
        fprintf('  q_wfm:  %.3e\n', q1_fit);
        fprintf('  q_rwfm: %.3e\n', q2_fit);
        if q_irwfm > 0
            fprintf('  q_irwfm: %.3e\n', q_irwfm);
        end
        
        fit_regions = [];  % No fitting regions for noise mode
    else
        fprintf('\nStep 2: Fitting noise components...\n');
        fprintf('  Interactive process - follow prompts\n\n');
        
        % Use appropriate fitting function
        if contains(deviation_type, 'mhtotdev')
            [q0_fit, q1_fit, q2_fit, fit_regions] = mhtot_fit(deviation_vals, tau, ci_for_fit, ...
                                                               config.fit_max_iterations);
        else
            [q0_fit, q1_fit, q2_fit, fit_regions] = mhdev_fit(deviation_vals, tau, ci_for_fit, ...
                                                              config.fit_max_iterations);
        end
        
        q_irwfm = 0;  % Default for fitted parameters
    end
    
    % Create noise parameter struct
    noise_params = struct('q_wpm', q0_fit, ...
                         'q_wfm', q1_fit, ...
                         'q_rwfm', q2_fit, ...
                         'q_irwfm', q_irwfm);
    
    %% Step 3: Initial KF performance
    fprintf('\nStep 3: Testing initial KF performance...\n');
    
    % Setup KF parameters
    kf_params = struct('nstates', config.opt.nstates, ...
                      'maturity', config.opt.maturity, ...
                      'max_horizon', max(config.opt.target_horizons));
    
    pred_params = struct('g_p', 0, 'g_i', 0, 'g_d', 0, ...
                        'init_cov', 1e30, ...
                        'verbose', config.verbose);
    
    % Run initial prediction test
    initial_results = kf_predict(phase_data, config.tau0, noise_params, kf_params, pred_params);
    
    % Compute steady-state covariances for initial parameters (silent)
    initial_covariance_info = compute_covariance_uncertainties_silent(noise_params, config.tau0, config.opt.nstates);
    
    %% Step 4: Optimize Q parameters
    fprintf('\nStep 4: Optimizing Q parameters...\n');
    
    % Use fitted values as initial guess
    q_initial = struct('q_wpm', q0_fit, ...
                      'q_wfm', q1_fit, ...
                      'q_rwfm', q2_fit, ...
                      'q_irwfm', 0);
    
    % Run optimization
    [q_optimal, opt_results] = optimize_kf(phase_data, config.tau0, q_initial, config.opt);
    
    %% Step 5: Test optimized KF performance
    fprintf('\nStep 5: Testing optimized KF performance...\n');
    
    % Update noise parameters with optimized values
    noise_params_opt = struct('q_wpm', q_optimal.q_wpm, ...
                             'q_wfm', q_optimal.q_wfm, ...
                             'q_rwfm', q_optimal.q_rwfm, ...
                             'q_irwfm', q_optimal.q_irwfm);
    
    % Run optimized prediction test
    final_results = kf_predict(phase_data, config.tau0, noise_params_opt, kf_params, pred_params);
    
    %% Display and save results
    display_results_summary(config, q_initial, q_optimal, initial_results, final_results);
    
    % Create plots
    if config.save_plots || config.verbose
        if ~strcmp(pipeline_mode, 'noise')
            % Only create deviation plots for non-noise modes
            create_result_plots(config, timestamp, tau, deviation_vals, ci_low, ci_high, ...
                               q0_fit, q1_fit, q2_fit, deviation_type, opt_results, final_results);
        end
        
        % Create optimization plots for all modes
        create_optimization_plots(config, timestamp, noise_params, q_optimal, ...
                                 initial_results, final_results, opt_results);
    end
    
    % Compute steady-state covariance uncertainties and display comparison
    covariance_info = compute_covariance_uncertainties_silent(noise_params_opt, config.tau0, config.opt.nstates);
    
    % Display comparison of initial vs optimized steady-state uncertainties
    fprintf('\nSteady-state uncertainties (1-sigma) comparison:\n');
    fprintf('                    Initial      Optimized    Improvement\n');
    fprintf('  Phase:          %8.3f    %8.3f    %8.1f%%\n', ...
            initial_covariance_info.sigma_phase, covariance_info.sigma_phase, ...
            100 * (initial_covariance_info.sigma_phase - covariance_info.sigma_phase) / initial_covariance_info.sigma_phase);
    fprintf('  Frequency:      %8.3e  %8.3e  %8.1f%%\n', ...
            initial_covariance_info.sigma_freq, covariance_info.sigma_freq, ...
            100 * (initial_covariance_info.sigma_freq - covariance_info.sigma_freq) / initial_covariance_info.sigma_freq);
    if config.opt.nstates >= 3
        fprintf('  Drift:          %8.3e  %8.3e  %8.1f%%\n', ...
                initial_covariance_info.sigma_drift, covariance_info.sigma_drift, ...
                100 * (initial_covariance_info.sigma_drift - covariance_info.sigma_drift) / initial_covariance_info.sigma_drift);
    end
    
    % Package results
    results = package_results(config, timestamp, phase_data, tau, deviation_vals, ...
                             ci_low, ci_high, noise_params, q_optimal, ...
                             initial_results, final_results, opt_results, deviation_type, ...
                             initial_covariance_info, covariance_info);
    
    % Save results
    if config.save_results
        save_all_results(config, timestamp, results, tau, deviation_vals, ci_low, ci_high);
    end
    
    fprintf('\n=== Pipeline Complete ===\n');
end

%% Helper Functions

function display_horizons = subsample_horizons_for_display(horizons, max_display)
    % Intelligently subsample horizons for display when there are too many
    % 
    % Inputs:
    %   horizons - Array of horizon values
    %   max_display - Maximum number of horizons to display (default 20)
    %
    % Output:
    %   display_horizons - Subsampled horizons for display
    
    if nargin < 2
        max_display = 20;
    end
    
    if length(horizons) <= max_display
        % No subsampling needed
        display_horizons = horizons;
        return;
    end
    
    % Strategy: Include first, last, and evenly spaced points in between
    % This ensures we capture the full range while keeping readability
    
    % Always include first and last
    first_horizon = horizons(1);
    last_horizon = horizons(end);
    
    % Calculate indices for evenly spaced points
    n_interior = max_display - 2;  % Reserve 2 spots for first and last
    if n_interior < 1
        display_horizons = [first_horizon, last_horizon];
        return;
    end
    
    % Create evenly spaced indices
    indices = round(linspace(2, length(horizons)-1, n_interior));
    
    % Combine and sort
    display_horizons = unique([first_horizon, horizons(indices), last_horizon]);
end

function [phase_data, tau, deviation_vals, ci, deviation_type] = load_raw_data(config, dev_type)
    % Load raw phase data and compute deviation
    
    % Load phase data
    phase_data = load(config.data_file);
    if size(phase_data, 2) > 1
        phase_data = phase_data(:, config.data_column);
    end
    phase_data = phase_data(:) * config.conv_factor;
    
    N = length(phase_data);
    fprintf('  Loaded %d samples (duration: %.2f, rate: %.1f)\n', ...
            N, N * config.tau0 / 3600, 1/config.tau0);
    
    % Compute deviation
    fprintf('  Computing %s...\n', upper(dev_type));
    tic;
    
    if strcmp(dev_type, 'mhtotdev')
        [tau, deviation_vals, edf, ci, alpha] = ...
            allanlab.mhtotdev_par(phase_data, config.tau0, config.mhtotdev_m_list);
        deviation_type = 'mhtotdev';
    else
        [tau, deviation_vals, edf, ci, alpha] = ...
            allanlab.mhdev(phase_data, config.tau0, config.mhtotdev_m_list);
        deviation_type = 'mhdev';
    end
    
    compute_time = toc;
    fprintf('  Computed %d tau values in %.1f seconds\n', length(tau), compute_time);
    
    % Ensure all vectors are column vectors (critical for fitting functions)
    tau = tau(:);
    deviation_vals = deviation_vals(:);
end

function [phase_data, tau, deviation_vals, ci, deviation_type] = load_precomputed_data(config, dev_type)
    % Load pre-computed deviation data
    
    deviation_type = dev_type;
    
    % Load phase data if provided
    if isfield(config, 'phase_file') && ~isempty(config.phase_file)
        phase_data = load(config.phase_file);
        if size(phase_data, 2) > 1
            phase_data = phase_data(:, 1);
        end
        phase_data = phase_data(:) * config.conv_factor;
        fprintf('  Loaded phase data: %d samples\n', length(phase_data));
    else
        phase_data = [];
        fprintf('  No phase data provided\n');
    end
    
    % Load deviation data - handle both combined and separate files
    if isfield(config, 'mhtotdev_file') || isfield(config, 'mhdev_file')
        % Combined file format
        if isfield(config, 'mhtotdev_file')
            data = load(config.mhtotdev_file);
        else
            data = load(config.mhdev_file);
        end
        
        tau = data(:, 1);
        deviation_vals = data(:, 2);
        
        if size(data, 2) >= 4
            ci = [data(:, 3), data(:, 4)];
        else
            ci = [];
        end
    else
        % Separate files format
        tau = load(config.tau_file);
        if isfield(config, 'mhtotdev_values_file')
            deviation_vals = load(config.mhtotdev_values_file);
        else
            deviation_vals = load(config.mhdev_values_file);
        end
        
        if isfield(config, 'ci_file') && ~isempty(config.ci_file)
            ci = load(config.ci_file);
        else
            ci = [];
        end
    end
    
    fprintf('  Loaded pre-computed %s: %d tau values\n', deviation_type, length(tau));
    
    % Ensure all vectors are column vectors (critical for fitting functions)
    tau = tau(:);
    deviation_vals = deviation_vals(:);
    if ~isempty(ci)
        if size(ci, 1) == 2 && size(ci, 2) > 2
            ci = ci';  % Transpose if it's [2 x N] instead of [N x 2]
        end
    end
end

function display_results_summary(config, q_initial, q_optimal, initial_results, final_results)
    % Display optimization results summary
    
    fprintf('\n=== OPTIMIZATION RESULTS ===\n');
    fprintf('Initial Q parameters (from fit):\n');
    fprintf('  q_wpm:  %.3e\n', q_initial.q_wpm);
    fprintf('  q_wfm:  %.3e\n', q_initial.q_wfm);
    fprintf('  q_rwfm: %.3e\n', q_initial.q_rwfm);
    
    fprintf('\nOptimized Q parameters:\n');
    fprintf('  q_wpm:  %.3e (fixed)\n', q_optimal.q_wpm);
    fprintf('  q_wfm:  %.3e\n', q_optimal.q_wfm);
    fprintf('  q_rwfm: %.3e\n', q_optimal.q_rwfm);
    
    % Subsample horizons if there are too many
    display_horizons = subsample_horizons_for_display(config.opt.target_horizons, 20);
    
    fprintf('\nPrediction RMS comparison');
    if length(config.opt.target_horizons) > length(display_horizons)
        fprintf(' (showing %d of %d horizons)', length(display_horizons), length(config.opt.target_horizons));
    end
    fprintf(':\n');
    fprintf('Horizon | Initial    | Optimized  | Improvement\n');
    fprintf('--------|------------|------------|-------------\n');
    
    for h = display_horizons
        idx_init = find(initial_results.rms_stats.horizon == h, 1);
        idx_opt = find(final_results.rms_stats.horizon == h, 1);
        
        if ~isempty(idx_init) && ~isempty(idx_opt)
            rms_init = initial_results.rms_stats.rms_error(idx_init);
            rms_opt = final_results.rms_stats.rms_error(idx_opt);
            improvement = 100 * (rms_init - rms_opt) / rms_init;
            
            fprintf('%7d | %10.3f | %10.3f | %10.1f%%\n', ...
                    h, rms_init, rms_opt, improvement);
        end
    end
    
    if length(config.opt.target_horizons) > length(display_horizons)
        fprintf('\nNote: Full results for all %d horizons are saved in the output files.\n', ...
                length(config.opt.target_horizons));
    end
end

function create_result_plots(config, timestamp, tau, deviation_vals, ci_low, ci_high, ...
                            q0_fit, q1_fit, q2_fit, deviation_type, opt_results, final_results)
    % Create and save result plots
    
    % Plot 1: Deviation fit
    fig1 = figure('Name', sprintf('%s Fit', upper(deviation_type)), 'Position', [100, 100, 800, 600]);
    loglog(tau, deviation_vals, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 8);
    hold on;
    
    % Add confidence intervals
    fill([tau; flipud(tau)], [ci_low; flipud(ci_high)], ...
         'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    
    % Add fitted noise components
    tau_fit = logspace(log10(min(tau)), log10(max(tau)), 1000);
    if contains(deviation_type, 'mhtotdev')
        sigma2_wpm = (10/3) * q0_fit * tau_fit.^(-3);
        sigma2_wfm = (4/7) * q1_fit * tau_fit.^(-1);
        sigma2_rwfm = (5/22) * q2_fit * tau_fit;
    else
        % MHDEV coefficients would be different
        sigma2_wpm = q0_fit * tau_fit.^(-3);
        sigma2_wfm = q1_fit * tau_fit.^(-1);
        sigma2_rwfm = q2_fit * tau_fit;
    end
    sigma2_total = sigma2_wpm + sigma2_wfm + sigma2_rwfm;
    
    plot(tau_fit, sqrt(sigma2_wpm), 'r--', 'LineWidth', 1.5, 'DisplayName', 'WPM');
    plot(tau_fit, sqrt(sigma2_wfm), 'g--', 'LineWidth', 1.5, 'DisplayName', 'WFM');
    plot(tau_fit, sqrt(sigma2_rwfm), 'm--', 'LineWidth', 1.5, 'DisplayName', 'RWFM');
    plot(tau_fit, sqrt(sigma2_total), 'k-', 'LineWidth', 2, 'DisplayName', 'Total fit');
    
    xlabel('Averaging Time τ [s]');
    ylabel(sprintf('%s [ns]', upper(deviation_type)));
    title(sprintf('%s - Noise Component Fit', config.data_name));
    legend('Data', 'Confidence', 'Location', 'best');
    grid on;
    
    if config.save_plots
        saveas(fig1, fullfile(config.results_dir, sprintf('%s_fit_%s.png', deviation_type, timestamp)));
        saveas(fig1, fullfile(config.results_dir, sprintf('%s_fit_%s.fig', deviation_type, timestamp)));
    end
    
    % This is now handled by create_optimization_plots above
end

function create_optimization_plots(config, timestamp, q_initial, q_optimal, initial_results, final_results, opt_results)
    % Create optimization plots based on the original plot_optimization.m format
    
    % Create figure with 6-panel layout
    figure('Name', 'KF Q-Parameter Optimization Summary', ...
           'Units', 'normalized', ...
           'OuterPosition', [0.03 0.05 0.94 0.88]);

    tl = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    %% Panel 1: RMS prediction error vs horizon
    nexttile(tl);
    plot_rms_comparison(initial_results, final_results, config.opt);

    %% Panel 2: Percentage improvement
    nexttile(tl);
    plot_improvement(initial_results, final_results);

    %% Panel 3: Q-parameter bar chart
    nexttile(tl);
    plot_parameter_comparison(q_initial, q_optimal);

    %% Panels 4-6: Search space visualization (if search history available)
    if isfield(opt_results, 'search_history') && ~isempty(opt_results.search_history)
        % Panel 4: 2D cost surface (q_wfm vs q_rwfm)
        nexttile(tl);
        plot_2d_cost_surface(opt_results.search_history, q_initial, q_optimal);
        
        % Panel 5: Cost vs q_wfm (at optimal q_rwfm)
        nexttile(tl);
        plot_1d_slice(opt_results.search_history, q_initial, q_optimal, 'wfm');
        
        % Panel 6: Cost vs q_rwfm (at optimal q_wfm)
        nexttile(tl);
        plot_1d_slice(opt_results.search_history, q_initial, q_optimal, 'rwfm');
    else
        % Fill remaining panels with text for fmincon or other methods
        for i = 4:6
            nexttile(tl);
            text(0.5, 0.5, {'Search history not available', '(fmincon method)'}, ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'center', ...
                 'FontSize', 12);
            set(gca, 'XTick', [], 'YTick', []);
        end
    end

    %% Add overall title with improvement metric
    add_summary_title(initial_results, final_results);
    
    % Save if requested
    if config.save_plots
        saveas(gcf, fullfile(config.results_dir, sprintf('optimization_results_%s.png', timestamp)));
        saveas(gcf, fullfile(config.results_dir, sprintf('optimization_results_%s.fig', timestamp)));
    end
end

%% Helper functions for optimization plots (based on original plot_optimization.m)

function plot_rms_comparison(res0, resOpt, cfg)
    % Plot RMS error vs prediction horizon for initial and optimized parameters
    
    % Plot main curves
    plot(res0.rms_stats.horizon, res0.rms_stats.rms_error, '-r', ...
         'LineWidth', 1.6, 'DisplayName', 'Initial');
    hold on;
    plot(resOpt.rms_stats.horizon, resOpt.rms_stats.rms_error, '-b', ...
         'LineWidth', 1.6, 'DisplayName', 'Optimized');
    
    % Highlight target horizons if provided
    if isfield(cfg, 'target_horizons')
        % Subsample horizons if there are too many markers
        display_horizons = subsample_horizons_for_display(cfg.target_horizons, 25);
        
        for h = display_horizons(:)'
            idx = find(resOpt.rms_stats.horizon == h, 1);
            if ~isempty(idx)
                plot(h, resOpt.rms_stats.rms_error(idx), 'bo', ...
                     'MarkerSize', 7, 'LineWidth', 1.2, ...
                     'HandleVisibility', 'off');
            end
        end
        % Add one marker to legend with appropriate label
        if length(cfg.target_horizons) > length(display_horizons)
            label_str = sprintf('Target horizons (%d of %d shown)', ...
                               length(display_horizons), length(cfg.target_horizons));
        else
            label_str = 'Target horizons';
        end
        plot(NaN, NaN, 'bo', 'MarkerSize', 7, 'LineWidth', 1.2, ...
             'DisplayName', label_str);
    end
    
    % Formatting
    xlabel('τ (prediction horizon) [samples]');
    ylabel('RMS error');
    title('Prediction Performance');
    grid on;
    legend('Location', 'northwest');
    set(gca, 'XScale', 'log', 'YScale', 'log');
end

function plot_improvement(res0, resOpt)
    % Plot percentage improvement vs horizon
    
    % Find common horizons
    common_h = intersect(res0.rms_stats.horizon, resOpt.rms_stats.horizon);
    
    % Interpolate RMS values at common horizons
    rms0 = interp1(res0.rms_stats.horizon, res0.rms_stats.rms_error, common_h);
    rmsOpt = interp1(resOpt.rms_stats.horizon, resOpt.rms_stats.rms_error, common_h);
    
    % Calculate percentage improvement
    improvement = 100 * (rms0 - rmsOpt) ./ rms0;
    
    % Plot
    plot(common_h, improvement, '-g', 'LineWidth', 1.6);
    hold on;
    yline(0, 'k:', 'LineWidth', 1);
    
    % Formatting
    xlabel('τ (prediction horizon) [samples]');
    ylabel('Improvement [%]');
    title('Optimization Gain');
    grid on;
    set(gca, 'XScale', 'log');
    xlim([min(common_h), max(common_h)]);
end

function plot_parameter_comparison(q0, qOpt)
    % Bar chart comparing initial and optimized Q parameters
    
    % Parameter names and values
    names = {'q_{wpm}', 'q_{wfm}', 'q_{rwfm}', 'q_{irwfm}'};
    v0 = [q0.q_wpm, q0.q_wfm, q0.q_rwfm, 0];
    v1 = [qOpt.q_wpm, qOpt.q_wfm, qOpt.q_rwfm, 0];
    
    % Handle optional q_irwfm
    if isfield(q0, 'q_irwfm'), v0(4) = q0.q_irwfm; end
    if isfield(qOpt, 'q_irwfm'), v1(4) = qOpt.q_irwfm; end
    
    % Create grouped bar chart
    b = bar(1:4, [v0; v1]', 'grouped');
    b(1).FaceColor = [0.85 0.33 0.09];  % Orange for initial
    b(2).FaceColor = [0.00 0.45 0.74];  % Blue for optimized
    
    % Add text annotation for fixed WPM
    text(1, max(v0(1), v1(1)) * 1.5, 'FIXED', ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    
    % Formatting
    set(gca, 'YScale', 'log', 'XTick', 1:4, 'XTickLabel', names);
    ylabel('Value');
    title('Q Parameters');
    legend('Initial', 'Optimized', ...
           'Location', 'southoutside', 'Orientation', 'horizontal');
    grid on;
end

function plot_2d_cost_surface(search_history, q0, qOpt)
    % Plot 2D contour plot for q_wfm vs q_rwfm cost surface
    
    % Extract search history matrix: [q_wpm, q_wfm, q_rwfm, q_irwfm, cost]
    H = search_history;
    
    % Work in log space for proper interpolation
    log_q_wfm = log10(H(:, 2));
    log_q_rwfm = log10(H(:, 3));
    costs = H(:, 5);
    
    % Try to create contour plot in log space
    try
        % Create regular grid in log space
        log_q_wfm_range = linspace(min(log_q_wfm), max(log_q_wfm), 50);
        log_q_rwfm_range = linspace(min(log_q_rwfm), max(log_q_rwfm), 50);
        [LOG_Q_WFM_GRID, LOG_Q_RWFM_GRID] = meshgrid(log_q_wfm_range, log_q_rwfm_range);
        
        % Interpolate costs onto regular grid
        COST_GRID = griddata(log_q_wfm, log_q_rwfm, costs, LOG_Q_WFM_GRID, LOG_Q_RWFM_GRID, 'cubic');
        
        % Create filled contour plot in log space
        contourf(LOG_Q_WFM_GRID, LOG_Q_RWFM_GRID, COST_GRID, 15);
        
        % Create custom tick labels for log scale
        xlims = xlim;
        ylims = ylim;
        
        % Generate nice tick positions
        x_ticks = floor(xlims(1)):1:ceil(xlims(2));
        y_ticks = floor(ylims(1)):1:ceil(ylims(2));
        
        % Set ticks and labels
        set(gca, 'XTick', x_ticks, 'XTickLabel', arrayfun(@(x) sprintf('10^{%d}', x), x_ticks, 'UniformOutput', false));
        set(gca, 'YTick', y_ticks, 'YTickLabel', arrayfun(@(y) sprintf('10^{%d}', y), y_ticks, 'UniformOutput', false));
        
    catch
        % Fallback to scatter plot if contour fails
        scatter(log_q_wfm, log_q_rwfm, 60, costs, 'filled', 'Marker', 's');
        
        % Custom tick labels for scatter plot too
        xlims = xlim;
        ylims = ylim;
        x_ticks = floor(xlims(1)):1:ceil(xlims(2));
        y_ticks = floor(ylims(1)):1:ceil(ylims(2));
        set(gca, 'XTick', x_ticks, 'XTickLabel', arrayfun(@(x) sprintf('10^{%d}', x), x_ticks, 'UniformOutput', false));
        set(gca, 'YTick', y_ticks, 'YTickLabel', arrayfun(@(y) sprintf('10^{%d}', y), y_ticks, 'UniformOutput', false));
    end
    
    % Add colormap
    colormap(gca, 'turbo');
    c = colorbar;
    c.Label.String = 'Weighted RMS Cost';
    
    % Overlay initial and optimal points (in log space)
    hold on;
    
    % Initial point - red triangle
    plot(log10(q0.q_wfm), log10(q0.q_rwfm), '^r', ...
         'MarkerFaceColor', 'r', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', ...
         'DisplayName', 'Initial');
    
    % Optimal point - green star
    plot(log10(qOpt.q_wfm), log10(qOpt.q_rwfm), '*g', ...
         'MarkerSize', 15, 'LineWidth', 2, 'MarkerEdgeColor', 'k', ...
         'DisplayName', 'Optimal');
    
    % Formatting
    xlabel('q_{wfm}');
    ylabel('q_{rwfm}');
    title('Cost Surface (WPM fixed)');
    grid on;
    view(2);  % Force 2D view
    legend('Location', 'best');
end

function plot_1d_slice(search_history, q0, qOpt, param_type)
    % Plot 1D slice of cost function for specified parameter
    
    % Extract search history
    H = search_history;
    
    switch param_type
        case 'wfm'
            % Fix q_rwfm at optimal value, vary q_wfm
            fixed_idx = 3;
            vary_idx = 2;
            fixed_val = qOpt.q_rwfm;
            x_label = 'q_{wfm}';
            title_str = sprintf('Cost vs q_{wfm} (q_{rwfm} = %.2e)', fixed_val);
            x0 = q0.q_wfm;
            xopt = qOpt.q_wfm;
            
        case 'rwfm'
            % Fix q_wfm at optimal value, vary q_rwfm
            fixed_idx = 2;
            vary_idx = 3;
            fixed_val = qOpt.q_wfm;
            x_label = 'q_{rwfm}';
            title_str = sprintf('Cost vs q_{rwfm} (q_{wfm} = %.2e)', fixed_val);
            x0 = q0.q_rwfm;
            xopt = qOpt.q_rwfm;
    end
    
    % Extract points along this slice (with tolerance)
    tol = fixed_val * 0.1;  % 10% tolerance
    slice_mask = abs(H(:, fixed_idx) - fixed_val) < tol;
    slice_data = H(slice_mask, :);
    
    if ~isempty(slice_data)
        % Sort by varying parameter
        [~, sort_idx] = sort(slice_data(:, vary_idx));
        slice_data = slice_data(sort_idx, :);
        
        % Plot cost vs parameter
        plot(slice_data(:, vary_idx), slice_data(:, 5), 'b.-', ...
             'LineWidth', 1.5, 'MarkerSize', 8);
    else
        % No exact slice data - plot all points with transparency
        scatter(H(:, vary_idx), H(:, 5), 20, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
        title_str = [title_str ' (all points shown)'];
    end
    
    % Mark initial and optimal values
    hold on;
    xline(x0, 'r--', 'LineWidth', 1.5, 'Alpha', 0.7);
    xline(xopt, 'g--', 'LineWidth', 1.5, 'Alpha', 0.7);
    
    % Add markers at specific points
    ylims = ylim;
    plot(x0, ylims(2)*0.95, 'rv', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    plot(xopt, ylims(2)*0.95, 'g^', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
    
    % Formatting
    set(gca, 'XScale', 'log');
    xlabel(x_label);
    ylabel('Weighted RMS Cost');
    title(title_str);
    grid on;
end

function add_summary_title(res0, resOpt)
    % Add figure title with overall improvement metric
    
    if isfield(res0, 'weighted_rms') && isfield(resOpt, 'weighted_rms')
        % Calculate weighted improvement
        weighted_gain = 100 * (res0.weighted_rms - resOpt.weighted_rms) / res0.weighted_rms;
        sgtitle(sprintf('KF Q-Parameter Optimization   •   Weighted improvement = %.1f%%', ...
                        weighted_gain));
    else
        sgtitle('KF Q-Parameter Optimization Results');
    end
end

function results = package_results(config, timestamp, phase_data, tau, deviation_vals, ...
                                  ci_low, ci_high, noise_params, q_optimal, ...
                                  initial_results, final_results, opt_results, deviation_type, ...
                                  initial_covariance_info, covariance_info)
    % Package all results into structure
    
    results = struct();
    results.timestamp = timestamp;
    results.config = config;
    
    % Phase data info
    if ~isempty(phase_data)
        results.phase_data_info = struct(...
            'num_samples', length(phase_data), ...
            'duration_hours', length(phase_data) * config.tau0 / 3600, ...
            'sample_rate_hz', 1 / config.tau0, ...
            'phase_range_ns', [min(phase_data), max(phase_data)]);
    else
        results.phase_data_info = struct('num_samples', 0, 'note', 'No phase data provided');
    end
    
    % Deviation analysis
    results.deviation = struct(...
        'type', deviation_type, ...
        'tau', tau, ...
        'values', deviation_vals, ...
        'ci_low', ci_low, ...
        'ci_high', ci_high, ...
        'num_points', length(tau));
    
    % Noise fit
    results.noise_fit = struct(...
        'q0_wpm', noise_params.q_wpm, ...
        'q1_wfm', noise_params.q_wfm, ...
        'q2_rwfm', noise_params.q_rwfm);
    
    % KF results
    results.initial_kf = struct(...
        'q_params', noise_params, ...
        'results', initial_results);
    
    results.optimized_kf = struct(...
        'q_params', q_optimal, ...
        'results', final_results, ...
        'optimization_history', opt_results);
    
    % Steady-state covariance uncertainties
    results.initial_covariance_uncertainties = initial_covariance_info;
    results.covariance_uncertainties = covariance_info;
end

function save_all_results(config, timestamp, results, tau, deviation_vals, ci_low, ci_high)
    % Save results to files
    
    % Save MAT file
    mat_file = fullfile(config.results_dir, sprintf('kf_design_%s.mat', timestamp));
    save(mat_file, 'results');
    fprintf('\nResults saved:\n');
    fprintf('  MAT file: %s\n', mat_file);
    
    % Save summary text
    summary_file = fullfile(config.results_dir, sprintf('kf_summary_%s.txt', timestamp));
    write_summary_file(summary_file, results);
    fprintf('  Summary: %s\n', summary_file);
    
    % Save deviation data table
    deviation_file = fullfile(config.results_dir, sprintf('%s_data_%s.csv', results.deviation.type, timestamp));
    save_deviation_table(deviation_file, tau, deviation_vals, ci_low, ci_high);
    fprintf('  %s table: %s\n', upper(results.deviation.type), deviation_file);
end

function write_summary_file(filename, results)
    % Write human-readable summary (simplified from original)
    
    fid = fopen(filename, 'w');
    fprintf(fid, 'Kalman Filter Design Summary\n');
    fprintf(fid, '============================\n\n');
    fprintf(fid, 'Timestamp: %s\n', results.timestamp);
    fprintf(fid, 'Data: %s\n', results.config.data_name);
    fprintf(fid, 'Deviation type: %s\n\n', upper(results.deviation.type));
    
    fprintf(fid, 'Noise Parameters (fitted):\n');
    fprintf(fid, '  q0 (WPM):  %.3e\n', results.noise_fit.q0_wpm);
    fprintf(fid, '  q1 (WFM):  %.3e\n', results.noise_fit.q1_wfm);
    fprintf(fid, '  q2 (RWFM): %.3e\n\n', results.noise_fit.q2_rwfm);
    
    fprintf(fid, 'Optimization Results:\n');
    fprintf(fid, '  States: %d\n', results.config.opt.nstates);
    fprintf(fid, '  Target horizons: %d values', length(results.config.opt.target_horizons));
    if length(results.config.opt.target_horizons) <= 20
        fprintf(fid, ' - %s', mat2str(results.config.opt.target_horizons));
    else
        fprintf(fid, ' (range: %d to %d)', ...
                results.config.opt.target_horizons(1), ...
                results.config.opt.target_horizons(end));
    end
    fprintf(fid, '\n\n');
    
    % Add performance summary with subsampled horizons
    opt_res = results.optimized_kf.results;
    display_horizons = subsample_horizons_for_display(results.config.opt.target_horizons, 15);
    
    if length(results.config.opt.target_horizons) > length(display_horizons)
        fprintf(fid, 'Performance Summary (showing %d of %d horizons):\n', ...
                length(display_horizons), length(results.config.opt.target_horizons));
    else
        fprintf(fid, 'Performance Summary:\n');
    end
    
    for h = display_horizons
        idx = find(opt_res.rms_stats.horizon == h, 1);
        if ~isempty(idx)
            fprintf(fid, '  %d-step RMS: %.3f\n', h, opt_res.rms_stats.rms_error(idx));
        end
    end
    
    % Add steady-state uncertainties comparison
    fprintf(fid, '\nSteady-state Uncertainties:\n');
    fprintf(fid, '                    Initial      Optimized    Improvement\n');
    fprintf(fid, '  Phase:          %8.3f    %8.3f    %8.1f%%\n', ...
            results.initial_covariance_uncertainties.sigma_phase, ...
            results.covariance_uncertainties.sigma_phase, ...
            100 * (results.initial_covariance_uncertainties.sigma_phase - results.covariance_uncertainties.sigma_phase) / results.initial_covariance_uncertainties.sigma_phase);
    fprintf(fid, '  Frequency:      %8.3e  %8.3e  %8.1f%%\n', ...
            results.initial_covariance_uncertainties.sigma_freq, ...
            results.covariance_uncertainties.sigma_freq, ...
            100 * (results.initial_covariance_uncertainties.sigma_freq - results.covariance_uncertainties.sigma_freq) / results.initial_covariance_uncertainties.sigma_freq);
    if results.config.opt.nstates >= 3
        fprintf(fid, '  Drift:          %8.3e  %8.3e  %8.1f%%\n', ...
                results.initial_covariance_uncertainties.sigma_drift, ...
                results.covariance_uncertainties.sigma_drift, ...
                100 * (results.initial_covariance_uncertainties.sigma_drift - results.covariance_uncertainties.sigma_drift) / results.initial_covariance_uncertainties.sigma_drift);
    end
    
    fprintf(fid, '\nInitial Covariance Matrix (P_steady_initial):\n');
    for i = 1:results.config.opt.nstates
        fprintf(fid, '  [');
        for j = 1:results.config.opt.nstates
            fprintf(fid, '%12.3e', results.initial_covariance_uncertainties.P_steady(i,j));
            if j < results.config.opt.nstates
                fprintf(fid, ', ');
            end
        end
        fprintf(fid, ']\n');
    end
    
    fprintf(fid, '\nOptimized Covariance Matrix (P_steady_optimized):\n');
    for i = 1:results.config.opt.nstates
        fprintf(fid, '  [');
        for j = 1:results.config.opt.nstates
            fprintf(fid, '%12.3e', results.covariance_uncertainties.P_steady(i,j));
            if j < results.config.opt.nstates
                fprintf(fid, ', ');
            end
        end
        fprintf(fid, ']\n');
    end
    
    fclose(fid);
end

function save_deviation_table(filename, tau, values, ci_low, ci_high)
    % Save deviation data to CSV
    
    T = table(tau(:), values(:), ci_low(:), ci_high(:), ...
              'VariableNames', {'Tau_s', 'Deviation_ns', 'CI_Low_ns', 'CI_High_ns'});
    writetable(T, filename);
end

function [phase_data, tau, deviation_vals, ci, deviation_type] = load_noise_mode_data(config)
    % Load phase data for noise mode (skip deviation computation)
    
    % Load phase data
    if isempty(config.data_file)
        error('For noise mode, data_file is required to load phase data');
    end
    
    phase_data = load(config.data_file);
    if size(phase_data, 2) > 1
        phase_data = phase_data(:, config.data_column);
    end
    phase_data = phase_data(:) * config.conv_factor;
    
    N = length(phase_data);
    fprintf('  Loaded %d samples (duration: %.2f, rate: %.1f)\n', ...
            N, N * config.tau0 / 3600, 1/config.tau0);
    
    % Set dummy values for deviation data (not used in noise mode)
    tau = [];
    deviation_vals = [];
    ci = [];
    deviation_type = 'noise_provided';
end

function covariance_info = compute_covariance_uncertainties(noise_params, tau0, nstates)
    % Compute steady-state covariance matrix and uncertainties using DARE
    
    % Build system matrices
    Phi = build_phi_matrix(nstates, tau0);
    Q = build_Q_matrix(noise_params, tau0, nstates);
    H = [1, zeros(1, nstates-1)];  % Measurement matrix
    R = noise_params.q_wpm;  % Measurement noise
    
    % Solve DARE for steady-state covariance
    try
        [~, P_steady, ~] = dlqr(Phi', H', Q, R);
    catch
        % Fallback to dare if dlqr fails
        try
            [~, P_steady] = dare(Phi', H', Q, R);
        catch
            warning('DARE solution failed. Using large initial covariance approximation.');
            P_steady = 1e10 * eye(nstates);
        end
    end
    
    % Compute uncertainties as sqrt(abs(P_ij))
    P_uncertainties = sqrt(abs(P_steady));
    
    % Package results
    covariance_info = struct();
    covariance_info.P_steady = P_steady;
    covariance_info.P_uncertainties = P_uncertainties;
    covariance_info.noise_params = noise_params;
    
    % Extract individual uncertainties for easy access
    covariance_info.sigma_phase = P_uncertainties(1, 1);
    covariance_info.sigma_freq = P_uncertainties(2, 2);
    if nstates >= 3
        covariance_info.sigma_drift = P_uncertainties(3, 3);
    end
    
    % Cross-correlation terms
    covariance_info.sigma_phase_freq = P_uncertainties(1, 2);
    if nstates >= 3
        covariance_info.sigma_phase_drift = P_uncertainties(1, 3);
        covariance_info.sigma_freq_drift = P_uncertainties(2, 3);
    end
    
    % Add labels for clarity
    if nstates == 2
        covariance_info.state_labels = {'phase', 'frequency'};
    elseif nstates == 3
        covariance_info.state_labels = {'phase', 'frequency', 'drift'};
    elseif nstates == 5
        covariance_info.state_labels = {'phase', 'frequency', 'drift', 'diurnal_sin', 'diurnal_cos'};
    end
    
    % Don't display here - will be shown in comparison format later
end

function covariance_info = compute_covariance_uncertainties_silent(noise_params, tau0, nstates)
    % Compute steady-state covariance matrix and uncertainties using DARE (silent version)
    
    % Build system matrices
    Phi = build_phi_matrix(nstates, tau0);
    Q = build_Q_matrix(noise_params, tau0, nstates);
    H = [1, zeros(1, nstates-1)];  % Measurement matrix
    R = noise_params.q_wpm;  % Measurement noise
    
    % Solve DARE for steady-state covariance
    try
        [~, P_steady, ~] = dlqr(Phi', H', Q, R);
    catch
        % Fallback to dare if dlqr fails
        try
            [~, P_steady] = dare(Phi', H', Q, R);
        catch
            warning('DARE solution failed. Using large initial covariance approximation.');
            P_steady = 1e10 * eye(nstates);
        end
    end
    
    % Compute uncertainties as sqrt(abs(P_ij))
    P_uncertainties = sqrt(abs(P_steady));
    
    % Package results
    covariance_info = struct();
    covariance_info.P_steady = P_steady;
    covariance_info.P_uncertainties = P_uncertainties;
    covariance_info.noise_params = noise_params;
    
    % Extract individual uncertainties for easy access
    covariance_info.sigma_phase = P_uncertainties(1, 1);
    covariance_info.sigma_freq = P_uncertainties(2, 2);
    if nstates >= 3
        covariance_info.sigma_drift = P_uncertainties(3, 3);
    end
    
    % Cross-correlation terms
    covariance_info.sigma_phase_freq = P_uncertainties(1, 2);
    if nstates >= 3
        covariance_info.sigma_phase_drift = P_uncertainties(1, 3);
        covariance_info.sigma_freq_drift = P_uncertainties(2, 3);
    end
    
    % Add labels for clarity
    if nstates == 2
        covariance_info.state_labels = {'phase', 'frequency'};
    elseif nstates == 3
        covariance_info.state_labels = {'phase', 'frequency', 'drift'};
    elseif nstates == 5
        covariance_info.state_labels = {'phase', 'frequency', 'drift', 'diurnal_sin', 'diurnal_cos'};
    end
    
    % No console output - silent version
end

function Phi = build_phi_matrix(nstates, tau)
    % Build state transition matrix
    
    if nstates == 2
        Phi = [1, tau; 
               0, 1];
    elseif nstates == 3
        Phi = [1, tau, tau^2/2;
               0, 1,   tau;
               0, 0,   1];
    elseif nstates == 5
        % Include diurnal terms
        omega = 2*pi/(24*3600);  % Daily frequency
        Phi = [1, tau, tau^2/2, 0, 0;
               0, 1,   tau,     0, 0;
               0, 0,   1,       0, 0;
               0, 0,   0,       cos(omega*tau), sin(omega*tau);
               0, 0,   0,      -sin(omega*tau), cos(omega*tau)];
    else
        error('Unsupported number of states: %d', nstates);
    end
end

function Q = build_Q_matrix(noise_params, tau, nstates)
    % Build process noise covariance matrix
    
    q0 = noise_params.q_wpm;
    q1 = noise_params.q_wfm;
    q2 = noise_params.q_rwfm;
    
    if nstates == 2
        Q = [q0*tau + q1*tau^3/3 + q2*tau^5/20,  q1*tau^2/2 + q2*tau^4/8;
             q1*tau^2/2 + q2*tau^4/8,             q1*tau + q2*tau^3/3];
    elseif nstates == 3
        Q = [q0*tau + q1*tau^3/3 + q2*tau^5/20,  q1*tau^2/2 + q2*tau^4/8,  q2*tau^3/6;
             q1*tau^2/2 + q2*tau^4/8,             q1*tau + q2*tau^3/3,      q2*tau^2/2;
             q2*tau^3/6,                          q2*tau^2/2,               q2*tau];
    elseif nstates == 5
        % Start with 3-state Q
        Q = zeros(5, 5);
        Q(1:3, 1:3) = build_Q_matrix(noise_params, tau, 3);
        if isfield(noise_params, 'q_diurnal') && noise_params.q_diurnal > 0
            Q(4, 4) = noise_params.q_diurnal;
            Q(5, 5) = noise_params.q_diurnal;
        else
            Q(4, 4) = 0;
            Q(5, 5) = 0;
        end
    else
        error('Unsupported number of states: %d', nstates);
    end
end