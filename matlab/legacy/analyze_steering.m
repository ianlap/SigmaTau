function results = analyze_steering(kf_config, gain_config, data_config, output_config)
%% ANALYZE_STEERING - Unified Kalman filter steering analysis with multiple gain modes
%
% This function analyzes the effect of steering gains on Kalman filter
% performance for oscillator control. Supports three gain specification modes:
% - 'pd_timeconstant': Compute PD gains from time constants (g_i = 0)
% - 'pid_timeconstant': Compute PID gains from time constants using critical damping
% - 'free': Use directly specified gain values (user-specified g_p, g_i, g_d)
%
% PD gains (pd_timeconstant mode):
%   g_p = (1 - exp(-1/T))²
%   g_i = 0  (no integral term)
%   g_d = 1 - exp(-2/T)
%
% PID gains (pid_timeconstant mode):
%   a = exp(-τ/T)
%   g_p = 1 - 3a² + 2a³
%   g_i = 1 - 3a + 3a² - a³
%   g_d = 1 - a³
%
% Syntax:
%   results = analyze_steering()                                                    % Interactive mode
%   results = analyze_steering(kf_config, gain_config, data_config, output_config) % Programmatic mode
%
% Inputs (Programmatic mode):
%   kf_config    - Kalman filter configuration struct with fields:
%                  .q_wfm - White frequency modulation noise
%                  .q_rwfm - Random walk frequency modulation noise
%                  .q_irwfm - Integrated RWFM (optional, default 0)
%                  .R - Measurement noise variance
%                  .nstates - Number of states (default 3)
%                  .start_cov - Initial covariance (default 1e6)
%                  .maturity - Filter maturity samples (default 45000 for PD, 50000 for PID)
%   gain_config  - Gain configuration struct with fields:
%                  .mode - 'pd_timeconstant', 'pid_timeconstant', or 'free'
%                  For pd_timeconstant/pid_timeconstant modes:
%                  .T_values - Array of time constants
%                  .tau0 - Sampling interval for gain calculation
%                  For free mode:
%                  .gains - Struct array with fields g_p, g_i, g_d, label
%   data_config  - Data configuration struct with fields:
%                  .data_file - Phase error data file path
%                  .data_column - Column containing phase data (default 2)
%                  .cycles_to_ns - Conversion factor (default 20)
%                  .tau0 - Sampling interval [s] (default 1.0)
%   output_config - Output configuration struct with fields:
%                   .output_dir - Output directory (default 'output_steering')
%                   .save_results - Save results flag (default true)
%                   .save_csv - Save individual CSV files (default false)
%                   .innovation_plot_range - Plot range for innovations (default [])
%
% Outputs:
%   results - Results structure with analysis data and statistics
%             Each test result in .kf_results{i} now includes complete KF data:
%             .kf_data - Complete Kalman filter outputs (phase_est, freq_est, 
%                       drift_est, residuals, innovations, steers, covariances,
%                       sumsteers, sumsumsteers)
%
% Examples:
%   % Interactive mode
%   results = analyze_steering();
%
%   % PD timeconstant mode
%   kf_cfg = struct('q_wfm', 0.0014, 'q_rwfm', 6.513e-11, 'R', 79.3);
%   gain_cfg = struct('mode', 'pd_timeconstant', 'T_values', 1:10, 'tau0', 1.0);
%   data_cfg = struct('data_file', '6k27febunsteered.txt');
%   output_cfg = struct('save_results', true);
%   results = analyze_steering(kf_cfg, gain_cfg, data_cfg, output_cfg);
%
%   % PID timeconstant mode
%   gain_cfg = struct('mode', 'pid_timeconstant', 'T_values', [1,2,3,100,200,400], 'tau0', 1.0);
%   results = analyze_steering(kf_cfg, gain_cfg, data_cfg, output_cfg);
%
%   % Free gain mode
%   gains(1) = struct('g_p', 0.5, 'g_i', 0.3, 'g_d', 0.2, 'label', 'Custom1');
%   gains(2) = struct('g_p', 0.8, 'g_i', 0.0, 'g_d', 0.4, 'label', 'PD_Custom');
%   gain_cfg = struct('mode', 'free', 'gains', gains);
%   results = analyze_steering(kf_cfg, gain_cfg, data_cfg, output_cfg);
%
% See also: KALMAN_FILTER

%% Handle input arguments
if nargin == 0
    % Interactive mode - prompt user for configuration
    fprintf('=== STEERING ANALYSIS CONFIGURATION ===\n');
    
    % First ask for steering type
    fprintf('\nChoose steering controller type:\n');
    fprintf('  1. PD steering (no integral term, g_i = 0)\n');
    fprintf('  2. PID steering (with integral term)\n');
    fprintf('  3. Free gain specification (manual g_p, g_i, g_d)\n');
    
    steering_choice = input('Select steering type [2]: ');
    if isempty(steering_choice)
        steering_choice = 2;
    end
    
    if steering_choice == 1
        [kf_config, gain_config, data_config, output_config] = prompt_user_analyze_config('PD');
        gain_config.mode = 'pd_timeconstant';
    elseif steering_choice == 2
        [kf_config, gain_config, data_config, output_config] = prompt_user_analyze_config('PID');
        gain_config.mode = 'pid_timeconstant';
    elseif steering_choice == 3
        [kf_config, gain_config, data_config, output_config] = prompt_user_analyze_config('PID');
        gain_config.mode = 'free';
    else
        error('ANALYZE_STEERING:InvalidChoice', 'Invalid steering type choice');
    end
    
elseif nargin ~= 4
    error('ANALYZE_STEERING:InvalidArgs', ...
        'Must provide all four config structs or no arguments for interactive mode');
end

% Import Allan variance package
import allanlab.*

%% Validate gain mode
valid_modes = {'pd_timeconstant', 'pid_timeconstant', 'free'};
if ~isfield(gain_config, 'mode') || ~ismember(gain_config.mode, valid_modes)
    error('ANALYZE_STEERING:InvalidGainMode', ...
        'gain_config.mode must be one of: %s', strjoin(valid_modes, ', '));
end

%% Merge configurations with defaults - use appropriate controller type for merge function
if strcmp(gain_config.mode, 'pid_timeconstant')
    controller_type = 'PID';
elseif strcmp(gain_config.mode, 'pd_timeconstant')
    controller_type = 'PD';
else
    % For free mode, default to PID-style settings but allow both
    controller_type = 'PID';
end

config = merge_analyze_config(kf_config, gain_config, data_config, output_config, controller_type);
timestamp = config.timestamp;

% Override gain mode to preserve the unified mode
config.gain_mode = gain_config.mode;

%% Determine analysis characteristics based on mode
if strcmp(config.gain_mode, 'pd_timeconstant')
    analysis_type = 'PD';
    plot_title_prefix = 'PD';
    analysis_title = '=== PD STEERING ANALYSIS ===';
    summary_title = '=== SUMMARY STATISTICS (PD Control) ===';
    mode_note = 'Note: PD control has g_i=0 (no integral term)';
elseif strcmp(config.gain_mode, 'pid_timeconstant')
    analysis_type = 'PID';
    plot_title_prefix = 'PID';
    analysis_title = '=== PID STEERING ANALYSIS ===';
    summary_title = '=== SUMMARY STATISTICS (PID Control) ===';
    mode_note = 'PID control with critical damping formulas';
else % free mode
    analysis_type = 'STEERING';
    plot_title_prefix = 'Steering';
    analysis_title = '=== STEERING ANALYSIS (FREE GAINS) ===';
    summary_title = '=== SUMMARY STATISTICS (Free Gain Control) ===';
    mode_note = 'Free gain specification - supports both PD and PID configurations';
end

%% Create output directory structure
% Extract base filename from data file path
[~, data_filename, ~] = fileparts(config.data.data_file);

% Create main steering results directory if it doesn't exist
main_results_dir = 'steering_results';
if ~exist(main_results_dir, 'dir')
    mkdir(main_results_dir);
end

% Create specific analysis directory with filename, type, and timestamp
analysis_dir_name = sprintf('%s_%s_%s', data_filename, upper(analysis_type), timestamp);
analysis_output_dir = fullfile(main_results_dir, analysis_dir_name);

% Create the analysis-specific directory
if ~exist(analysis_output_dir, 'dir')
    mkdir(analysis_output_dir);
end

% Display where results will be saved
fprintf('\nOutput directory: steering_results/%s\n', analysis_dir_name);

% Create subdirectories for better organization
subdirs = {'plots', 'csv_data', 'summaries', 'mat_files', 'adev_analysis'};
for i = 1:length(subdirs)
    subdir_path = fullfile(analysis_output_dir, subdirs{i});
    if ~exist(subdir_path, 'dir')
        mkdir(subdir_path);
    end
end

% Store subdirectory paths for easy access
output_paths = struct();
output_paths.base = analysis_output_dir;
output_paths.plots = fullfile(analysis_output_dir, 'plots');
output_paths.csv = fullfile(analysis_output_dir, 'csv_data');
output_paths.summaries = fullfile(analysis_output_dir, 'summaries');
output_paths.mat = fullfile(analysis_output_dir, 'mat_files');
output_paths.adev = fullfile(analysis_output_dir, 'adev_analysis');

% README file will be created after data loading when N is available

%% Load and prepare data
fprintf('%s\n', analysis_title);
fprintf('Loading data from: %s\n', config.data.data_file);

try
    data = readmatrix(config.data.data_file);
catch ME
    error('ANALYZE_STEERING:FileLoadError', ...
        'Failed to load data file "%s": %s', config.data.data_file, ME.message);
end
phase_counts = data(:, config.data.data_column);
N = length(phase_counts);

% Convert to nanoseconds
phase_ns = phase_counts * config.data.cycles_to_ns;

fprintf('Loaded %d samples (%.2f hours @ %.1f Hz)\n', ...
        N, N * config.data.tau0 / 3600, 1/config.data.tau0);
fprintf('Phase range: [%.2f, %.2f] ns\n\n', min(phase_ns), max(phase_ns));

% Create README file now that we have the data information
readme_file = fullfile(output_paths.base, 'README.txt');
fid = fopen(readme_file, 'w');
fprintf(fid, 'Steering Analysis Output Directory\n');
fprintf(fid, '==================================\n\n');
fprintf(fid, 'Analysis: %s\n', analysis_dir_name);
fprintf(fid, 'Generated: %s\n', datestr(now));
fprintf(fid, 'Data File: %s\n', config.data.data_file);
fprintf(fid, 'Analysis Type: %s\n', analysis_type);
fprintf(fid, 'Gain Mode: %s\n\n', config.gain_mode);
fprintf(fid, 'Directory Contents:\n');
fprintf(fid, '- plots/         : Analysis plots (PNG and FIG formats)\n');
fprintf(fid, '- csv_data/      : Individual Kalman filter time series for each test\n');
fprintf(fid, '- summaries/     : Human-readable summary text files\n');
fprintf(fid, '- mat_files/     : Complete MATLAB results structures\n');
fprintf(fid, '- adev_analysis/ : Allan deviation analysis results\n\n');
fprintf(fid, 'Main Results:\n');
fprintf(fid, '- mat_files/*_analysis_*.mat : Complete results structure\n');
fprintf(fid, '- summaries/*_summary_*.txt  : Summary statistics and configuration\n');
fprintf(fid, '- plots/*_gains_analysis.*   : Comprehensive analysis plots\n\n');
fprintf(fid, 'Data Info:\n');
fprintf(fid, '- Samples: %d\n', N);
fprintf(fid, '- Duration: %.2f hours\n', N * config.data.tau0 / 3600);
fprintf(fid, '- Sample Rate: %.1f Hz\n', 1/config.data.tau0);
fclose(fid);

%% Initialize storage
results_all = cell(config.n_tests, 1);
adev_results = cell(config.n_tests, 1);
summary_stats = zeros(config.n_tests, 5); % T/index, phase_rms, freq_rms, steer_rms, min_adev

%% Run Kalman filter for each test configuration
fprintf('Running Kalman filters...\n');
fprintf('%-8s | %-8s %-8s %-8s | %s\n', 'Test', 'g_p', 'g_i', 'g_d', 'Status');
fprintf('---------|---------------------------|--------\n');

for i = 1:config.n_tests
    if strcmp(config.gain_mode, 'pd_timeconstant')
        T = config.test_params(i).T;
        
        % Compute PD gains from time constant
        g_p = (1 - exp(-1/T))^2;
        g_i = 0.0;                   % No integral term for PD control
        g_d = 1 - exp(-2/T);
        test_label = sprintf('T=%d', T);
        
        fprintf('%-8s | %8.6f %8.6f %8.6f | ', test_label, g_p, g_i, g_d);
        
    elseif strcmp(config.gain_mode, 'pid_timeconstant')
        T = config.test_params(i).T;
        
        % Compute PID gains from time constant using critical damping
        a = exp(-config.data.tau0 / T);
        g_p = 1 - 3*a^2 + 2*a^3;
        g_i = 1 - 3*a + 3*a^2 - a^3;
        g_d = 1 - a^3;
        test_label = sprintf('T=%d', T);
        
        fprintf('%-8s | %8.6f %8.6f %8.6f | ', test_label, g_p, g_i, g_d);
        
    else % free mode
        g_p = config.test_params(i).g_p;
        g_i = config.test_params(i).g_i;
        g_d = config.test_params(i).g_d;
        test_label = config.test_params(i).label;
        
        % Identify if this is essentially PD control (g_i ≈ 0)
        if abs(g_i) < 1e-10
            gain_type_note = '(PD-like)';
        elseif g_i > 0
            gain_type_note = '(PID)';
        else
            gain_type_note = '(neg-I)';
        end
        
        fprintf('%-8s | %8.6f %8.6f %8.6f | %s ', test_label, g_p, g_i, g_d, gain_type_note);
    end
    
    % Initialize state estimate
    mean_increment = mean(diff(phase_ns(1:min(100, N))));
    init_state = [phase_ns(1); mean_increment; 0.0];
    
    % Run Kalman filter
    tic;
    [phase_est, freq_est, drift_est, residuals, innovations, steers, ...
     rtP00, rtP11, rtP22, rtP01, rtP02, rtP12, sumsteers, sumsumsteers] = ...
        kalman_filter(phase_ns, ...
                     config.kf.q_wfm, ...
                     config.kf.q_rwfm, ...
                     config.kf.R, ...
                     g_p, g_i, g_d, ...
                     config.kf.nstates, ...
                     config.data.tau0, ...
                     config.kf.start_cov, ...
                     init_state, ...
                     config.kf.q_irwfm);
    elapsed = toc;
    
    fprintf('Done (%.2fs)\n', elapsed);
    
    % Store results with complete KF data
    results_all{i} = struct(...
        'test_label', test_label, ...
        'g_p', g_p, ...
        'g_i', g_i, ...
        'g_d', g_d ...
    );
    
    % Store complete Kalman filter data
    results_all{i}.kf_data = struct();
    results_all{i}.kf_data.phase_est = phase_est;
    results_all{i}.kf_data.freq_est = freq_est;
    results_all{i}.kf_data.drift_est = drift_est;
    results_all{i}.kf_data.residuals = residuals;
    results_all{i}.kf_data.innovations = innovations;
    results_all{i}.kf_data.steers = steers;
    results_all{i}.kf_data.sumsteers = sumsteers;
    results_all{i}.kf_data.sumsumsteers = sumsumsteers;
    
    % Include covariance matrices
    results_all{i}.kf_data.covariances = struct();
    results_all{i}.kf_data.covariances.P00 = rtP00;
    results_all{i}.kf_data.covariances.P11 = rtP11;
    results_all{i}.kf_data.covariances.P22 = rtP22;
    results_all{i}.kf_data.covariances.P01 = rtP01;
    results_all{i}.kf_data.covariances.P02 = rtP02;
    results_all{i}.kf_data.covariances.P12 = rtP12;
    
    % Store backward-compatible fields for existing code
    if config.output.save_csv || i <= 6  % Keep first 6 for plotting
        results_all{i}.phase = phase_est;
        results_all{i}.freq = freq_est;
        results_all{i}.drift = drift_est;
        results_all{i}.innovation = innovations;
        results_all{i}.steer = steers;
        results_all{i}.rtP00 = rtP00;
    else
        % Store only what's needed for ADEV calculation and final period for plotting
        final_period_start = max(1, N - 3000);  % Last 3000 points for innovation plot
        results_all{i}.phase = phase_est(config.kf.maturity:end);  % For ADEV
        results_all{i}.innovation = innovations(final_period_start:end);  % For plotting
        % Don't store freq, drift, steer, rtP00 to save memory (but full data is in kf_data)
    end
    
    % Add T value for timeconstant modes (for backward compatibility)
    if ismember(config.gain_mode, {'pd_timeconstant', 'pid_timeconstant'})
        results_all{i}.T = T;
    end
    
    % Calculate post-maturity statistics
    post_phase = results_all{i}.kf_data.phase_est(config.kf.maturity:end);
    post_freq = results_all{i}.kf_data.freq_est(config.kf.maturity:end);
    post_steer = results_all{i}.kf_data.steers(config.kf.maturity:end);
    
    if ismember(config.gain_mode, {'pd_timeconstant', 'pid_timeconstant'})
        summary_stats(i, 1) = T;
    else
        summary_stats(i, 1) = i;  % Use index for free mode
    end
    summary_stats(i, 2) = rms(post_phase);      % Phase RMS
    summary_stats(i, 3) = rms(post_freq);       % Frequency RMS
    summary_stats(i, 4) = rms(post_steer);      % Steer RMS
    
    % Save individual CSV if requested
    if config.output.save_csv
        if ismember(config.gain_mode, {'pd_timeconstant', 'pid_timeconstant'})
            csv_file = fullfile(output_paths.csv, sprintf('kalman_%s_T%d.csv', upper(analysis_type), T));
        else
            csv_file = fullfile(output_paths.csv, sprintf('kalman_%s_%s.csv', upper(analysis_type), test_label));
        end
        save_kalman_results(results_all{i}, csv_file, analysis_type);
    end
end

%% Compute Allan deviations
fprintf('\nComputing Allan deviations...\n');

for i = 1:config.n_tests
    test_label = results_all{i}.test_label;
    fprintf('  %s...', test_label);
    
    % Use phase data after maturity, convert to seconds
    phase_after_maturity = results_all{i}.kf_data.phase_est(config.kf.maturity:end);
    phase_data_s = phase_after_maturity * 1e-9;
    
    if length(phase_data_s) > 100
        try
            [tau_vals, adev_vals, ~, ~, ~] = allanlab.adev(phase_data_s, config.data.tau0, []);
            adev_results{i} = struct('test_label', test_label, 'tau', tau_vals, 'adev', adev_vals);
            
            [min_adev, ~] = min(adev_vals);
            summary_stats(i, 5) = min_adev;
            fprintf(' Min ADEV: %.2e\n', min_adev);
        catch ME
            fprintf(' Failed: %s\n', ME.message);
            adev_results{i} = [];
            summary_stats(i, 5) = NaN;
        end
    else
        fprintf(' Insufficient data\n');
        adev_results{i} = [];
        summary_stats(i, 5) = NaN;
    end
end

%% Create comprehensive plots
fprintf('\nCreating analysis plots...\n');

% Create plots using utility function
fig = create_steering_plots(results_all, adev_results, config, N, analysis_type);

% Save plots if requested
if config.output.save_results
    saveas(fig, fullfile(output_paths.plots, sprintf('%s_gains_analysis.png', lower(analysis_type))));
    saveas(fig, fullfile(output_paths.plots, sprintf('%s_gains_analysis.fig', lower(analysis_type))));
end

%% Display and save summary statistics
% Add output directory to config for display_steering_summary
config.output.output_dir = output_paths.summaries;
display_steering_summary(results_all, adev_results, summary_stats, config, analysis_type, mode_note);

%% Display mode-specific comparison notes
if strcmp(config.gain_mode, 'pd_timeconstant')
    fprintf('\n=== PD vs PID Comparison ===\n');
    fprintf('PD control (g_i=0) may result in steady-state errors but can improve stability.\n');
    fprintf('Compare with PID results to evaluate trade-offs.\n');
elseif strcmp(config.gain_mode, 'free')
    fprintf('\n=== Free Gain Analysis Notes ===\n');
    fprintf('Free gain mode allows testing of arbitrary PID/PD configurations.\n');
    fprintf('PD-like configurations have g_i ≈ 0, PID configurations have g_i > 0.\n');
    fprintf('Negative integral gains (g_i < 0) may indicate unusual control strategies.\n');
end

%% Step 9: Save results
results = struct();
results.timestamp = timestamp;
results.config = config;
results.analysis_type = analysis_type;
results.gain_mode = config.gain_mode;
results.phase_data_info = struct('N', N, 'tau0', config.data.tau0, ...
                                'duration_hours', N * config.data.tau0 / 3600, ...
                                'range', [min(phase_ns), max(phase_ns)]);
results.gain_analysis = struct('mode', config.gain_mode, 'n_tests', config.n_tests, ...
                              'test_params', config.test_params);
results.kf_results = results_all;
results.adev_analysis = adev_results;
results.summary_statistics = summary_stats;

if config.output.save_results
    fprintf('\nStep 9: Saving results...\n');
    
    % Save to MAT file
    save_file = fullfile(output_paths.mat, sprintf('%s_analysis_%s.mat', lower(analysis_type), timestamp));
    save(save_file, 'results');
    fprintf('  Results saved to: %s\n', save_file);
    
    % Save summary to text file
    summary_file = fullfile(output_paths.summaries, sprintf('%s_summary_%s.txt', lower(analysis_type), timestamp));
    write_analysis_summary(summary_file, results);
    fprintf('  Summary saved to: %s\n', summary_file);
    
    % Save Allan deviation data to CSV
    adev_file = fullfile(output_paths.adev, sprintf('%s_adev_data_%s.csv', lower(analysis_type), timestamp));
    save_adev_table(adev_file, adev_results);
    fprintf('  ADEV data saved to: %s\n', adev_file);
    
    if config.output.save_csv
        fprintf('  Individual KF results saved to: %s\n', output_paths.csv);
    end
    
    fprintf('  Plots saved to: %s\n', output_paths.plots);
    fprintf('\n  All results organized in: %s\n', output_paths.base);
    fprintf('  (steering_results/%s)\n', analysis_dir_name);
end

fprintf('\n=== %s ANALYSIS COMPLETE ===\n', upper(analysis_type));

end

%% Helper functions

function write_analysis_summary(filename, results)
% Write human-readable summary of steering analysis results

fid = fopen(filename, 'w');
fprintf(fid, '%s Steering Analysis Summary\n', results.analysis_type);
fprintf(fid, '%s\n', repmat('=', 1, length(results.analysis_type) + 25));
fprintf(fid, 'Generated: %s\n\n', results.timestamp);

fprintf(fid, 'Data Information:\n');
fprintf(fid, '  Samples: %d\n', results.phase_data_info.N);
fprintf(fid, '  Duration: %.2f hours\n', results.phase_data_info.duration_hours);
fprintf(fid, '  Sample rate: %.1f Hz\n\n', 1/results.phase_data_info.tau0);

fprintf(fid, 'Analysis Configuration:\n');
fprintf(fid, '  Gain mode: %s\n', results.gain_analysis.mode);
fprintf(fid, '  Number of tests: %d\n', results.gain_analysis.n_tests);

% Add mode-specific notes
if strcmp(results.gain_mode, 'pd_timeconstant')
    fprintf(fid, '  Note: PD control typically uses g_i=0 (no integral term)\n');
elseif strcmp(results.gain_mode, 'pid_timeconstant')
    fprintf(fid, '  Note: PID gains computed using critical damping formulas\n');
else
    fprintf(fid, '  Note: Free gain specification allows arbitrary PID/PD configurations\n');
end
fprintf(fid, '\n');

fprintf(fid, 'Kalman Filter Parameters:\n');
fprintf(fid, '  q_wfm:  %.3e\n', results.config.kf.q_wfm);
fprintf(fid, '  q_rwfm: %.3e\n', results.config.kf.q_rwfm);
fprintf(fid, '  R:      %.3e\n', results.config.kf.R);
fprintf(fid, '  Maturity: %d samples\n\n', results.config.kf.maturity);

% Test configurations table
fprintf(fid, 'Test Configurations:\n');
if ismember(results.gain_analysis.mode, {'pd_timeconstant', 'pid_timeconstant'})
    fprintf(fid, '%-8s | %-8s | %-8s | %-8s | %-8s\n', 'Label', 'T', 'g_p', 'g_i', 'g_d');
else
    fprintf(fid, '%-8s | %-8s | %-8s | %-8s\n', 'Label', 'g_p', 'g_i', 'g_d');
end
fprintf(fid, '---------|----------|----------|----------|----------\n');

for i = 1:results.gain_analysis.n_tests
    kf_result = results.kf_results{i};
    if ismember(results.gain_analysis.mode, {'pd_timeconstant', 'pid_timeconstant'})
        fprintf(fid, '%-8s | %8d | %8.6f | %8.6f | %8.6f\n', ...
                kf_result.test_label, kf_result.T, kf_result.g_p, kf_result.g_i, kf_result.g_d);
    else
        fprintf(fid, '%-8s | %8.6f | %8.6f | %8.6f\n', ...
                kf_result.test_label, kf_result.g_p, kf_result.g_i, kf_result.g_d);
    end
    
    % Flag special configurations
    if strcmp(results.gain_mode, 'free')
        if abs(kf_result.g_i) < 1e-10
            fprintf(fid, '         Note: PD-like control (g_i ≈ 0)\n');
        elseif kf_result.g_i < 0
            fprintf(fid, '         Note: Negative integral gain\n');
        end
    elseif strcmp(results.gain_mode, 'pd_timeconstant') && kf_result.g_i ~= 0
        fprintf(fid, '         WARNING: Non-zero integral gain for PD control\n');
    end
end

% Performance summary
fprintf(fid, '\nPerformance Summary (first 5 tests):\n');
fprintf(fid, '%-8s | %-12s | %-12s | %-12s | %-12s\n', 'Label', 'Phase RMS', 'Freq RMS', 'Steer RMS', 'Min ADEV');
fprintf(fid, '---------|--------------|--------------|--------------|-------------\n');

n_show = min(5, results.gain_analysis.n_tests);
for i = 1:n_show
    stats = results.summary_statistics(i, :);
    fprintf(fid, '%-8s | %12.3f | %12.6f | %12.3f | %12.2e\n', ...
            results.kf_results{i}.test_label, stats(2), stats(3), stats(4), stats(5));
end

if results.gain_analysis.n_tests > 5
    fprintf(fid, '... (see full CSV for all %d tests)\n', results.gain_analysis.n_tests);
end

% Mode-specific notes
if strcmp(results.gain_mode, 'pd_timeconstant')
    fprintf(fid, '\nPD vs PID Comparison:\n');
    fprintf(fid, 'PD control (g_i=0) may result in steady-state errors but can improve stability.\n');
    fprintf(fid, 'Compare with PID results to evaluate trade-offs.\n\n');
elseif strcmp(results.gain_mode, 'free')
    fprintf(fid, '\nFree Gain Analysis:\n');
    fprintf(fid, 'This mode allows testing arbitrary gain combinations.\n');
    fprintf(fid, 'PD-like: g_i ≈ 0, PID: g_i > 0, Special cases: g_i < 0\n\n');
end

fprintf(fid, 'Complete results saved to MAT file.\n');
fclose(fid);

end