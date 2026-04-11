function [kf_config, gain_config, data_config, output_config] = prompt_user_analyze_config(analysis_type)
% PROMPT_USER_ANALYZE_CONFIG - Interactive configuration for analyze functions
%
% This function prompts the user for configuration parameters needed for
% PID/PD analysis functions.
%
% Syntax:
%   [kf_config, gain_config, data_config, output_config] = prompt_user_analyze_config(analysis_type)
%
% Inputs:
%   analysis_type - 'PID' or 'PD'
%
% Outputs:
%   kf_config     - Kalman filter configuration struct
%   gain_config   - Gain configuration struct
%   data_config   - Data configuration struct
%   output_config - Output configuration struct

fprintf('=== %s ANALYSIS CONFIGURATION ===\n', analysis_type);

%% Kalman Filter Configuration
fprintf('\n--- Kalman Filter Parameters ---\n');
kf_config = struct();

kf_config.q_wfm = input('White frequency modulation noise (q_wfm) [0.0014]: ');
if isempty(kf_config.q_wfm)
    kf_config.q_wfm = 0.0014;
end

kf_config.q_rwfm = input('Random walk frequency modulation noise (q_rwfm) [6.513e-11]: ');
if isempty(kf_config.q_rwfm)
    kf_config.q_rwfm = 6.513e-11;
end

kf_config.R = input('Measurement noise variance (R) [79.3]: ');
if isempty(kf_config.R)
    kf_config.R = 79.3;
end

kf_config.maturity = input('Filter maturity samples [50000]: ');
if isempty(kf_config.maturity)
    kf_config.maturity = 50000;
end

kf_config.q_irwfm = 0;  % Default
kf_config.nstates = 3;  % Default
kf_config.start_cov = 1e6;  % Default

%% Gain Configuration
fprintf('\n--- Gain Configuration ---\n');
fprintf('Choose gain specification mode:\n');
fprintf('  1. Time constant mode (compute gains from T values)\n');
fprintf('  2. Free gain mode (specify gains directly)\n');

mode_choice = input('Select mode [1]: ');
if isempty(mode_choice)
    mode_choice = 1;
end

gain_config = struct();

if mode_choice == 1
    % Time constant mode
    gain_config.mode = 'timeconstant';
    
    fprintf('\nEnter time constant values (T):\n');
    fprintf('Example formats:\n');
    fprintf('  [1, 2, 3, 100, 200, 400]  - specific values\n');
    fprintf('  1:5                        - range 1 to 5\n');
    
    T_input = input('Time constants: ');
    if isempty(T_input)
        if strcmp(analysis_type, 'PID')
            T_input = [1, 2, 3, 100, 200, 400];
        else % PD
            T_input = 1:10;
        end
    end
    gain_config.T_values = T_input;
    
    tau0_gain = input('Sampling interval for gain calculation (tau0) [1.0]: ');
    if isempty(tau0_gain)
        tau0_gain = 1.0;
    end
    gain_config.tau0 = tau0_gain;
    
else
    % Free gain mode
    gain_config.mode = 'free';
    
    n_gain_sets = input('Number of gain sets to test: ');
    
    % Pre-allocate struct array with proper structure
    gains(n_gain_sets) = struct('g_p', [], 'g_i', [], 'g_d', [], 'label', '');
    
    for i = 1:n_gain_sets
        fprintf('\n-- Gain Set %d --\n', i);
        
        label = input(sprintf('Label for gain set %d: ', i), 's');
        if isempty(label)
            label = sprintf('Set%d', i);
        end
        
        g_p = input('Proportional gain (g_p): ');
        g_i = input('Integral gain (g_i): ');
        g_d = input('Derivative gain (g_d): ');
        
        gains(i).g_p = g_p;
        gains(i).g_i = g_i;
        gains(i).g_d = g_d;
        gains(i).label = label;
    end
    
    gain_config.gains = gains;
end

%% Data Configuration
fprintf('\n--- Data Configuration ---\n');
data_config = struct();

data_file = input('Phase error data file [6k27febunsteered.txt]: ', 's');
if isempty(data_file)
    data_file = '6k27febunsteered.txt';
end
data_config.data_file = data_file;

data_config.data_column = input('Data column [2]: ');
if isempty(data_config.data_column)
    data_config.data_column = 2;
end

data_config.cycles_to_ns = input('Cycles to nanoseconds conversion [20]: ');
if isempty(data_config.cycles_to_ns)
    data_config.cycles_to_ns = 20;
end

data_config.tau0 = input('Sampling interval (tau0) [1.0]: ');
if isempty(data_config.tau0)
    data_config.tau0 = 1.0;
end

%% Output Configuration
fprintf('\n--- Output Configuration ---\n');
output_config = struct();

output_dir = input(sprintf('Output directory [output_%s]: ', analysis_type), 's');
if isempty(output_dir)
    output_dir = sprintf('output_%s', analysis_type);
end
output_config.output_dir = output_dir;

save_results = input('Save results to files? (y/n) [y]: ', 's');
if isempty(save_results) || strcmpi(save_results, 'y')
    output_config.save_results = true;
else
    output_config.save_results = false;
end

if output_config.save_results
    save_csv = input('Save individual CSV files? (y/n) [n]: ', 's');
    if strcmpi(save_csv, 'y')
        output_config.save_csv = true;
    else
        output_config.save_csv = false;
    end
else
    output_config.save_csv = false;
end

output_config.innovation_plot_range = [];  % Default: auto-select

fprintf('\nConfiguration complete!\n');

end