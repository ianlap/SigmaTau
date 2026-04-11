function config = merge_analyze_config(kf_config, gain_config, data_config, output_config, analysis_type)
% MERGE_ANALYZE_CONFIG - Merge and validate configuration for analyze functions
%
% This utility function merges user-provided configuration structs with
% defaults and validates the parameters for PID/PD analysis functions.
%
% Syntax:
%   config = merge_analyze_config(kf_config, gain_config, data_config, output_config, analysis_type)
%
% Inputs:
%   kf_config     - Kalman filter configuration struct
%   gain_config   - Gain configuration struct
%   data_config   - Data configuration struct  
%   output_config - Output configuration struct
%   analysis_type - 'PID' or 'PD'
%
% Outputs:
%   config - Merged and validated configuration struct

%% Set defaults for Kalman filter config
kf_defaults = struct();
kf_defaults.q_wfm = 0.0014;
kf_defaults.q_rwfm = 6.513e-11;
kf_defaults.q_irwfm = 0;
kf_defaults.R = 79.3;
kf_defaults.nstates = 3;
kf_defaults.start_cov = 1e6;

% Different maturity defaults for PID vs PD
if strcmp(analysis_type, 'PD')
    kf_defaults.maturity = 45000;
else
    kf_defaults.maturity = 50000;
end

%% Set defaults for data config
data_defaults = struct();
data_defaults.data_file = '6k27febunsteered.txt';
data_defaults.data_column = 2;
data_defaults.cycles_to_ns = 20;
data_defaults.tau0 = 1.0;

%% Set defaults for output config
output_defaults = struct();
output_defaults.output_dir = sprintf('output_%s', analysis_type);
output_defaults.save_results = true;
output_defaults.save_csv = false;
output_defaults.innovation_plot_range = [];

%% Merge with defaults
config = struct();
config.kf = merge_struct_with_defaults(kf_config, kf_defaults);
config.data = merge_struct_with_defaults(data_config, data_defaults);
config.output = merge_struct_with_defaults(output_config, output_defaults);

%% Process gain configuration
config.gain_mode = gain_config.mode;

if ismember(gain_config.mode, {'timeconstant', 'pd_timeconstant', 'pid_timeconstant'})
    % Time constant mode (includes PD and PID variants)
    config.test_params = struct();
    for i = 1:length(gain_config.T_values)
        config.test_params(i).T = gain_config.T_values(i);
    end
    config.n_tests = length(gain_config.T_values);
    
    % Store tau0 for gain calculation (may differ from data tau0)
    if isfield(gain_config, 'tau0')
        config.gain_tau0 = gain_config.tau0;
    else
        config.gain_tau0 = config.data.tau0;  % Use data tau0 as default
    end
    
elseif strcmp(gain_config.mode, 'free')
    % Free gain mode
    config.test_params = gain_config.gains;
    config.n_tests = length(gain_config.gains);
    
    % Validate that each gain struct has required fields
    required_fields = {'g_p', 'g_i', 'g_d', 'label'};
    for i = 1:config.n_tests
        for j = 1:length(required_fields)
            if ~isfield(config.test_params(i), required_fields{j})
                error('MERGE_ANALYZE_CONFIG:MissingField', ...
                    'Gain struct %d missing required field: %s', i, required_fields{j});
            end
        end
    end
else
    error('MERGE_ANALYZE_CONFIG:InvalidMode', ...
        'gain_config.mode must be one of: ''timeconstant'', ''pd_timeconstant'', ''pid_timeconstant'', or ''free''');
end

%% Validate file paths
if ~exist(config.data.data_file, 'file')
    error('MERGE_ANALYZE_CONFIG:FileNotFound', ...
        'Data file not found: %s', config.data.data_file);
end

%% Create output directory
if ~exist(config.output.output_dir, 'dir')
    mkdir(config.output.output_dir);
    fprintf('Created output directory: %s\n', config.output.output_dir);
end

%% Add timestamp
config.timestamp = datestr(now, 'yyyymmdd_HHMMSS');

end

%% Helper function to merge structs with defaults
function merged = merge_struct_with_defaults(user_struct, defaults)
% Merge user-provided struct with defaults

merged = defaults;
if ~isempty(user_struct) && isstruct(user_struct)
    fields = fieldnames(user_struct);
    for i = 1:length(fields)
        merged.(fields{i}) = user_struct.(fields{i});
    end
end

end