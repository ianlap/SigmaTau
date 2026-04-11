function [data_config, opt_config, output_config] = prompt_user_config(pipeline_mode)
% PROMPT_USER_CONFIG - Interactive configuration prompting for unified pipeline
%
% Prompts user for all required configuration parameters based on pipeline mode
%
% Input:
%   pipeline_mode - String specifying pipeline mode
%
% Outputs:
%   data_config   - Data configuration struct
%   opt_config    - Optimization configuration struct  
%   output_config - Output configuration struct

    fprintf('\n=== Interactive Configuration ===\n');
    fprintf('Pipeline mode: %s\n\n', pipeline_mode);
    
    %% Data Configuration
    fprintf('--- Data Configuration ---\n');
    
    data_config = struct();
    
    % Get data file and basic parameters
    if strcmp(pipeline_mode, 'noise')
        data_config.data_file = input('Phase data file path: ', 's');
        data_config.data_column = input('Data column number [2]: ');
        if isempty(data_config.data_column), data_config.data_column = 2; end
        
        data_config.tau0 = input('Sampling interval (tau0): ');
        data_config.data_name = input('Dataset name: ', 's');
        data_config.conv_factor = input('Unit conversion factor [1]: ');
        if isempty(data_config.conv_factor), data_config.conv_factor = 1; end
        
        % Get noise parameters
        fprintf('\n--- Noise Parameters ---\n');
        data_config.q_wpm = input('q_wpm (white phase modulation): ');
        data_config.q_wfm = input('q_wfm (white frequency modulation): ');  
        data_config.q_rwfm = input('q_rwfm (random walk frequency modulation): ');
        
        q_irwfm = input('q_irwfm (integrated RWFM) [0]: ');
        if isempty(q_irwfm), q_irwfm = 0; end
        if q_irwfm > 0
            data_config.q_irwfm = q_irwfm;
        end
        
    elseif contains(pipeline_mode, 'precomputed')
        % Precomputed data mode
        if contains(pipeline_mode, 'mhtotdev')
            deviation_file = input('MHTOTDEV data file (combined format): ', 's');
            if ~isempty(deviation_file)
                data_config.mhtotdev_file = deviation_file;
            else
                data_config.tau_file = input('Tau values file: ', 's');
                data_config.mhtotdev_values_file = input('MHTOTDEV values file: ', 's');
                data_config.ci_file = input('Confidence intervals file [optional]: ', 's');
            end
        else
            deviation_file = input('MHDEV data file (combined format): ', 's');
            if ~isempty(deviation_file)
                data_config.mhdev_file = deviation_file;
            else
                data_config.tau_file = input('Tau values file: ', 's');
                data_config.mhdev_values_file = input('MHDEV values file: ', 's');
                data_config.ci_file = input('Confidence intervals file [optional]: ', 's');
            end
        end
        
        data_config.phase_file = input('Phase data file: ', 's');
        data_config.tau0 = input('Sampling interval (tau0): ');
        data_config.data_name = input('Dataset name: ', 's');
        data_config.conv_factor = input('Unit conversion factor [1]: ');
        if isempty(data_config.conv_factor), data_config.conv_factor = 1; end
        
    else
        % Raw data mode
        data_config.data_file = input('Phase data file path: ', 's');
        data_config.data_column = input('Data column number [2]: ');
        if isempty(data_config.data_column), data_config.data_column = 2; end
        
        data_config.tau0 = input('Sampling interval (tau0): ');
        data_config.data_name = input('Dataset name: ', 's');
        data_config.conv_factor = input('Unit conversion factor [1]: ');
        if isempty(data_config.conv_factor), data_config.conv_factor = 1; end
        
        data_config.fit_max_iterations = input('Max fitting iterations [6]: ');
        if isempty(data_config.fit_max_iterations), data_config.fit_max_iterations = 6; end
    end
    
    %% Optimization Configuration
    fprintf('\n--- Optimization Configuration ---\n');
    
    opt_config = struct();
    
    % First ask for optimization method
    method = input('Optimization method (grid/fmincon) [grid]: ', 's');
    if isempty(method), method = 'grid'; end
    opt_config.method = method;
    
    % Ask for method-specific parameters
    if strcmpi(method, 'grid')
        % Grid-specific parameters
        if ~strcmp(pipeline_mode, 'noise')
            opt_config.search_range = input('Search range (decades) [2]: ');
            if isempty(opt_config.search_range), opt_config.search_range = 2; end
        else
            % For noise mode, still need search range for grid optimization
            opt_config.search_range = input('Search range (decades) [2]: ');
            if isempty(opt_config.search_range), opt_config.search_range = 2; end
        end
        
        opt_config.n_grid_per_decade = input('Grid points per decade [5]: ');
        if isempty(opt_config.n_grid_per_decade), opt_config.n_grid_per_decade = 5; end
        
    elseif strcmpi(method, 'fmincon')
        % fmincon-specific parameters
        if ~strcmp(pipeline_mode, 'noise')
            opt_config.search_range = input('Search range (decades) [2]: ');
            if isempty(opt_config.search_range), opt_config.search_range = 2; end
        else
            % For noise mode, still need search range for fmincon bounds
            opt_config.search_range = input('Search range (decades) [2]: ');
            if isempty(opt_config.search_range), opt_config.search_range = 2; end
        end
        
        % fmincon doesn't need grid points per decade
        opt_config.n_grid_per_decade = 5;  % Default (not used)
    end
    
    opt_config.nstates = input('Number of KF states (2, 3, or 5) [3]: ');
    if isempty(opt_config.nstates), opt_config.nstates = 3; end
    
    fprintf('Target prediction horizons (space-separated): ');
    horizons_str = input('', 's');
    if isempty(horizons_str)
        opt_config.target_horizons = [10, 100, 1000];
    else
        opt_config.target_horizons = str2num(horizons_str);
    end
    
    weights_str = input('Horizon weights (space-separated, or enter for equal weights): ', 's');
    if isempty(weights_str)
        opt_config.horizon_weights = ones(1, length(opt_config.target_horizons));
    else
        opt_config.horizon_weights = str2num(weights_str);
    end
    
    opt_config.maturity = input('Filter maturity time [50000]: ');
    if isempty(opt_config.maturity), opt_config.maturity = 50000; end
    
    %% Output Configuration
    fprintf('\n--- Output Configuration ---\n');
    
    output_config = struct();
    
    save_results = input('Save results? (y/n) [y]: ', 's');
    if isempty(save_results) || strcmpi(save_results, 'y')
        output_config.save_results = true;
    else
        output_config.save_results = false;
    end
    
    save_plots = input('Save plots? (y/n) [y]: ', 's');
    if isempty(save_plots) || strcmpi(save_plots, 'y')
        output_config.save_plots = true;
    else
        output_config.save_plots = false;
    end
    
    results_dir = input('Results directory (or enter for auto-generate): ', 's');
    output_config.results_dir = results_dir;
    
    verbose = input('Verbose output? (y/n) [y]: ', 's');
    if isempty(verbose) || strcmpi(verbose, 'y')
        output_config.verbose = true;
    else
        output_config.verbose = false;
    end
    
    fprintf('\n=== Configuration Complete ===\n');
end