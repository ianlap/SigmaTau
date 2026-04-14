function dev_summary = compute_all_devs_from_file(filename, tCol, xCol, tunits, xunits, tau0, m_list)
%COMPUTE_ALL_DEVS_FROM_FILE
% Computes all standard stability deviations and writes individual CSV files.
%
% dev_summary = compute_all_devs_from_file(filename, tCol, xCol, tunits, xunits, tau0, m_list)
%
% Inputs:
%   filename - path to data file (text or CSV)
%   tCol     - column index of time data (use 0 for implicit time vector)
%   xCol     - column index of phase data
%   tunits   - time units: 's', 'ns', 'ps', 'mjd'
%   xunits   - phase units: 's', 'ns', 'ps'
%   tau0     - sampling interval in seconds (used only if tCol == 0)
%   m_list   - (optional) averaging factors for tau = m*tau0
%
% Returns and saves a summary table with:
% tau | alpha | adev | mdev | tdev | hdev | mhdev | ldev | totdev | mtotdev | htotdev | mhtotdev

    import allanlab.*
    
    %-- Default values
    if nargin < 6 || isempty(tau0)
        tau0 = 1;
    end
    
    if nargin < 7 || isempty(m_list)
        m_list = [];  % Will be generated later based on data length
    end
    
    %-- Load file
    try
        opts = detectImportOptions(filename, 'FileType', 'text');
        data = readmatrix(filename, opts);
    catch ME
        error('Failed to read file "%s": %s', filename, ME.message);
    end
    
    %-- Validate column indices
    [nRows, nCols] = size(data);
    if tCol > nCols || xCol > nCols
        error('Column index exceeds number of columns (%d) in data file.', nCols);
    end
    
    %-- Extract time and phase
    if tCol == 0
        if xCol == 0
            x = data(:, 1);  % Default to column 1 if xCol is also 0
        else
            x = data(:, xCol);
        end
        t = (0:length(x)-1).' * tau0;
    else
        t = data(:, tCol);
        x = data(:, xCol);
    end
    
    %-- Convert units
    x = convert_units(x, xunits);
    t = convert_units(t, tunits);
    
    %-- Update tau0 if time column was provided
    if tCol ~= 0
        tau0_computed = mean(diff(t));
        if abs(tau0_computed - tau0) / tau0 > 0.1  % More than 10% difference
            warning('Computed tau0 (%.6g s) differs from specified tau0 (%.6g s). Using computed value.', ...
                    tau0_computed, tau0);
        end
        tau0 = tau0_computed;
    end
    
    %-- Generate default m_list if not provided
    if isempty(m_list)
        max_m = floor(length(x)/3);
        m_list = 2.^(0:floor(log2(max_m)));
        m_list = m_list(m_list <= max_m);
    end
    
    %-- Prepare output directory
    [folder, base, ~] = fileparts(filename);
    if isempty(folder)
        folder = '.';
    end
    outdir = fullfile(folder, [base '_devs']);
    
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end
    
    %-- Display info
    fprintf('\nComputing all deviations for: %s\n', filename);
    fprintf('Data points: %d, tau0 = %.6g s\n', length(x), tau0);
    fprintf('Output directory: %s\n', outdir);
    fprintf('----------------------------------------\n');
    
    %-- List of deviation types
    dev_types = {'adev', 'mdev', 'tdev', 'hdev', 'mhdev', ...
                 'ldev', 'totdev', 'mtotdev', 'htotdev', 'mhtotdev'};
    
    %-- Initialize summary struct
    summary_data = struct();
    summary_data.tau = [];
    summary_data.alpha = [];
    
    % Initialize all deviation fields
    for i = 1:numel(dev_types)
        summary_data.(dev_types{i}) = [];
    end
    
    %-- Warn if totdevs are requested and estimated runtime is high
    totdev_types = {'totdev','mtotdev','htotdev','mhtotdev'};
    N = length(x);
    warn_threshold = 30; % seconds
    for k = 1:numel(dev_types)
        if any(strcmpi(dev_types{k}, totdev_types))
            est_runtime = 2e-5 * N^1.4;
            if est_runtime > warn_threshold
                fprintf(['WARNING: Calculation of %s may take a long time (estimated %.1f seconds for N = %d).\n'], ...
                    upper(dev_types{k}), est_runtime, N);
            end
        end
    end
    
    %-- Compute each deviation and write individual file
    computed_count = 0;
    
    for i = 1:numel(dev_types)
        devname = dev_types{i};
        fprintf('Calculating %s ...\n', upper(devname));
        
        try
            func = str2func(['allanlab.' devname]);
            fprintf('Computing %s... ', upper(devname));
            tic;
            
            [tau, sigma, edf, ci, alpha] = func(x, tau0, m_list);
            
            % Ensure column vectors
            tau = tau(:);
            sigma = sigma(:);
            alpha = alpha(:);
            edf = edf(:);
            
            %-- Round tau to remove floating point errors
            tau_rounded = round(tau);
            tau_error = abs(tau - tau_rounded) ./ tau;
            is_integer_tau = tau_error < 1e-10;
            tau(is_integer_tau) = tau_rounded(is_integer_tau);
            
            %-- Trim results where EDF is too low (but not just NaN)
            min_edf = 2;
            valid_edf = edf >= min_edf & ~isnan(edf);
            has_low_edf = any(edf < min_edf & ~isnan(edf));
            
            if has_low_edf
                last_valid = find(valid_edf, 1, 'last');
                if isempty(last_valid)
                    warning('No valid results with EDF >= %g for %s', min_edf, upper(devname));
                    continue;
                end
                
                % Trim all arrays
                tau = tau(1:last_valid);
                sigma = sigma(1:last_valid);
                alpha = alpha(1:last_valid);
                edf = edf(1:last_valid);
                ci = ci(1:last_valid, :);
                
                fprintf('(trimmed at tau=%g) ', tau(end));
            end
            
            %-- Validate results
            if isempty(tau) || all(isnan(sigma))
                warning('No valid results for %s', upper(devname));
                continue;
            end
            
            %-- Create output table
            output_table = table(tau, alpha, sigma, ci(:,1), ci(:,2), edf, ...
                'VariableNames', {'tau', 'alpha', 'sigma', 'sigma_min', 'sigma_max', 'edf'});
            
            %-- Write to CSV file
            fname = fullfile(outdir, [base '_' devname '.csv']);
            writetable(output_table, fname, 'WriteVariableNames', true);
            
            %-- Add to summary
            if isempty(summary_data.tau)
                summary_data.tau = tau;
                summary_data.alpha = alpha;
            end
            summary_data.(devname) = sigma;
            
            comp_time = toc;
            fprintf('done (%.2f s)\n', comp_time);
            computed_count = computed_count + 1;
            
        catch ME
            warning('\nCould not compute %s: %s', upper(devname), ME.message);
        end
    end
    
    %-- Check if we have any valid data
    if isempty(summary_data.tau)
        error('No deviation calculations succeeded. Check your input data.');
    end
    
    %-- Construct summary table with only successfully computed deviations
    % Find reference tau (use the first computed deviation)
    ref_tau = summary_data.tau;
    ref_alpha = summary_data.alpha;
    
    % Check which deviations match the reference tau length
    matching_devs = {};
    mismatched_devs = {};
    
    for i = 1:numel(dev_types)
        devname = dev_types{i};
        if ~isempty(summary_data.(devname))
            if length(summary_data.(devname)) == length(ref_tau)
                matching_devs{end+1} = devname;
            else
                mismatched_devs{end+1} = devname;
                warning('%s has %d points while reference has %d points - excluding from summary', ...
                        upper(devname), length(summary_data.(devname)), length(ref_tau));
            end
        end
    end
    
    % Build summary with matching deviations only
    var_names = {'tau', 'alpha'};
    var_data = {ref_tau, ref_alpha};
    
    for i = 1:length(matching_devs)
        var_names{end+1} = matching_devs{i};
        var_data{end+1} = summary_data.(matching_devs{i});
    end
    
    % Create table
    dev_summary = table(var_data{:}, 'VariableNames', var_names);
    
    % Save summary CSV
    csvname = fullfile(outdir, [base '_summary.csv']);
    writetable(dev_summary, csvname);
    
    fprintf('----------------------------------------\n');
    fprintf('Successfully computed: %d/%d deviations\n', computed_count, length(dev_types));
    if ~isempty(mismatched_devs)
        fprintf('Excluded from summary (different tau grid): %s\n', strjoin(upper(mismatched_devs), ', '));
    end
    fprintf('All results saved to folder: %s\n', outdir);
    fprintf('Summary saved as: %s\n', csvname);
end

function y = convert_units(x, unit)
    switch lower(unit)
        case {'s', 'sec', 'seconds'}
            y = x;
        case {'ns', 'nanoseconds'}
            y = x * 1e-9;
        case {'ps', 'picoseconds'}
            y = x * 1e-12;
        case {'mjd'}
            y = (x - x(1)) * 86400; % days to seconds
        otherwise
            warning('Unknown unit "%s", passing data unchanged.', unit);
            y = x;
    end
end