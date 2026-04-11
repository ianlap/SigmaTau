function results = compute_devs_from_file(filename, tCol, xCol, tunits, xunits, tau0, dev_list, m_list)
%COMPUTE_DEVS_FROM_FILE Compute selected stability deviations from a data file.
%
% results = compute_devs_from_file(filename, tCol, xCol, tunits, xunits, tau0, dev_list, m_list)
%
% Inputs:
%   filename - path to data file (text or CSV)
%   tCol     - column index of time data (use 0 for implicit time vector)
%   xCol     - column index of phase data
%   tunits   - time units: 's', 'ns', 'ps', 'mjd'
%   xunits   - phase units: 's', 'ns', 'ps'
%   tau0     - sampling interval in seconds (used only if tCol == 0)
%   dev_list - cell array of deviation types to compute, e.g., {'adev', 'mdev'}
%   m_list   - (optional) averaging factors for tau = m*tau0
%
% Outputs:
%   results  - struct containing computed deviations and metadata
%
% Files created:
%   For each deviation, writes a CSV file: basename_devtype.csv in the folder:
%   ./basename_devs/
%   Each file has columns:
%   tau | alpha | sigma | sigma_min | sigma_max | edf
%
% Example:
%   results = compute_devs_from_file('data.txt', 1, 2, 's', 's', 1, {'adev', 'mdev'});
%   results = compute_devs_from_file('data.txt', 1, 2, 's', 's', 1, {'adev'}, 2.^(0:10));

    import allanlab.*
    
    %-- Default values
    if nargin < 6 || isempty(tau0)
        tau0 = 1;
    end
    
    if nargin < 7 || isempty(dev_list)
        dev_list = {'adev', 'mdev'};  % Default to ADEV and MDEV
    end
    
    if nargin < 8 || isempty(m_list)
        % Default m_list will be generated later based on data length
        m_list = [];
    end
    
    %-- Ensure dev_list is a cell array
    if ischar(dev_list)
        dev_list = {dev_list};
    end
    
    %-- Read data using detected import options
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
    
    if tCol == xCol && tCol ~= 0
        error('Time and phase columns cannot be the same (both are column %d).', tCol);
    end
    
    %-- Extract time and phase columns
    if tCol == 0
        % Implicit time vector
        if xCol == 0
            x = data(:, 1);  % Default to first column
        else
            x = data(:, xCol);
        end
        t = (0:length(x)-1).' * tau0;
    else
        t = data(:, tCol);
        x = data(:, xCol);
    end
    
    %-- Convert to seconds
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
    
    %-- Prepare output directory
    [folder, base, ~] = fileparts(filename);
    if isempty(folder)
        folder = '.';
    end
    outdir = fullfile(folder, [base '_devs']);
    
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end
    
    %-- Generate default m_list if not provided
    if isempty(m_list)
        max_m = floor(length(x)/3);
        m_list = 2.^(0:floor(log2(max_m)));
        m_list = m_list(m_list <= max_m);
    end
    
    %-- Initialize results structure
    results = struct();
    results.filename = filename;
    results.data_points = length(x);
    results.tau0 = tau0;
    results.m_list = m_list;
    results.outdir = outdir;
    results.computed = {};
    results.failed = {};
    
    %-- Display computation info
    fprintf('\nComputing deviations for: %s\n', filename);
    fprintf('Data points: %d, tau0 = %.6g s\n', length(x), tau0);
    fprintf('Output directory: %s\n', outdir);
    fprintf('Deviations to compute: %s\n', strjoin(upper(dev_list), ', '));
    fprintf('----------------------------------------\n');
    
    %-- Warn if totdevs are requested and estimated runtime is high
    totdev_types = {'totdev','mtotdev','htotdev','mhtotdev'};
    N = length(x);
    warn_threshold = 30; % seconds
    for k = 1:numel(dev_list)
        if any(strcmpi(dev_list{k}, totdev_types))
            est_runtime = 2e-5 * N^1.4;
            if est_runtime > warn_threshold
                fprintf(['WARNING: Calculation of %s may take a long time (estimated %.1f seconds for N = %d).\n'], ...
                    upper(dev_list{k}), est_runtime, N);
            end
        end
    end
    
    %-- Loop over requested deviation types
    for i = 1:numel(dev_list)
        devtype = lower(dev_list{i});
        
        try
            %-- Get function handle
            devfun = str2func(['allanlab.' devtype]);
            
            %-- Compute deviation
            fprintf('Calculating %s ... ', upper(devtype));
            tic;
            [tau, sigma, edf, ci, alpha] = devfun(x, tau0, m_list);
            comp_time = toc;
            
            %-- Ensure column vectors and clean up tau values
            tau = tau(:);
            sigma = sigma(:);
            alpha = alpha(:);
            edf = edf(:);
            
            %-- Round tau to remove floating point errors
            % For tau values that should be integers, round them
            tau_rounded = round(tau);
            tau_error = abs(tau - tau_rounded) ./ tau;
            is_integer_tau = tau_error < 1e-10;  % Relative error less than 1e-10
            tau(is_integer_tau) = tau_rounded(is_integer_tau);
            
            %-- Trim results where EDF is too low
            min_edf = 2;  % Minimum useful EDF
            valid_edf = edf >= min_edf & ~isnan(edf);
            
            % For some deviations (like totdev), EDF might be NaN
            % Only trim if we have actual low EDF values, not NaN
            has_low_edf = any(edf < min_edf & ~isnan(edf));
            
            if has_low_edf
                % Find last valid index based on EDF
                last_valid = find(valid_edf, 1, 'last');
                if isempty(last_valid)
                    warning('No valid results with EDF >= %g for %s', min_edf, upper(devtype));
                    results.failed{end+1} = devtype;
                    continue;
                end
                
                % Trim all arrays
                tau = tau(1:last_valid);
                sigma = sigma(1:last_valid);
                alpha = alpha(1:last_valid);
                edf = edf(1:last_valid);
                ci = ci(1:last_valid, :);
                
                fprintf('\n  Note: Trimmed %s at tau = %g (EDF < %g)', ...
                    upper(devtype), tau(end), min_edf);
            end
            
            %-- Validate results
            if isempty(tau) || all(isnan(sigma))
                warning('No valid results for %s', upper(devtype));
                results.failed{end+1} = devtype;
                continue;
            end
            
            %-- Create output table
            output_table = table(tau, alpha, sigma, ci(:,1), ci(:,2), edf, ...
                'VariableNames', {'tau', 'alpha', 'sigma', 'sigma_min', 'sigma_max', 'edf'});
            
            %-- Write to CSV file
            outfile = fullfile(outdir, [base '_' devtype '.csv']);
            writetable(output_table, outfile, ...
                'WriteVariableNames', true);
            
            %-- Store results
            results.(devtype) = output_table;
            results.computed{end+1} = devtype;
            
            fprintf('done (%.2f s). Written to: %s\n', comp_time, [base '_' devtype '.csv']);
            
        catch ME
            warning('\nFailed to compute %s: %s', upper(devtype), ME.message);
            results.failed{end+1} = devtype;
        end
    end
    
    %-- Summary
    fprintf('----------------------------------------\n');
    fprintf('Successfully computed: %d/%d deviations\n', ...
            length(results.computed), length(dev_list));
    
    if ~isempty(results.failed)
        fprintf('Failed: %s\n', strjoin(upper(results.failed), ', '));
    end
    
    %-- Create summary file if multiple deviations were computed
    if length(results.computed) > 1
        create_summary_file(results, base, outdir);
    end
end

function create_summary_file(results, base, outdir)
    %-- Create a summary CSV with all computed deviations
    
    % Get the first computed deviation to establish tau grid and alpha
    first_dev = results.computed{1};
    tau_ref = results.(first_dev).tau;
    alpha_ref = results.(first_dev).alpha;
    
    % Initialize summary table
    summary = table();
    summary.tau = tau_ref;
    summary.alpha = alpha_ref;  % Single alpha column
    
    % Add each computed deviation
    all_match = true;
    for i = 1:length(results.computed)
        devtype = results.computed{i};
        dev_data = results.(devtype);
        
        % Check if tau grids match
        if isequal(dev_data.tau, tau_ref)
            summary.(devtype) = dev_data.sigma;
        else
            all_match = false;
            warning('Tau grid for %s does not match reference. Skipping in summary.', upper(devtype));
        end
    end
    
    if all_match && width(summary) > 2  % More than just tau and alpha
        % Save summary
        summaryfile = fullfile(outdir, [base '_summary.csv']);
        writetable(summary, summaryfile);
        fprintf('\nSummary file created: %s\n', summaryfile);
    end
end

function y = convert_units(x, unit)
%CONVERT_UNITS Convert time/phase data to seconds
    switch lower(unit)
        case {'s', 'sec', 'seconds'}
            y = x;
        case {'ns', 'nanoseconds'}
            y = x * 1e-9;
        case {'ps', 'picoseconds'}
            y = x * 1e-12;
        case {'mjd'}
            y = (x - x(1)) * 86400;  % Convert to seconds from first point
        otherwise
            warning('Unknown unit "%s", using as-is.', unit);
            y = x;
    end
end