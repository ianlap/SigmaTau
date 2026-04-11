function save_kalman_results(results, filename, analysis_type)
% SAVE_KALMAN_RESULTS - Save Kalman filter results to CSV file
%
% This utility function saves Kalman filter time series results to a CSV file
% with appropriate headers and formatting.
%
% Inputs:
%   results       - Struct containing KF results with fields:
%                   .test_label, .phase, .freq, .drift, .innovation, .steer, .rtP00
%                   .g_p, .g_i, .g_d (gain values)
%   filename      - Output filename for CSV
%   analysis_type - String describing analysis type (e.g., 'PID', 'PD', 'STEERING')
%
% Note: This function assumes results contains full time series data.
%       If using memory-efficient storage, ensure required fields exist.

% Check if we have the required data
if ~isfield(results, 'phase') || isempty(results.phase)
    warning('SAVE_KALMAN_RESULTS:NoData', ...
        'No phase data available for %s - CSV export skipped', results.test_label);
    return;
end

N = length(results.phase);

% Build data matrix with available fields
data_columns = (1:N)';  % time index
column_names = {'time'};

% Add available data columns
if isfield(results, 'phase') && length(results.phase) == N
    data_columns = [data_columns, results.phase];
    column_names{end+1} = 'phase[ns]';
end

if isfield(results, 'freq') && length(results.freq) == N
    data_columns = [data_columns, results.freq];
    column_names{end+1} = 'freq[ns/s]';
end

if isfield(results, 'drift') && length(results.drift) == N
    data_columns = [data_columns, results.drift];
    column_names{end+1} = 'drift[ns/s²]';
end

if isfield(results, 'innovation') && length(results.innovation) == N
    data_columns = [data_columns, results.innovation];
    column_names{end+1} = 'innovation[ns]';
end

if isfield(results, 'steer') && length(results.steer) == N
    data_columns = [data_columns, results.steer];
    column_names{end+1} = 'steer[ns]';
end

if isfield(results, 'rtP00') && length(results.rtP00) == N
    data_columns = [data_columns, results.rtP00];
    column_names{end+1} = 'rtP00[ns]';
end

% Write header
fid = fopen(filename, 'w');
if fid == -1
    error('SAVE_KALMAN_RESULTS:FileError', 'Cannot create file: %s', filename);
end

fprintf(fid, '# Kalman filter output for %s (%s control)\n', results.test_label, upper(analysis_type));
fprintf(fid, '# Gains: g_p=%.6f, g_i=%.6f, g_d=%.6f\n', ...
        results.g_p, results.g_i, results.g_d);

% Add notes about control type
if abs(results.g_i) == 0
    fprintf(fid, '# Note: g_i≈0 indicates PD-style control (no integral term)\n');
elseif results.g_i < 0
    fprintf(fid, '# Note: Negative integral gain (g_i<0)\n');
end

% Write column names
fprintf(fid, '# Columns: %s\n', strjoin(column_names, ', '));
fclose(fid);

% Append data using writematrix (faster than dlmwrite)
writematrix(data_columns, filename, 'WriteMode', 'append', 'Delimiter', ',');

end