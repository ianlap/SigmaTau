function save_adev_table(filename, adev_results)
% SAVE_ADEV_TABLE - Save Allan deviation data to CSV file
%
% This utility function saves Allan deviation results from multiple tests
% to a single CSV file for easy comparison and analysis.
%
% Inputs:
%   filename     - Output filename for CSV
%   adev_results - Cell array of ADEV result structs, each containing:
%                  .test_label - Label for the test
%                  .tau - Tau values [s]
%                  .adev - Allan deviation values
%
% Output format:
%   CSV file with columns: Test_Label, Tau_s, ADEV

% Collect all data
test_labels = {};
tau_vals = [];
adev_vals = [];

for i = 1:length(adev_results)
    if ~isempty(adev_results{i}) && isfield(adev_results{i}, 'tau') && isfield(adev_results{i}, 'adev')
        n_points = length(adev_results{i}.tau);
        
        % Ensure tau and adev are column vectors
        tau_i = adev_results{i}.tau(:);
        adev_i = adev_results{i}.adev(:);
        
        % Append to arrays
        test_labels = [test_labels; repmat({adev_results{i}.test_label}, n_points, 1)];
        tau_vals = [tau_vals; tau_i];
        adev_vals = [adev_vals; adev_i];
    end
end

if ~isempty(tau_vals)
    % Create table
    T = table(test_labels, tau_vals, adev_vals, ...
              'VariableNames', {'Test_Label', 'Tau_s', 'ADEV'});
    
    % Write to CSV
    try
        writetable(T, filename);
    catch ME
        error('SAVE_ADEV_TABLE:WriteError', ...
            'Failed to write ADEV table to %s: %s', filename, ME.message);
    end
else
    % Create empty file with header
    fid = fopen(filename, 'w');
    if fid ~= -1
        fprintf(fid, 'Test_Label,Tau_s,ADEV\n');
        fprintf(fid, '# No ADEV data available\n');
        fclose(fid);
    else
        warning('SAVE_ADEV_TABLE:NoData', 'No ADEV data to save');
    end
end

end