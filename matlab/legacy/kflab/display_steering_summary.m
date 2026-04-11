function display_steering_summary(results_all, adev_results, summary_stats, config, analysis_type, mode_note)
% DISPLAY_STEERING_SUMMARY - Display and save steering analysis summary statistics
%
% This function handles the display and CSV export of steering analysis
% summary statistics in a consistent format.
%
% Inputs:
%   results_all   - Cell array of KF results for each test
%   adev_results  - Cell array of ADEV results for each test  
%   summary_stats - Matrix of summary statistics [T/index, phase_rms, freq_rms, steer_rms, min_adev]
%   config        - Configuration structure
%   analysis_type - String: analysis type for output file naming
%   mode_note     - String: explanatory note about the analysis mode

%% Display summary statistics
summary_title = sprintf('=== SUMMARY STATISTICS (%s Control) ===', upper(analysis_type));
fprintf('\n%s\n', summary_title);
fprintf('%s\n', mode_note);

% Display header based on mode
if ismember(config.gain_mode, {'pd_timeconstant', 'pid_timeconstant'})
    fprintf('%-8s | %-12s | %-12s | %-12s | %-12s | %-12s\n', ...
            'T', 'Phase RMS', 'Freq RMS', 'Steer RMS', 'Min ADEV', 'τ @ Min');
else
    fprintf('%-8s | %-12s | %-12s | %-12s | %-12s | %-12s\n', ...
            'Label', 'Phase RMS', 'Freq RMS', 'Steer RMS', 'Min ADEV', 'τ @ Min');
end
fprintf('---------|--------------|--------------|--------------|--------------|-------------\n');

%% Extract statistics arrays
T_array = summary_stats(:, 1);
phase_rms_array = summary_stats(:, 2);
freq_rms_array = summary_stats(:, 3);
steer_rms_array = summary_stats(:, 4);
min_adev_array = summary_stats(:, 5);
tau_min_array = zeros(size(T_array));

%% Display each test result
for i = 1:config.n_tests
    % Find tau at minimum ADEV
    if ~isempty(adev_results{i}) && ~isnan(min_adev_array(i))
        [~, min_idx] = min(adev_results{i}.adev);
        tau_min = adev_results{i}.tau(min_idx);
        tau_min_array(i) = tau_min;
        tau_str = sprintf('%.1f', tau_min);
    else
        tau_min_array(i) = NaN;
        tau_str = 'N/A';
    end
    
    % Display based on gain mode
    if ismember(config.gain_mode, {'pd_timeconstant', 'pid_timeconstant'})
        fprintf('%-8d | %12.3f | %12.6f | %12.3f | %12.2e | %12s\n', ...
                round(T_array(i)), phase_rms_array(i), freq_rms_array(i), ...
                steer_rms_array(i), min_adev_array(i), tau_str);
    else
        fprintf('%-8s | %12.3f | %12.6f | %12.3f | %12.2e | %12s\n', ...
                results_all{i}.test_label, phase_rms_array(i), freq_rms_array(i), ...
                steer_rms_array(i), min_adev_array(i), tau_str);
    end
end

%% Save summary to CSV
if config.output.save_results
    % Create test labels array
    test_labels = cell(config.n_tests, 1);
    for i = 1:config.n_tests
        test_labels{i} = results_all{i}.test_label;
    end
    
    % Create table based on gain mode
    if ismember(config.gain_mode, {'pd_timeconstant', 'pid_timeconstant'})
        summary_table = table(test_labels, T_array, phase_rms_array, freq_rms_array, steer_rms_array, ...
                             min_adev_array, tau_min_array, ...
                             'VariableNames', {'Test_Label', 'T', 'Phase_RMS_ns', 'Freq_RMS_ns_per_s', ...
                                              'Steer_RMS_ns', 'Min_ADEV', 'Tau_at_Min_s'});
    else
        summary_table = table(test_labels, phase_rms_array, freq_rms_array, steer_rms_array, ...
                             min_adev_array, tau_min_array, ...
                             'VariableNames', {'Test_Label', 'Phase_RMS_ns', 'Freq_RMS_ns_per_s', ...
                                              'Steer_RMS_ns', 'Min_ADEV', 'Tau_at_Min_s'});
    end
    
    % Save to CSV
    summary_filename = fullfile(config.output.output_dir, ...
                               sprintf('%s_summary_stats_%s.csv', lower(analysis_type), datestr(now, 'yyyymmdd_HHMMSS')));
    writetable(summary_table, summary_filename);
    fprintf('\nSummary statistics saved to: %s\n', summary_filename);
end

end