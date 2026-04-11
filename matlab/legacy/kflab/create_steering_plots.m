function fig = create_steering_plots(results_all, adev_results, config, N, analysis_type)
% CREATE_STEERING_PLOTS - Create standardized plots for steering analysis
%
% This function creates a 2x2 grid plot for steering analysis:
% - Top Left: Allan Deviation curves for all tests
% - Bottom Left: Time Deviation curves for all tests
% - Top Right: Phase time series for selected tests  
% - Bottom Right: Frequency time series for selected tests
%
% Inputs:
%   results_all  - Cell array of KF results for each test
%   adev_results - Cell array of ADEV results for each test
%   config       - Configuration structure
%   N            - Number of data points
%   analysis_type - String: analysis type for titles
%
% Output:
%   fig - Figure handle

%% Create main figure
fig = figure('Position', [50, 50, 1400, 800]);
colors = lines(config.n_tests);

%% Top Left: Allan Deviation for all tests
subplot(2, 2, 1);
hold on;

for i = 1:config.n_tests
    if ~isempty(adev_results{i})
        loglog(adev_results{i}.tau, adev_results{i}.adev, 'o-', ...
               'Color', colors(i,:), 'LineWidth', 1, 'MarkerSize', 4, ...
               'DisplayName', results_all{i}.test_label);
    end
end

xlabel('Averaging Time τ [s]');
ylabel('Allan Deviation');

% Set title based on gain mode
if strcmp(config.gain_mode, 'pd_timeconstant')
    title('ADEV - PD Gains (No Integral Term)');
elseif strcmp(config.gain_mode, 'pid_timeconstant')
    title('ADEV - PID Critical Gains');
else
    title('ADEV - Free Gain Specification');
end

grid on;
legend('Location', 'southwest', 'NumColumns', 2);
set(gca, 'XScale', 'log', 'YScale', 'log');
xlim([1, 1e6]);
ylim([1e-20, 1e-5]);

%% Bottom Left: Time Deviation for all tests
subplot(2, 2, 3);
hold on;

% Compute TDEV for each test
for i = 1:config.n_tests
    % Use phase data after maturity, convert to seconds
    phase_after_maturity = results_all{i}.kf_data.phase_est(config.kf.maturity:end);
    phase_data_s = phase_after_maturity * 1e-9;
    
    if length(phase_data_s) > 100
        try
            [tau_vals, tdev_vals, ~, ~, ~] = allanlab.tdev(phase_data_s, config.data.tau0, []);
            loglog(tau_vals, tdev_vals, 'o-', ...
                   'Color', colors(i,:), 'LineWidth', 1, 'MarkerSize', 4, ...
                   'DisplayName', results_all{i}.test_label);
        catch
            % Skip if TDEV computation fails
        end
    end
end

xlabel('Averaging Time τ [s]');
ylabel('Time Deviation [s]');

% Set title based on gain mode
if strcmp(config.gain_mode, 'pd_timeconstant')
    title('TDEV - PD Gains (No Integral Term)');
elseif strcmp(config.gain_mode, 'pid_timeconstant')
    title('TDEV - PID Critical Gains');
else
    title('TDEV - Free Gain Specification');
end

grid on;
legend('Location', 'southwest', 'NumColumns', 2);
set(gca, 'XScale', 'log', 'YScale', 'log');
xlim([1, 1e6]);

%% Top Right: Phase time series for selected tests
subplot(2, 2, 2);
hold on;

% Show up to 6 tests for clarity (4 for PID mode to match original)
if strcmp(config.gain_mode, 'pid_timeconstant')
    max_show = 4;
else
    max_show = 6;
end
show_indices = 1:min(max_show, config.n_tests);
time_s = (0:N-1)' * config.data.tau0;

for idx = show_indices(end:-1:1)  % Plot in reverse order
    plot(time_s, results_all{idx}.kf_data.phase_est, ...
         'Color', colors(idx,:), 'DisplayName', results_all{idx}.test_label, ...
         'LineWidth', 0.8);
end

xlabel('Time [s]');
ylabel('Phase [ns]');
if ismember(config.gain_mode, {'pd_timeconstant', 'pid_timeconstant'})
    title(sprintf('%s Steering - Phase Estimates', upper(analysis_type)));
else
    title('Steering - Phase Estimates');
end
grid on;
legend('Location', 'best');
ylim([-25, 25]);

%% Bottom Right: Frequency time series for selected tests
subplot(2, 2, 4);
hold on;

% Plot frequency estimates for the same selected tests
% Plot in normal order (large T to small T) so smaller variations are visible
for idx = show_indices  % Normal order for frequency
    plot(time_s, results_all{idx}.kf_data.freq_est, ...
         'Color', colors(idx,:), 'DisplayName', results_all{idx}.test_label, ...
         'LineWidth', 0.8);
end

xlabel('Time [s]');
ylabel('Frequency [ns/s]');
if ismember(config.gain_mode, {'pd_timeconstant', 'pid_timeconstant'})
    title(sprintf('%s Steering - Frequency Estimates', upper(analysis_type)));
else
    title('Steering - Frequency Estimates');
end
grid on;
legend('Location', 'best');

% Set appropriate y-limits based on data
all_freq = [];
for idx = show_indices
    all_freq = [all_freq; results_all{idx}.kf_data.freq_est(:)];
end
if ~isempty(all_freq)
    ylim([-2.5, 2.5]);
end

