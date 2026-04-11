function plot_optimization_results(q0, qOpt, res0, resOpt, cfg)
% PLOT_OPTIMIZATION_RESULTS - Visualize KF Q-parameter optimization results
%
% Creates a 6-panel figure showing optimization performance and parameter changes:
%   Top row:    RMS vs horizon | % improvement | Q-parameter comparison
%   Bottom row: Cost surface | Cost vs q_wfm | Cost vs q_rwfm
%
% Syntax:
%   plot_optimization_results(q0, qOpt, res0, resOpt, cfg)
%
% Inputs:
%   q0     - Initial Q parameters struct (.q_wpm, .q_wfm, .q_rwfm, .q_irwfm)
%   qOpt   - Optimized Q parameters struct (same fields)
%   res0   - Initial results struct (.rms_stats with .horizon and .rms_error)
%   resOpt - Optimized results struct (same fields plus .search_history)
%   cfg    - Optimization config struct (optional, for target horizons)
%
% Notes:
%   - WPM is fixed during optimization, so only WFM and RWFM are varied
%   - Bottom panels show the cost function landscape and 1D slices
%   - Color scale uses 'turbo' colormap for better visualization
%
% Example:
%   [q_opt, results] = optimize_kf(phase_data, tau, q_init, config);
%   plot_optimization_results(q_init, q_opt, results_init, results, config);
%
% See also: OPTIMIZE_KF, KF_PREDICT

%% Create figure with specified layout
figure('Name', 'KF Q-Parameter Optimization Summary', ...
       'Units', 'normalized', ...
       'OuterPosition', [0.03 0.05 0.94 0.88]);

tl = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Panel 1: RMS prediction error vs horizon
nexttile(tl);
plot_rms_comparison(res0, resOpt, cfg);

%% Panel 2: Percentage improvement
nexttile(tl);
plot_improvement(res0, resOpt);

%% Panel 3: Q-parameter bar chart
nexttile(tl);
plot_parameter_comparison(q0, qOpt);

%% Panels 4-6: Search space visualization (if search history available)
if isfield(resOpt, 'search_history')
    % Panel 4: 2D cost surface (q_wfm vs q_rwfm)
    nexttile(tl);
    plot_2d_cost_surface(resOpt.search_history, q0, qOpt);
    
    % Panel 5: Cost vs q_wfm (at optimal q_rwfm)
    nexttile(tl);
    plot_1d_slice(resOpt.search_history, q0, qOpt, 'wfm');
    
    % Panel 6: Cost vs q_rwfm (at optimal q_wfm)
    nexttile(tl);
    plot_1d_slice(resOpt.search_history, q0, qOpt, 'rwfm');
end

%% Add overall title with improvement metric
add_summary_title(res0, resOpt);

end

%% ===================== Helper Functions =====================

function plot_rms_comparison(res0, resOpt, cfg)
% Plot RMS error vs prediction horizon for initial and optimized parameters

% Plot main curves
plot(res0.rms_stats.horizon, res0.rms_stats.rms_error, '-r', ...
     'LineWidth', 1.6, 'DisplayName', 'Initial');
hold on;
plot(resOpt.rms_stats.horizon, resOpt.rms_stats.rms_error, '-b', ...
     'LineWidth', 1.6, 'DisplayName', 'Optimized');

% Highlight target horizons if provided
if nargin >= 3 && isfield(cfg, 'target_horizons')
    for h = cfg.target_horizons(:)'
        idx = find(resOpt.rms_stats.horizon == h, 1);
        if ~isempty(idx)
            plot(h, resOpt.rms_stats.rms_error(idx), 'bo', ...
                 'MarkerSize', 7, 'LineWidth', 1.2, ...
                 'HandleVisibility', 'off');
        end
    end
    % Add one marker to legend
    plot(NaN, NaN, 'bo', 'MarkerSize', 7, 'LineWidth', 1.2, ...
         'DisplayName', 'Target horizons');
end

% Formatting
xlabel('τ (prediction horizon) [samples]');
ylabel('RMS error [ns]');
title('Prediction Performance');
grid on;
legend('Location', 'northwest');
set(gca, 'XScale', 'log', 'YScale', 'log');

end

function plot_improvement(res0, resOpt)
% Plot percentage improvement vs horizon

% Find common horizons
common_h = intersect(res0.rms_stats.horizon, resOpt.rms_stats.horizon);

% Interpolate RMS values at common horizons
rms0 = interp1(res0.rms_stats.horizon, res0.rms_stats.rms_error, common_h);
rmsOpt = interp1(resOpt.rms_stats.horizon, resOpt.rms_stats.rms_error, common_h);

% Calculate percentage improvement
improvement = 100 * (rms0 - rmsOpt) ./ rms0;

% Plot
plot(common_h, improvement, '-g', 'LineWidth', 1.6);
hold on;
yline(0, 'k:', 'LineWidth', 1);

% Formatting
xlabel('τ (prediction horizon) [samples]');
ylabel('Improvement [%]');
title('Optimization Gain');
grid on;
set(gca, 'XScale', 'log');
xlim([min(common_h), max(common_h)]);

end

function plot_parameter_comparison(q0, qOpt)
% Bar chart comparing initial and optimized Q parameters

% Parameter names and values
names = {'q_{wpm}', 'q_{wfm}', 'q_{rwfm}', 'q_{irwfm}'};
v0 = [q0.q_wpm, q0.q_wfm, q0.q_rwfm, 0];
v1 = [qOpt.q_wpm, qOpt.q_wfm, qOpt.q_rwfm, 0];

% Handle optional q_irwfm
if isfield(q0, 'q_irwfm'), v0(4) = q0.q_irwfm; end
if isfield(qOpt, 'q_irwfm'), v1(4) = qOpt.q_irwfm; end

% Create grouped bar chart
b = bar(1:4, [v0; v1]', 'grouped');
b(1).FaceColor = [0.85 0.33 0.09];  % Orange for initial
b(2).FaceColor = [0.00 0.45 0.74];  % Blue for optimized

% Add text annotation for fixed WPM
text(1, max(v0(1), v1(1)) * 1.5, 'FIXED', ...
     'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Formatting
set(gca, 'YScale', 'log', 'XTick', 1:4, 'XTickLabel', names);
ylabel('Value');
title('Q Parameters');
legend('Initial', 'Optimized', ...
       'Location', 'southoutside', 'Orientation', 'horizontal');
grid on;

end

function plot_2d_cost_surface(search_history, q0, qOpt)
% Plot 2D contour plot for q_wfm vs q_rwfm

% Extract search history matrix: [q_wpm, q_wfm, q_rwfm, q_irwfm, cost]
H = search_history;

% Check if we have a regular grid or scattered points
q1_unique = unique(H(:, 2));
q2_unique = unique(H(:, 3));

if length(q1_unique) > 2 && length(q2_unique) > 2 && ...
   length(H) == length(q1_unique) * length(q2_unique)
    % Regular grid data - use contour
    try
        [Q1, Q2] = meshgrid(q1_unique, q2_unique);
        Cost = reshape(H(:, 5), length(q2_unique), length(q1_unique));
        contourf(Q1, Q2, Cost, 15);
        
        % Mark initial and optimal points
        hold on;
        plot(q0.q_wfm, q0.q_rwfm, '^r', ...
             'MarkerFaceColor', 'r', 'MarkerSize', 8, ...
             'DisplayName', 'Initial');
        plot(qOpt.q_wfm, qOpt.q_rwfm, '*g', ...
             'MarkerSize', 12, 'LineWidth', 1.5, ...
             'DisplayName', 'Optimal');
              
    catch
        % Fallback to scatter plot
        scatter(H(:, 2), H(:, 3), 60, H(:, 5), 'filled', 'Marker', 's');
        
        % Mark initial and optimal points
        hold on;
        plot(q0.q_wfm, q0.q_rwfm, '^r', ...
             'MarkerFaceColor', 'r', 'MarkerSize', 8, ...
             'DisplayName', 'Initial');
        plot(qOpt.q_wfm, qOpt.q_rwfm, '*g', ...
             'MarkerSize', 12, 'LineWidth', 1.5, ...
             'DisplayName', 'Optimal');
    end
else
    % Scattered data - use scatter
    scatter(H(:, 2), H(:, 3), 60, H(:, 5), 'filled', 'Marker', 's');
    
    % Mark initial and optimal points
    hold on;
    plot(q0.q_wfm, q0.q_rwfm, '^r', ...
         'MarkerFaceColor', 'r', 'MarkerSize', 8, ...
         'DisplayName', 'Initial');
    plot(qOpt.q_wfm, qOpt.q_rwfm, '*g', ...
         'MarkerSize', 12, 'LineWidth', 1.5, ...
         'DisplayName', 'Optimal');
end

% Set log scale and formatting
set(gca, 'XScale', 'log', 'YScale', 'log');
colormap(gca, 'turbo');
c = colorbar;
c.Label.String = 'Weighted RMS Cost';
xlabel('q_{wfm}');
ylabel('q_{rwfm}');
title('Cost Surface (WPM fixed)');
grid on;
legend('Location', 'best');
view(2);  % Force 2D view

end

function plot_1d_slice(search_history, q0, qOpt, param_type)
% Plot 1D slice of cost function for specified parameter

% Extract search history
H = search_history;

switch param_type
    case 'wfm'
        % Fix q_rwfm at optimal value, vary q_wfm
        fixed_idx = 3;
        vary_idx = 2;
        fixed_val = qOpt.q_rwfm;
        x_label = 'q_{wfm}';
        title_str = sprintf('Cost vs q_{wfm} (q_{rwfm} = %.2e)', fixed_val);
        x0 = q0.q_wfm;
        xopt = qOpt.q_wfm;
        
    case 'rwfm'
        % Fix q_wfm at optimal value, vary q_rwfm
        fixed_idx = 2;
        vary_idx = 3;
        fixed_val = qOpt.q_wfm;
        x_label = 'q_{rwfm}';
        title_str = sprintf('Cost vs q_{rwfm} (q_{wfm} = %.2e)', fixed_val);
        x0 = q0.q_rwfm;
        xopt = qOpt.q_rwfm;
end

% Extract points along this slice (with tolerance)
tol = fixed_val * 0.1;  % 10% tolerance
slice_mask = abs(H(:, fixed_idx) - fixed_val) < tol;
slice_data = H(slice_mask, :);

if ~isempty(slice_data)
    % Sort by varying parameter
    [~, sort_idx] = sort(slice_data(:, vary_idx));
    slice_data = slice_data(sort_idx, :);
    
    % Plot cost vs parameter
    plot(slice_data(:, vary_idx), slice_data(:, 5), 'b.-', ...
         'LineWidth', 1.5, 'MarkerSize', 8);
else
    % No exact slice data - plot all points with transparency
    scatter(H(:, vary_idx), H(:, 5), 20, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
    title_str = [title_str ' (all points shown)'];
end

% Mark initial and optimal values
hold on;
xline(x0, 'r--', 'LineWidth', 1.5, 'Alpha', 0.7);
xline(xopt, 'g--', 'LineWidth', 1.5, 'Alpha', 0.7);

% Add markers at specific points
ylims = ylim;
plot(x0, ylims(2)*0.95, 'rv', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(xopt, ylims(2)*0.95, 'g^', 'MarkerSize', 8, 'MarkerFaceColor', 'g');

% Formatting
set(gca, 'XScale', 'log');
xlabel(x_label);
ylabel('Weighted RMS Cost');
title(title_str);
grid on;

end

function add_summary_title(res0, resOpt)
% Add figure title with overall improvement metric

if isfield(res0, 'weighted_rms') && isfield(resOpt, 'weighted_rms')
    % Calculate weighted improvement
    weighted_gain = 100 * (res0.weighted_rms - resOpt.weighted_rms) / res0.weighted_rms;
    sgtitle(sprintf('KF Q-Parameter Optimization   •   Weighted improvement = %.1f%%', ...
                    weighted_gain));
else
    sgtitle('KF Q-Parameter Optimization Results');
end

end