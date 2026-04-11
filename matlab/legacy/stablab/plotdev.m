function plotdev(x, tau0, devtypes, m_list)
%PLOTDEV   Plot one or more stability deviation types using AllanLab tools
%           with Stable32-style formatting and data table overlay
%
%   plotdev(x, tau0, devtypes, m_list)
%
%   Inputs:
%     x         – phase data (seconds), vector
%     tau0      – basic sampling interval (seconds)
%     devtypes  – cell array of strings: e.g., {'adev','mdev','hdev'}
%     m_list    – optional averaging factors (powers of 2 recommended)
%
%   Example:
%     plotdev(phase, 1, {'adev','mdev','tdev'});

    import allanlab.*

    if isrow(x), x = x.'; end
    N = numel(x);

    % Default to octave m_list with N/3 maximum (Stable32 convention)
    if nargin < 4 || isempty(m_list)
        max_m = floor(N/3);
        % Use powers of 2 for cleaner display
        m_list = 2.^(0:floor(log2(max_m)));
        m_list = m_list(m_list <= max_m);
    end

    if ischar(devtypes)
        devtypes = {devtypes};  % allow single string input
    end

    % Create figure with specific size
    figure('Color', 'white', 'Position', [100 100 1200 700]);
    
    % Main plot axes - leave room for table
    ax = axes('Position', [0.08 0.11 0.58 0.815], 'Box', 'on', 'XScale', 'log', 'YScale', 'log');
    hold on;
    
    % Color scheme similar to Stable32
    colormap = [
        0    0    1;    % Blue (ADEV)
        1    0    0;    % Red (MDEV)
        0    0.5  0;    % Dark Green (HDEV)
        1    0.5  0;    % Orange (TDEV)
        0.5  0    0.5;  % Purple (PDEV)
        0    0.75 0.75; % Cyan
        0.75 0    0.75; % Magenta
    ];
    
    % Line styles for different deviations
    linestyles = {'-', '--', '-.', ':', '-', '--', '-.'};
    
    % Markers for different deviations
    markers = {'o', 's', '^', 'd', 'v', '>', '<'};
    
    legends = {};
    handles = [];
    
    % Store all results for table
    all_results = {};
    max_points = 0;
    
    % Store all tau and sigma values for axis limits
    all_tau = [];
    all_sigma = [];

    % Warn if totdevs are requested and estimated runtime is high
    totdev_types = {'totdev','mtotdev','htotdev','mhtotdev'};
    warn_threshold = 30; % seconds
    for k = 1:numel(devtypes)
        if any(strcmpi(devtypes{k}, totdev_types))
            est_runtime = 2e-5 * N^1.4;
            if est_runtime > warn_threshold
                fprintf(['WARNING: Calculation of %s may take a long time (estimated %.1f seconds for N = %d).\n'], ...
                    upper(devtypes{k}), est_runtime, N);
            end
        end
    end
    
    % Loop through each requested deviation type
    for i = 1:numel(devtypes)
        devtype = lower(devtypes{i});
        fprintf('Calculating %s ...\n', upper(devtype));
        
        try
            % Call the appropriate deviation function
            devfunc = str2func(['allanlab.' devtype]);
            
            % Get the deviation values - always try to get all 5 outputs
            % The function will handle if it doesn't support all outputs
            try
                [tau, sigma, edf, ci, alpha] = devfunc(x, tau0, m_list);
            catch
                % If 5 outputs failed, try with 4
                [tau, sigma, edf, ci] = devfunc(x, tau0, m_list);
            end
            
        catch ME
            warning("Could not compute %s: %s", devtype, ME.message);
            continue;
        end
        
        % Remove any NaN or Inf values
        valid_idx = isfinite(tau) & isfinite(sigma) & tau > 0 & sigma > 0;
        tau = tau(valid_idx);
        sigma = sigma(valid_idx);
        if exist('ci', 'var') && ~isempty(ci) && size(ci, 1) >= length(valid_idx)
            ci = ci(valid_idx, :);
        else
            ci = [];  % Clear ci if it's not properly sized
        end
        
        % Skip if no valid data
        if isempty(tau) || isempty(sigma)
            warning("No valid data for %s", upper(devtype));
            continue;
        end
        
        % Store results for table
        all_results{i}.tau = tau;
        all_results{i}.sigma = sigma;
        all_results{i}.type = upper(devtype);
        max_points = max(max_points, length(tau));
        
        % Store for axis limits
        all_tau = [all_tau; tau(:)];
        all_sigma = [all_sigma; sigma(:)];
        
        % Plot deviation curve with markers
        color_idx = mod(i-1, size(colormap, 1)) + 1;
        marker_idx = mod(i-1, length(markers)) + 1;
        
        % Determine which points to mark (max 20 markers for clarity)
        if length(tau) > 20
            marker_indices = round(linspace(1, length(tau), 20));
        else
            marker_indices = 1:length(tau);
        end
        
        % Plot line
        h = loglog(tau, sigma, ...
            'LineStyle', '-', ...  % Always use solid line
            'LineWidth', 2, ...
            'Color', colormap(color_idx, :));
        
        % Add markers at selected points
        loglog(tau(marker_indices), sigma(marker_indices), ...
            'LineStyle', 'none', ...
            'Marker', markers{marker_idx}, ...
            'MarkerSize', 8, ...
            'MarkerFaceColor', colormap(color_idx, :), ...
            'MarkerEdgeColor', colormap(color_idx, :) * 0.7, ...
            'HandleVisibility', 'off');
        
        handles(end+1) = h;
        legends{end+1} = upper(devtype);
        
        % Add error bars if confidence intervals exist
        if exist('ci','var') && size(ci,2) == 2 && all(ci(:) > 0)
            % For log-log plots, we need to handle error bars carefully
            % CI values are the actual deviation bounds, not distances
            
            % Only plot error bars at marker positions
            for j = marker_indices
                if ci(j,1) > 0 && ci(j,2) > 0
                    % Plot vertical line from lower to upper CI
                    plot([tau(j) tau(j)], [ci(j,1) ci(j,2)], '-', ...
                        'Color', colormap(color_idx, :), ...
                        'LineWidth', 1.5, ...
                        'HandleVisibility', 'off');
                    
                    % Add caps to error bars
                    cap_width = 0.03; % Width of caps in log units
                    tau_left = tau(j) / (10^cap_width);
                    tau_right = tau(j) * (10^cap_width);
                    
                    % Lower cap
                    plot([tau_left tau_right], [ci(j,1) ci(j,1)], '-', ...
                        'Color', colormap(color_idx, :), ...
                        'LineWidth', 1.5, ...
                        'HandleVisibility', 'off');
                    
                    % Upper cap
                    plot([tau_left tau_right], [ci(j,2) ci(j,2)], '-', ...
                        'Color', colormap(color_idx, :), ...
                        'LineWidth', 1.5, ...
                        'HandleVisibility', 'off');
                end
            end
        end
    end
    
    % Set axis properties - Stable32 style with larger fonts
    xlabel('Averaging Time \tau (s)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Deviation \sigma_y(\tau)', 'FontSize', 14, 'FontWeight', 'bold');
    title('Frequency Stability Analysis', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Grid - Stable32 uses full grid
    grid on;
    ax.GridLineStyle = '-';
    ax.GridAlpha = 0.15;
    ax.MinorGridLineStyle = '-';
    ax.MinorGridAlpha = 0.05;
    ax.XMinorGrid = 'on';
    ax.YMinorGrid = 'on';
    
    % Set axis limits with some padding
    if ~isempty(all_tau) && ~isempty(all_sigma)
        tau_range = [min(all_tau) max(all_tau)];
        sigma_range = [min(all_sigma) max(all_sigma)];
        
        xlim([tau_range(1)/1.5, tau_range(2)*1.5]);
        ylim([sigma_range(1)/2, sigma_range(2)*2]);
    end
    
    % Legend with larger font
    if ~isempty(handles)
        leg = legend(handles, legends, ...
            'Location', 'southwest', ...
            'FontSize', 12, ...
            'Box', 'on', ...
            'Color', 'white', ...
            'EdgeColor', 'black', ...
            'LineWidth', 1);
    end
    
    % Set font properties for all text
    set(ax, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1);
    
    % Add data table overlay
    add_data_table(all_results, colormap);
    
    % Add timestamp and data info
    add_info_text(N, tau0);
end

function add_data_table(all_results, colormap)
    % Add a formatted data table showing key tau/sigma values
    
    if isempty(all_results)
        return;
    end
    
    % Create table axes - adjusted position for better fit
    table_ax = axes('Position', [0.68 0.15 0.30 0.75], 'Visible', 'off');
    
    % Table title
    text(0.5, 0.98, 'Deviation Values', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top');
    
    % Determine which tau values to show
    % Show up to 15 evenly spaced points
    n_display = min(15, length(all_results{1}.tau));
    if length(all_results{1}.tau) > n_display
        tau_display_indices = round(linspace(1, length(all_results{1}.tau), n_display));
    else
        tau_display_indices = 1:length(all_results{1}.tau);
    end
    
    % Column setup
    y_start = 0.93;
    row_height = 0.055;
    
    % Column positions - adjusted for better spacing
    tau_col_x = 0.05;
    dev_col_start = 0.35;
    dev_col_width = 0.6 / length(all_results);
    
    % Column headers
    text(tau_col_x, y_start, '\tau (s)', 'FontSize', 11, 'FontWeight', 'bold');
    
    % Add deviation type headers
    for i = 1:length(all_results)
        if ~isempty(all_results{i})
            x_pos = dev_col_start + (i-1) * dev_col_width + dev_col_width/2;
            color_idx = mod(i-1, size(colormap, 1)) + 1;
            text(x_pos, y_start, all_results{i}.type, ...
                'FontSize', 11, 'FontWeight', 'bold', ...
                'Color', colormap(color_idx, :), ...
                'HorizontalAlignment', 'center');
        end
    end
    
    % Add horizontal line under headers
    y_line = y_start - 0.02;
    line([0.02 0.98], [y_line y_line], 'Color', 'k', 'LineWidth', 1);
    
    % Add data rows
    y_pos = y_start - row_height;
    
    for j = 1:length(tau_display_indices)
        idx = tau_display_indices(j);
        
        % Tau value
        if idx <= length(all_results{1}.tau)
            text(tau_col_x, y_pos, sprintf('%.3e', all_results{1}.tau(idx)), ...
                'FontSize', 9, 'FontName', 'Courier');
            
            % Deviation values
            for i = 1:length(all_results)
                if ~isempty(all_results{i}) && idx <= length(all_results{i}.sigma)
                    x_pos = dev_col_start + (i-1) * dev_col_width + dev_col_width/2;
                    text(x_pos, y_pos, ...
                        sprintf('%.3e', all_results{i}.sigma(idx)), ...
                        'FontSize', 9, 'FontName', 'Courier', ...
                        'HorizontalAlignment', 'center', ...
                        'Color', 'k');  % Black text for better readability
                end
            end
            
            y_pos = y_pos - row_height * 0.85;
            
            % Stop if we're getting too low
            if y_pos < 0.05
                break;
            end
        end
    end
    
    % Add border around table
    rectangle('Position', [0.01 0.01 0.98 0.98], ...
        'EdgeColor', 'k', 'LineWidth', 1);
end

function add_info_text(N, tau0)
    % Add timestamp and data information at bottom of figure
    
    info_str = sprintf('Data points: %d  |  \\tau_0 = %.3g s  |  %s', ...
        N, tau0, datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    
    annotation('textbox', [0.08 0.01 0.58 0.04], ...
        'String', info_str, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 10, ...
        'EdgeColor', 'none', ...
        'BackgroundColor', 'white');
end