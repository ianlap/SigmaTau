function [q0, q1, q2, regions] = mhtot_fit_wplot(sigma, tau, ci, maxIter)
% MHTOT_FIT - Interactive fitting of MHTOTDEV variance data with visualization
%
% Fits noise components to Modified Hadamard Total Deviation data:
%   σ² = (10/3)·q0·τ⁻³  +  (4/7)·q1·τ⁻¹  +  (5/22)·q2·τ⁺¹
%
% Inputs:
%   sigma   - MHTOTDEV values [units]
%   tau     - Averaging times [s]
%   ci      - Confidence intervals [lower upper]
%   maxIter - Maximum iterations (default: 6)
%
% Outputs:
%   q0 - White phase modulation coefficient
%   q1 - White frequency modulation coefficient  
%   q2 - Random walk frequency modulation coefficient
%   regions - Details of fitted regions
%
% Usage:
%   The user selects noise type and tau range for each fit.
%   Fits are applied sequentially to the residual.

%% Initialize
if nargin < 4, maxIter = 6; end

% Sort data by tau
[tau, idx] = sort(tau(:));
sigma = sigma(idx);

% Extract confidence intervals
if size(ci, 2) == 2
    ciLow = ci(idx, 1);
    ciUp = ci(idx, 2);
else
    error('CI must be Nx2 matrix [lower upper]');
end

% Calculate weights from confidence intervals
weights = ci2weights(sigma, ciLow, ciUp, 'conservative');

% Convert to variance
var_original = sigma.^2;
var_residual = var_original;

% Initialize outputs
q0 = 0;  q1 = 0;  q2 = 0;
regions = struct('wpm', {{}}, 'wfm', {{}}, 'rwfm', {{}});
history = struct('var_residual', {}, 'q0', {}, 'q1', {}, 'q2', {}, 'regions', {});  % Initialize as empty struct array

%% Create figure
hFig = figure('Name', 'MHTOTDEV Fitting', 'Position', [100, 50, 1200, 800]);

% Show initial data
fprintf('\n=== MHTOTDEV FITTING ===\n');
showDataTable(tau, var_residual);
updatePlot(hFig, tau, sigma, ciLow, ciUp, var_original, var_residual, ...
           q0, q1, q2, 'Initial Data');

%% Main fitting loop
for iter = 1:maxIter
    % Get user choice
    fprintf('\nIteration %d - Select action:\n', iter);
    fprintf('  1 = Fit WPM (slope -3 fixed)\n');
    fprintf('  2 = Fit WFM (slope -1 fixed)\n');
    fprintf('  3 = Fit RWFM (slope +1 fixed)\n');
    fprintf('  5 = Undo last fit\n');
    fprintf('  0 = Done\n');
    choice = input('Choice: ');
    
    % Handle choice
    if choice == 0
        break;  % Done
    elseif choice == 5
        % Undo last fit
        if isempty(history)
            fprintf('Nothing to undo!\n');
            continue;
        end
        last = history(end);
        var_residual = last.var_residual;
        q0 = last.q0;  q1 = last.q1;  q2 = last.q2;
        regions = last.regions;
        history(end) = [];
        fprintf('Undone!\n');
        updatePlot(hFig, tau, sigma, ciLow, ciUp, var_original, var_residual, ...
                   q0, q1, q2, 'After Undo');
        continue;
    elseif ~ismember(choice, [1 2 3])
        fprintf('Invalid choice!\n');
        continue;
    end
    
    % Get tau range
    range = input('Index range [start end]: ');
    if length(range) ~= 2 || range(1) < 1 || range(2) > length(tau) || range(1) >= range(2)
        fprintf('Invalid range!\n');
        continue;
    end
    indices = range(1):range(2);
    
    % Save current state before fitting
    history(end+1) = struct('var_residual', var_residual, ...
                           'q0', q0, 'q1', q1, 'q2', q2, 'regions', regions);
    
    % Extract data for fitting
    tau_fit = tau(indices);
    var_fit = var_residual(indices);
    weights_fit = weights(indices);
    
    % Fit based on choice
    
    % Fixed slope fitting
    noiseTypes = {'wpm', 'wfm', 'rwfm'};
    noiseType = noiseTypes{choice};
    [q_fit, q_std] = fitFixedSlope(noiseType, tau_fit, var_fit, weights_fit);
    slopes = struct('wpm', -3, 'wfm', -1, 'rwfm', 1);
    slope_fit = slopes.(noiseType);
    
    % Update residual
    var_residual = updateResidual(noiseType, q_fit, tau, var_residual);
    
    % Store results
    regions.(noiseType){end+1} = struct('indices', indices, 'q', q_fit, ...
                                        'q_std', q_std, 'slope', slope_fit);
    
    % Update total coefficients
    switch noiseType
        case 'wpm',  q0 = q0 + q_fit;
        case 'wfm',  q1 = q1 + q_fit;
        case 'rwfm', q2 = q2 + q_fit;
    end
    
    % Update display
    fprintf('Fitted %s: q = %.3e ± %.3e (slope = %.2f)\n', ...
            upper(noiseType), q_fit, q_std, slope_fit);
    showDataTable(tau, var_residual);
    updatePlot(hFig, tau, sigma, ciLow, ciUp, var_original, var_residual, ...
               q0, q1, q2, sprintf('After Iteration %d', iter));
end

%% Final results
fprintf('\n=== FINAL RESULTS ===\n');
fprintf('q0 (WPM)  = %.3e\n', q0);
fprintf('q1 (WFM)  = %.3e\n', q1);
fprintf('q2 (RWFM) = %.3e\n', q2);

updatePlot(hFig, tau, sigma, ciLow, ciUp, var_original, var_residual, ...
           q0, q1, q2, 'Final Fit');

% Ask to keep plot
if strcmpi(input('\nKeep plot open? (y/n) [n]: ', 's'), 'y')
    fprintf('Plot kept open.\n');
else
    close(hFig);
end

end

%% ========== Helper Functions ==========

function showDataTable(tau, variance)
% Display data table with indices
fprintf('\nIndex    Tau[s]     Variance    Slope\n');
fprintf('-----  ----------  ----------  ------\n');
for k = 1:length(tau)
    if k < length(tau)
        slope = diff(log10(variance(k:k+1))) / diff(log10(tau(k:k+1)));
        fprintf('%3d    %10.2e  %10.2e  %+5.1f\n', k, tau(k), variance(k), slope);
    else
        fprintf('%3d    %10.2e  %10.2e    -\n', k, tau(k), variance(k));
    end
end
end

function [q, q_std] = fitFixedSlope(noiseType, tau, variance, weights)
% Fit with fixed theoretical slope using weighted mean

% Define theoretical parameters
slopes = struct('wpm', -3, 'wfm', -1, 'rwfm', 1);
coeffs = struct('wpm', 10/3, 'wfm', 4/7, 'rwfm', 5/22);

slope = slopes.(noiseType);
coeff = coeffs.(noiseType);

% Model: log10(variance) = log10(coeff * q) + slope * log10(tau)
% Rearrange: log10(variance) - slope * log10(tau) = log10(coeff * q)

log_tau = log10(tau);
log_var = log10(variance);

% Adjust y values by subtracting the known slope contribution
y_adjusted = log_var - slope * log_tau;

% Use weighted mean to get log10(coeff * q)
[log_coeff_q, log_std] = weightedMean(y_adjusted, weights);

% Extract q from coeff * q
coeff_q = 10^log_coeff_q;
q = coeff_q / coeff;

% Uncertainty propagation: d(q)/d(log) = q * ln(10)
q_std = q * log(10) * log_std;
end


function var_new = updateResidual(noiseType, q, tau, var_old)
% Subtract fitted component from variance

% Define theoretical parameters
slopes = struct('wpm', -3, 'wfm', -1, 'rwfm', 1);
coeffs = struct('wpm', 10/3, 'wfm', 4/7, 'rwfm', 5/22);

slope = slopes.(noiseType);
coeff = coeffs.(noiseType);

% Subtract component
var_new = var_old - coeff * q * tau.^slope;
var_new(var_new < 0) = 0;  % Ensure non-negative
end

function updatePlot(hFig, tau, sigma, ciLow, ciUp, var_orig, var_resid, q0, q1, q2, titleStr)
% Update the figure with current state

figure(hFig); clf;

% Main plot - MHTOTDEV with model
subplot(2,2,[1 3]);
loglog(tau, sigma, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 10);
hold on;

% Confidence intervals
fill([tau; flipud(tau)], [ciLow; flipud(ciUp)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Add index labels
addIndexLabels(tau, sigma, 10);

% Plot model if any components fitted
if q0 > 0 || q1 > 0 || q2 > 0
    tau_model = logspace(log10(min(tau)), log10(max(tau)), 200);
    sigma2_total = zeros(size(tau_model));
    
    % Store handles for legend
    h_components = [];
    legend_entries = {};
    
    if q0 > 0
        sigma2_wpm = (10/3) * q0 * tau_model.^(-3);
        sigma2_total = sigma2_total + sigma2_wpm;
        h = loglog(tau_model, sqrt(sigma2_wpm), 'r--', 'LineWidth', 1.5);
        h_components(end+1) = h;
        legend_entries{end+1} = sprintf('WPM (q=%.2e)', q0);
    end
    if q1 > 0
        sigma2_wfm = (4/7) * q1 * tau_model.^(-1);
        sigma2_total = sigma2_total + sigma2_wfm;
        h = loglog(tau_model, sqrt(sigma2_wfm), 'g--', 'LineWidth', 1.5);
        h_components(end+1) = h;
        legend_entries{end+1} = sprintf('WFM (q=%.2e)', q1);
    end
    if q2 > 0
        sigma2_rwfm = (5/22) * q2 * tau_model.^(1);
        sigma2_total = sigma2_total + sigma2_rwfm;
        h = loglog(tau_model, sqrt(sigma2_rwfm), 'm--', 'LineWidth', 1.5);
        h_components(end+1) = h;
        legend_entries{end+1} = sprintf('RWFM (q=%.2e)', q2);
    end
    
    h = loglog(tau_model, sqrt(sigma2_total), 'k-', 'LineWidth', 2);
    h_components(end+1) = h;
    legend_entries{end+1} = 'Total Model';
end

xlabel('Averaging Time τ [s]');
ylabel('MHTOTDEV σ');
title(titleStr);
grid on;

% Create legend with proper handles
if exist('h_components', 'var') && ~isempty(h_components)
    h_data = findobj(gca, 'Type', 'Line', 'LineWidth', 1.5, 'Marker', '.');
    legend([h_data(1); h_components'], ['Data'; legend_entries'], 'Location', 'best');
else
    legend('Data', 'Location', 'best');
end
% Residual plot
subplot(2,2,2);
loglog(tau, var_orig, 'b.-', 'LineWidth', 1.5);
hold on;
loglog(tau, var_resid, 'ro-', 'LineWidth', 1.5);
addIndexLabels(tau, var_resid, 10);
xlabel('τ [s]');
ylabel('Variance σ²');
title('Residual Analysis');
legend('Original', 'Residual', 'Location', 'best');
grid on;

% Summary text
subplot(2,2,4);
axis off;
text(0.1, 0.9, 'Fitted Coefficients:', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.7, sprintf('q₀ (WPM) = %.3e', q0), 'FontSize', 12);
text(0.1, 0.6, sprintf('q₁ (WFM) = %.3e', q1), 'FontSize', 12);
text(0.1, 0.5, sprintf('q₂ (RWFM) = %.3e', q2), 'FontSize', 12);

drawnow;
end

function addIndexLabels(tau, y, maxLabels)
% Add index labels to data points

N = length(tau);
if N <= maxLabels
    indices = 1:N;
else
    indices = unique(round(linspace(1, N, maxLabels)));
end

for idx = indices
    text(tau(idx)*1.1, y(idx)*1.1, num2str(idx), ...
         'FontSize', 9, 'Color', [0.5 0.5 0.5]);
end
end