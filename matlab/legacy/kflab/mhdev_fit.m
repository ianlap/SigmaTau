function [q0, q1, q2, regions, q3, sig0ffm, sig0fpm] = mhdev_fit(sigma, tau, ci, maxIter)
% MHDEV_FIT - Interactive fitting of MHDEV variance data with visualization
%
% Fits noise components to Modified Hadamard Deviation data:
%   σ² = (10/3)·q0·τ⁻³  +  (7/16)·q1·τ⁻¹  +  (1/9)·q2·τ⁺¹ + (11/120)·q3·τ⁺³
%   For flicker noise: σ = sig0ffm·τ⁰ (FFM) or σ = sig0fpm·τ⁻² (FPM)
%
% Inputs:
%   sigma   - MHDEV values [units]
%   tau     - Averaging times [s]
%   ci      - Confidence intervals [lower upper]
%   maxIter - Maximum iterations (default: 6)
%
% Outputs:
%   q0 - White phase modulation coefficient
%   q1 - White frequency modulation coefficient  
%   q2 - Random walk frequency modulation coefficient
%   regions - Details of fitted regions
%   q3 - Random run frequency modulation coefficient
%   sig0ffm - FFM intercept (sigma at tau=1 for slope 0)
%   sig0fpm - FPM intercept (sigma at tau=1 for slope -2)
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
sigma_residual = sigma;  % Keep track of sigma residual for flicker fits

% Initialize outputs
q0 = 0;  q1 = 0;  q2 = 0;  q3 = 0;
sig0ffm = 0;  sig0fpm = 0;
regions = struct('wpm', {{}}, 'wfm', {{}}, 'rwfm', {{}}, 'rrfm', {{}}, 'ffm', {{}}, 'fpm', {{}});
history = struct('var_residual', {}, 'sigma_residual', {}, ...
                 'q0', {}, 'q1', {}, 'q2', {}, 'q3', {}, ...
                 'sig0ffm', {}, 'sig0fpm', {}, 'regions', {});

%% Create figure
hFig = figure('Name', 'MHDEV Fitting', 'Position', [100, 50, 1200, 800]);

% Show initial data
fprintf('\n=== MHDEV FITTING ===\n');
showDataTable(tau, var_residual);
updatePlot(hFig, tau, sigma, ciLow, ciUp, var_original, var_residual, ...
           q0, q1, q2, q3, sig0ffm, sig0fpm, 'Initial Data');

%% Main fitting loop
for iter = 1:maxIter
    % Get user choice
    fprintf('\nIteration %d - Select action:\n', iter);
    fprintf('  1 = Fit WPM (slope -3 fixed)\n');
    fprintf('  2 = Fit WFM (slope -1 fixed)\n');
    fprintf('  3 = Fit RWFM (slope +1 fixed)\n');
    fprintf('  4 = Fit RRFM (slope +3 fixed)\n');
    fprintf('  5 = Fit FFM (slope 0 fixed)\n');
    fprintf('  6 = Fit FPM (slope -2 fixed)\n');
    fprintf('  7 = Undo last fit\n');
    fprintf('  0 = Done\n');
    choice = input('Choice: ');
    
    % Handle choice
    if choice == 0
        break;  % Done
    elseif choice == 7
        % Undo last fit
        if isempty(history)
            fprintf('Nothing to undo!\n');
            continue;
        end
        last = history(end);
        var_residual = last.var_residual;
        sigma_residual = last.sigma_residual;
        q0 = last.q0;  q1 = last.q1;  q2 = last.q2;  q3 = last.q3;
        sig0ffm = last.sig0ffm;  sig0fpm = last.sig0fpm;
        regions = last.regions;
        history(end) = [];
        fprintf('Undone!\n');
        updatePlot(hFig, tau, sigma, ciLow, ciUp, var_original, var_residual, ...
                   q0, q1, q2, q3, sig0ffm, sig0fpm, 'After Undo');
        continue;
    elseif ~ismember(choice, [1 2 3 4 5 6])
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
                           'sigma_residual', sigma_residual, ...
                           'q0', q0, 'q1', q1, 'q2', q2, 'q3', q3, ...
                           'sig0ffm', sig0ffm, 'sig0fpm', sig0fpm, ...
                           'regions', regions);
    
    % Extract data for fitting
    tau_fit = tau(indices);
    weights_fit = weights(indices);
    
    % Determine noise type
    noiseTypes = {'wpm', 'wfm', 'rwfm', 'rrfm', 'ffm', 'fpm'};
    noiseType = noiseTypes{choice};
    
    % Fit based on choice
    if choice <= 4  % Traditional noise types (WPM, WFM, RWFM, RRFM)
        var_fit = var_residual(indices);
        [q_fit, q_std] = fitFixedSlope(noiseType, tau_fit, var_fit, weights_fit);
        slopes = struct('wpm', -3, 'wfm', -1, 'rwfm', 1, 'rrfm', 3);
        slope_fit = slopes.(noiseType);
        
        % Update residual
        var_residual = updateResidual(noiseType, q_fit, tau, var_residual);
        sigma_residual = sqrt(var_residual);
        
        % Store results
        regions.(noiseType){end+1} = struct('indices', indices, 'q', q_fit, ...
                                            'q_std', q_std, 'slope', slope_fit);
        
        % Update total coefficients
        switch noiseType
            case 'wpm',  q0 = q0 + q_fit;
            case 'wfm',  q1 = q1 + q_fit;
            case 'rwfm', q2 = q2 + q_fit;
            case 'rrfm', q3 = q3 + q_fit;
        end
        
        fprintf('Fitted %s: q = %.3e ± %.3e (slope = %.2f)\n', ...
                upper(noiseType), q_fit, q_std, slope_fit);
                
    else  % Flicker noise types (FFM, FPM)
        sigma_fit = sigma_residual(indices);
        [sig0_fit, sig0_std] = fitFlickerNoise(noiseType, tau_fit, sigma_fit, weights_fit);
        slopes = struct('ffm', 0, 'fpm', -2);
        slope_fit = slopes.(noiseType);
        
        % Update residual (work with sigma directly for flicker)
        sigma_residual = updateFlickerResidual(noiseType, sig0_fit, tau, sigma_residual);
        var_residual = sigma_residual.^2;
        
        % Store results
        regions.(noiseType){end+1} = struct('indices', indices, 'sig0', sig0_fit, ...
                                            'sig0_std', sig0_std, 'slope', slope_fit);
        
        % Update total intercepts
        switch noiseType
            case 'ffm', sig0ffm = sig0ffm + sig0_fit;
            case 'fpm', sig0fpm = sig0fpm + sig0_fit;
        end
        
        fprintf('Fitted %s: σ₀ = %.3e ± %.3e (slope = %.2f)\n', ...
                upper(noiseType), sig0_fit, sig0_std, slope_fit);
    end
    
    % Update display
    showDataTable(tau, var_residual);
    updatePlot(hFig, tau, sigma, ciLow, ciUp, var_original, var_residual, ...
               q0, q1, q2, q3, sig0ffm, sig0fpm, sprintf('After Iteration %d', iter));
end

%% Final results
fprintf('\n=== FINAL RESULTS ===\n');
fprintf('q0 (WPM)  = %.3e\n', q0);
fprintf('q1 (WFM)  = %.3e\n', q1);
fprintf('q2 (RWFM) = %.3e\n', q2);
fprintf('q3 (RRFM) = %.3e\n', q3);
fprintf('σ₀ (FFM)  = %.3e\n', sig0ffm);
fprintf('σ₀ (FPM)  = %.3e\n', sig0fpm);

updatePlot(hFig, tau, sigma, ciLow, ciUp, var_original, var_residual, ...
           q0, q1, q2, q3, sig0ffm, sig0fpm, 'Final Fit');

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
slopes = struct('wpm', -3, 'wfm', -1, 'rwfm', 1, 'rrfm', 3);
coeffs = struct('wpm', 10/3, 'wfm', 7/16, 'rwfm', 1/9, 'rrfm', 11/120);

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

function [sig0, sig0_std] = fitFlickerNoise(noiseType, tau, sigma, weights)
% Fit flicker noise with fixed slope to get intercept

% Define slopes for flicker noise
slopes = struct('ffm', 0, 'fpm', -2);
slope = slopes.(noiseType);

% Model: log10(sigma) = log10(sig0) + slope * log10(tau)
% Rearrange: log10(sigma) - slope * log10(tau) = log10(sig0)

log_tau = log10(tau);
log_sigma = log10(sigma);

% Adjust y values by subtracting the known slope contribution
y_adjusted = log_sigma - slope * log_tau;

% Use weighted mean to get log10(sig0)
[log_sig0, log_std] = weightedMean(y_adjusted, weights);

% Extract sig0
sig0 = 10^log_sig0;

% Uncertainty propagation
sig0_std = sig0 * log(10) * log_std;
end

function var_new = updateResidual(noiseType, q, tau, var_old)
% Subtract fitted component from variance

% Define theoretical parameters
slopes = struct('wpm', -3, 'wfm', -1, 'rwfm', 1, 'rrfm', 3);
coeffs = struct('wpm', 10/3, 'wfm', 7/16, 'rwfm', 1/9, 'rrfm', 11/120);

slope = slopes.(noiseType);
coeff = coeffs.(noiseType);

% Subtract component
var_new = var_old - coeff * q * tau.^slope;
var_new(var_new < 0) = 0;  % Ensure non-negative
end

function sigma_new = updateFlickerResidual(noiseType, sig0, tau, sigma_old)
% Subtract fitted flicker component from sigma

% Define slopes for flicker noise
slopes = struct('ffm', 0, 'fpm', -2);
slope = slopes.(noiseType);

% Model: sigma = sig0 * tau^slope
sigma_component = sig0 * tau.^slope;

% Subtract component (in quadrature for independent noise sources)
sigma_new = sqrt(sigma_old.^2 - sigma_component.^2);
sigma_new(imag(sigma_new) ~= 0) = 0;  % Handle cases where subtraction goes negative
end

function updatePlot(hFig, tau, sigma, ciLow, ciUp, var_orig, var_resid, q0, q1, q2, q3, sig0ffm, sig0fpm, titleStr)
% Update the figure with current state

figure(hFig); clf;

% Main plot - MHDEV with model
subplot(2,2,[1 3]);
loglog(tau, sigma, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 10);
hold on;

% Confidence intervals
fill([tau; flipud(tau)], [ciLow; flipud(ciUp)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Add index labels
addIndexLabels(tau, sigma, 10);

% Plot model if any components fitted
if q0 > 0 || q1 > 0 || q2 > 0 || q3 > 0 || sig0ffm > 0 || sig0fpm > 0
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
        sigma2_wfm = (7/16) * q1 * tau_model.^(-1);
        sigma2_total = sigma2_total + sigma2_wfm;
        h = loglog(tau_model, sqrt(sigma2_wfm), 'g--', 'LineWidth', 1.5);
        h_components(end+1) = h;
        legend_entries{end+1} = sprintf('WFM (q=%.2e)', q1);
    end
    if q2 > 0
        sigma2_rwfm = (1/9) * q2 * tau_model.^(1);
        sigma2_total = sigma2_total + sigma2_rwfm;
        h = loglog(tau_model, sqrt(sigma2_rwfm), 'm--', 'LineWidth', 1.5);
        h_components(end+1) = h;
        legend_entries{end+1} = sprintf('RWFM (q=%.2e)', q2);
    end
    if q3 > 0
        sigma2_rrfm = (11/120) * q3 * tau_model.^(3);
        sigma2_total = sigma2_total + sigma2_rrfm;
        h = loglog(tau_model, sqrt(sigma2_rrfm), 'c--', 'LineWidth', 1.5);
        h_components(end+1) = h;
        legend_entries{end+1} = sprintf('RRFM (q=%.2e)', q3);
    end
    if sig0ffm > 0
        sigma_ffm = sig0ffm * tau_model.^0;
        sigma2_total = sigma2_total + sigma_ffm.^2;
        h = loglog(tau_model, sigma_ffm, 'Color', [1 0.5 0], 'LineStyle', '--', 'LineWidth', 1.5);
        h_components(end+1) = h;
        legend_entries{end+1} = sprintf('FFM (σ₀=%.2e)', sig0ffm);
    end
    if sig0fpm > 0
        sigma_fpm = sig0fpm * tau_model.^(-2);
        sigma2_total = sigma2_total + sigma_fpm.^2;
        h = loglog(tau_model, sigma_fpm, 'Color', [0.5 0 0.5], 'LineStyle', '--', 'LineWidth', 1.5);
        h_components(end+1) = h;
        legend_entries{end+1} = sprintf('FPM (σ₀=%.2e)', sig0fpm);
    end
    
    h = loglog(tau_model, sqrt(sigma2_total), 'k-', 'LineWidth', 2);
    h_components(end+1) = h;
    legend_entries{end+1} = 'Total Model';
end

xlabel('Averaging Time τ [s]');
ylabel('MHDEV σ');
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
text(0.1, 0.75, sprintf('q₀ (WPM) = %.3e', q0), 'FontSize', 12);
text(0.1, 0.65, sprintf('q₁ (WFM) = %.3e', q1), 'FontSize', 12);
text(0.1, 0.55, sprintf('q₂ (RWFM) = %.3e', q2), 'FontSize', 12);
text(0.1, 0.45, sprintf('q₃ (RRFM) = %.3e', q3), 'FontSize', 12);
text(0.1, 0.35, sprintf('σ₀ (FFM) = %.3e', sig0ffm), 'FontSize', 12);
text(0.1, 0.25, sprintf('σ₀ (FPM) = %.3e', sig0fpm), 'FontSize', 12);

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