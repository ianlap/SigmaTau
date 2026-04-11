function [tau, mtotdev, edf, ci, alpha] = mtotdev(x, tau0, m_list)
%MTOTDEV Computes Modified Total Deviation from phase data
%
% [tau, mtotdev, edf, ci, alpha] = allanlab.mtotdev(x, tau0, m_list)
%
% Inputs:
%   x      – Phase data (seconds), row or column vector
%   tau0   – Basic sampling interval (seconds)
%   m_list – Averaging factors (optional), defines τ = m·τ₀
%
% Outputs:
%   tau     – Averaging times τ = m·τ₀ (seconds)
%   mtotdev – Modified total deviation σ_Mtot(τ), unitless
%   edf     – Equivalent degrees of freedom
%   ci      – Confidence interval matrix [ci_lower, ci_upper]
%   alpha   – Noise type exponent

import allanlab.* % noise_id, totaldev_edf, compute_ci, bias_correction

%-- ensure column vector
if isrow(x), x = x.'; end
N = numel(x);

% Warn if estimated runtime is high
warn_threshold = 30; % seconds
est_runtime = 2e-5 * N^1.4;
if est_runtime > warn_threshold
    fprintf(['WARNING: Calculation of MTOTDEV may take a long time (estimated %.1f seconds for N = %d).\n'], est_runtime, N);
end

%-- default m list
if nargin < 3 || isempty(m_list)
    m_list = 2.^(0:floor(log2(N/3)));
end

%-- initialize outputs
tau = m_list * tau0;
mtotdev = NaN(size(m_list));
Mvar = NaN(size(m_list));
edf = NaN(size(m_list));
ci = NaN(numel(m_list), 2);
alpha = NaN(size(m_list));
N_eff = NaN(size(m_list)); % Effective number of samples for CI

%-- compute MTOTVAR
for k = 1:numel(m_list)
    m = m_list(k);
    nsubs = N - 3*m + 1;
    N_eff(k) = nsubs;
    
    if nsubs < 1, continue; end
    
    outer_sum = 0;
    
    for n = 1:nsubs
        % Extract 3m phase points
        seq = x(n : n + 3*m - 1);
        half_n = 3*m/2;
        
        % Detrend using half-average method
        if m == 1
            first_half = seq(1);
            last_half = seq(3);
            slope = (last_half - first_half) / (2 * tau0);
        else
            first_half = mean(seq(1:floor(half_n)));
            last_half = mean(seq(floor(half_n)+1:end));
            slope = (last_half - first_half) / (half_n * tau0);
        end
        
        % Remove linear trend
        seq_detrended = seq - slope * tau0 * (0:3*m-1)';
        
        % Extend by uninverted even reflection
        ext = [seq_detrended(end:-1:1); seq_detrended; seq_detrended(end:-1:1)];
        
        % Calculate second differences using cumsum
        cs = cumsum([0; ext]);
        avg1 = (cs((1:6*m) + m) - cs(1:6*m)) / m;
        avg2 = (cs((1:6*m) + 2*m) - cs((1:6*m) + m)) / m;
        avg3 = (cs((1:6*m) + 3*m) - cs((1:6*m) + 2*m)) / m;
        
        % Second differences
        d2 = avg3 - 2*avg2 + avg1;
        
        % Accumulate variance
        outer_sum = outer_sum + sum(d2.^2) / (6 * m);
    end
    
    % Normalize for Modified Total variance
    Mvar(k) = outer_sum / (2 * (m * tau0)^2 * nsubs);
end

%-- trim invalid values
valid = ~isnan(Mvar);
tau = tau(valid);
Mvar = Mvar(valid);
m_list = m_list(valid);
N_eff = N_eff(valid);

%-- estimate noise type
alpha = noise_id(x, m_list, 'phase', 0, 2);

%-- calculate EDF
T = N * tau0;
for k = 1:numel(m_list)
    try
        edf(k) = totaldev_edf('mtot', alpha(k), T, tau(k));
    catch
        edf(k) = NaN;
    end
end

%-- convert variance to deviation
mtotdev = sqrt(Mvar);

%-- compute confidence intervals
ci = compute_ci(mtotdev, edf, 0.683, alpha, N_eff);

end