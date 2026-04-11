function [tau, htotdev, edf, ci, alpha] = htotdev(x, tau0, m_list)
%HTOTDEV_ALLANTOOLS_EXACT Hadamard total deviation matching allantools/Stable32
%
% This implementation matches allantools/Stable32 results exactly.
%
% Key changes from original implementation:
% - Uses SP1065 detrending method (half-averages) instead of polyfit
% - Fixed indexing for second half mean calculation (half2_idx+1)
% - Applies bias correction at the end
%
% [tau, htotdev, edf, ci, alpha] = allanlab.htotdev_allantools_exact(x, tau0, m_list)
%
% Inputs:
%   x      – Phase data (seconds), row or column vector  
%   tau0   – Basic sampling interval (seconds)
%   m_list – Averaging factors (optional), defines τ = m·τ₀
%
% Outputs:
%   tau     – Averaging times τ = m·τ₀ (seconds)
%   htotdev – Hadamard total deviation σ_Htot(τ), dimensionless
%   edf     – Equivalent degrees of freedom
%   ci      – Confidence interval matrix [ci_lower, ci_upper]
%   alpha   – Noise type exponent

import allanlab.* % noise_id, totaldev_edf, bias_correction, compute_ci, hdev

%-- ensure column vector
if isrow(x), x = x.'; end
N = numel(x);

%-- convert phase to fractional frequency
y = diff(x) ./ tau0;
Ny = numel(y);

%-- default m list
if nargin < 3 || isempty(m_list)
    m_list = 2.^(0:floor(log2(Ny/3)));
end

%-- initialize
tau = m_list * tau0;
htotdev = NaN(size(tau));
edf = NaN(size(tau));
ci = NaN(numel(tau), 2);
htotvar = NaN(size(tau));

%-- estimate alpha for bias correction
alpha = noise_id(x, m_list, 'phase', 0, 2);

%-- compute HTOTVAR
text_progress(0, 'Computing HTOTDEV');
for idx = 1:numel(m_list)
    m = m_list(idx);
    
    % Special case: m=1 uses overlapping HDEV
    if m == 1
        [~, hdev_val, ~, ~, ~] = hdev(x, tau0, 1);
        htotdev(idx) = hdev_val;
        continue;
    end
    
    % Number of subsequences
    n_iterations = Ny - 3*m + 1;
    if n_iterations < 1, continue, end
    
    % Accumulator for variance
    dev_sum = 0;
    % Loop over subsequences
    for i = 0:(n_iterations-1)
        % Extract 3m points
        xs = y(i+1 : i+3*m);
        
        % Remove linear trend using half-average method
        half1_idx = floor(3*m/2);
        half2_idx = ceil(3*m/2);
        
        % Calculate means of first and second halves
        mean1 = mean(xs(1:half1_idx));
        mean2 = mean(xs(half2_idx+1:end));  % Fixed: added +1
        
        % Calculate slope based on odd/even
        if mod(3*m, 2) == 1  % 3m is odd
            slope = (mean2 - mean1) / (0.5*(3*m-1) + 1);
        else  % 3m is even
            slope = (mean2 - mean1) / (0.5*3*m);
        end
        
        % Detrend the sequence
        x0 = zeros(size(xs));
        for j = 0:(length(xs)-1)
            x0(j+1) = xs(j+1) - slope * (j - floor(3*m/2));
        end
        
        % Extend by uninverted even reflection
        xstar = [x0(end:-1:1); x0; x0(end:-1:1)];
        
        % Calculate Hadamard differences using cumsum for efficiency
        cs = [0; cumsum(xstar)];
        j_indices = 0:(6*m-1);
        
        % Calculate three m-point window sums
        sum1 = cs(j_indices+m+1) - cs(j_indices+1);
        sum2 = cs(j_indices+2*m+1) - cs(j_indices+m+1);
        sum3 = cs(j_indices+3*m+1) - cs(j_indices+2*m+1);
        
        % Convert to means
        xmean1 = sum1 / m;
        xmean2 = sum2 / m;
        xmean3 = sum3 / m;
        
        % Hadamard differences
        H = xmean3 - 2*xmean2 + xmean1;
        
        % Sum of squares normalized by 6m
        squaresum = sum(H.^2) / (6*m);
        dev_sum = dev_sum + squaresum;
        if mod(i, max(1, floor((n_iterations-1)/10))) == 0
            text_progress(idx/numel(m_list));
        end
    end
    
    % Final normalization per equation (29): divide by 6*(N-3m+1)
    htotvar(idx) = dev_sum / (6 * n_iterations);
    htotdev(idx) = sqrt(htotvar(idx));
end

text_progress(1);  % Complete progress

%-- trim invalid
valid = ~isnan(htotdev);
tau = tau(valid);
htotdev = htotdev(valid);
m_list = m_list(valid);
alpha = alpha(valid);

%-- EDF calculation
T = N * tau0;
edf = NaN(size(tau));
for k = 1:numel(valid)
    if valid(k)
        try
            edf(k) = totaldev_edf('htot', alpha(k), T, tau(k));
        catch
            edf(k) = NaN;
        end
    end
end

%-- Apply bias correction
% Calculate bias correction factors for each tau
T = N * tau0;  % Total observation time
B = bias_correction(alpha, 'htot', tau, T);

% Apply bias correction to htotdev
for k = 1:numel(htotdev)
    if m_list(k) ~= 1  % Skip m=1 since it uses HDEV
        htotdev(k) = htotdev(k) * sqrt(B(k));
    end
end

%-- Confidence intervals
Neff = Ny - 3*m_list(valid) + 1;
ci = compute_ci(htotdev, edf, 0.683, alpha, Neff);

end