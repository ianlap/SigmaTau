function [wmean, wstd, wvar] = weightedMean(x, weights)
% WEIGHTEDMEAN - Calculate weighted mean and uncertainty
%
% Computes the weighted mean of data points along with the weighted
% standard deviation and variance of the mean.
%
% Syntax:
%   [wmean, wstd, wvar] = weightedMean(x, weights)
%   [wmean, wstd] = weightedMean(x, weights)
%   wmean = weightedMean(x, weights)
%
% Inputs:
%   x       - Data values (vector)
%   weights - Weight for each data point (vector, same size as x)
%
% Outputs:
%   wmean - Weighted mean
%   wstd  - Weighted standard deviation of the mean
%   wvar  - Weighted variance of the mean
%
% Notes:
%   - Uses reliability weights (inverse variance weights)
%   - Handles the effective sample size for proper uncertainty estimation
%   - Weights should be non-negative; zero weights exclude points
%
% Example:
%   x = [1.2, 1.5, 1.3, 1.4];
%   w = [1.0, 0.5, 2.0, 1.5];  % More weight on 3rd measurement
%   [mean_val, std_val] = weightedMean(x, w);
%
% See also: MEAN, STD, WEIGHTEDLINEARFIT

%% Input validation
x = x(:);        % Ensure column vector
weights = weights(:);

if length(x) ~= length(weights)
    error('WEIGHTEDMEAN:SizeMismatch', 'x and weights must have the same length');
end

% Remove any NaN or Inf values
valid = isfinite(x) & isfinite(weights) & (weights >= 0);
x = x(valid);
weights = weights(valid);

if isempty(x)
    error('WEIGHTEDMEAN:NoValidData', 'No valid data points after filtering');
end

if sum(weights) == 0
    error('WEIGHTEDMEAN:ZeroWeights', 'Sum of weights is zero');
end

%% Calculate weighted mean
wmean = sum(weights .* x) / sum(weights);

%% Calculate uncertainty if requested
if nargout > 1
    % Weighted variance of the data
    residuals = x - wmean;
    data_variance = sum(weights .* residuals.^2) / sum(weights);
    
    % Effective sample size (accounts for unequal weights)
    % n_eff = (sum(w))^2 / sum(w^2)
    n_eff = sum(weights)^2 / sum(weights.^2);
    
    % Variance of the weighted mean
    wvar = data_variance / n_eff;
    
    % Standard deviation of the weighted mean
    wstd = sqrt(wvar);
end

end