function weights = ci2weights(sigma, ci_lower, ci_upper, method)
% CI2WEIGHTS - Convert confidence intervals to weights for weighted least squares
%
% Transforms confidence interval bounds into weights suitable for WLS fitting
% in Allan variance analysis. Supports multiple weighting schemes based on
% how confidence intervals are interpreted.
%
% Syntax:
%   weights = ci2weights(sigma, ci_lower, ci_upper, method)
%   weights = ci2weights(sigma, ci_lower, ci_upper)  % Default: 'symmetric'
%
% Inputs:
%   sigma    - Central values (typically Allan deviation values)
%   ci_lower - Lower confidence interval bounds
%   ci_upper - Upper confidence interval bounds  
%   method   - Weighting method (optional):
%              'symmetric'     - Use average interval width (default)
%              'conservative'  - Use larger of upper/lower uncertainty
%              'inverse'       - Simple inverse of total interval width
%
% Outputs:
%   weights - Weights for WLS fitting (normalized, non-negative)
%
% Example:
%   weights = ci2weights(adev, adev_low, adev_high, 'conservative');
%   [slope, intercept] = weightedLinearFit(log10(tau), log10(adev), weights);
%
% See also: WEIGHTEDLINEARFIT, MHTOT_FIT

%% Set default method
if nargin < 4
    method = 'symmetric';
end

%% Calculate weights based on selected method
switch method
    case 'symmetric'
        % Average half-width of confidence interval
        % Appropriate when CI is approximately symmetric
        ci_width = (ci_upper - ci_lower) / 2;
        weights = 1 ./ ci_width.^2;
        
    case 'conservative'
        % Use the larger uncertainty (more conservative approach)
        % Appropriate when CI is asymmetric or when being cautious
        ci_width = max(sigma - ci_lower, ci_upper - sigma);
        weights = 1 ./ ci_width.^2;
        
    case 'inverse'
        % Simple inverse of total interval width
        % Less statistically rigorous but sometimes used
        ci_width = ci_upper - ci_lower;
        weights = 1 ./ ci_width;  % Note: linear, not squared
        
    otherwise
        error('CI2WEIGHTS:UnknownMethod', ...
              'Unknown method "%s". Use "symmetric", "conservative", or "inverse".', method);
end

%% Handle edge cases
% Replace infinities and NaNs with zero weight
% These can occur if CI width is zero (perfect certainty - unlikely in practice)
weights(~isfinite(weights)) = 0;

% Note: Normalization removed as it can affect optimization scaling
% If needed, normalization should be done at the fitting stage

end