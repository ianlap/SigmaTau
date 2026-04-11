function [slope, intercept, cov_matrix] = weightedLinearFit(x, y, weights)
% WEIGHTEDLINEARFIT - Weighted least squares linear fit
%
% Performs weighted least squares regression in log-log space for
% Allan variance analysis. Fits the model: y = intercept + slope*x
% where x and y are already in log10 space.
%
% Syntax:
%   [slope, intercept, cov_matrix] = weightedLinearFit(x, y, weights)
%
% Inputs:
%   x       - Independent variable data (already in log10 space)
%   y       - Dependent variable data (already in log10 space)
%   weights - Weight for each data point (typically from confidence intervals)
%
% Outputs:
%   slope      - Fitted slope in log-log space
%   intercept  - Fitted intercept in log-log space
%   cov_matrix - 2×2 parameter covariance matrix (X'WX)^(-1)
%
% Example:
%   log_tau = log10(tau_values);
%   log_sigma = log10(sigma_values);
%   weights = 1./confidence_widths.^2;
%   [slope, intercept, cov] = weightedLinearFit(log_tau, log_sigma, weights);
%
% See also: MHTOT_FIT, CI2WEIGHTS

%% Ensure column vectors
x = x(:);
y = y(:);
weights = weights(:);

%% Build design matrix for linear model
% X = [1 x] for model: y = intercept + slope*x
X = [ones(length(x), 1), x];

%% Construct weight matrix
W = diag(weights);

%% Weighted least squares solution
% Normal equations: beta = (X'WX)^(-1)X'Wy
XtWX = X' * W * X;  % Weighted gram matrix
XtWy = X' * W * y;  % Weighted moments

% Solve using backslash for numerical stability
% Avoids explicit matrix inversion
params = XtWX \ XtWy;
intercept = params(1);
slope = params(2);

%% Parameter covariance matrix
% Covariance of estimates: Cov(beta) = (X'WX)^(-1)
% Required for uncertainty propagation in noise fitting
cov_matrix = inv(XtWX);  % Explicit inverse needed here

end