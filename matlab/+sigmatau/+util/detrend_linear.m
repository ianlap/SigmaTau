function x = detrend_linear(x)
% DETREND_LINEAR  Remove linear trend via least-squares fit.
%
%   x = sigmatau.util.detrend_linear(x)
%
%   Equivalent to Julia's detrend_linear: fits [1, t] and subtracts.
%   Does NOT use MATLAB's built-in detrend() to avoid toolbox dependencies
%   and ensure numerical equivalence with the Julia implementation.

x = x(:);
n = numel(x);
if n < 2
    return;
end
A = [ones(n,1), (1:n)'];
x = x - A * (A \ x);
end
