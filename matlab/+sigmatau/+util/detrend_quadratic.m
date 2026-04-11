function x = detrend_quadratic(x)
% DETREND_QUADRATIC  Remove quadratic trend via least-squares fit.
%
%   x = sigmatau.util.detrend_quadratic(x)
%
%   Fits [1, t, t^2] and subtracts. Used by noise_id for phase data.
%   Does NOT use MATLAB's built-in detrend() to ensure equivalence with
%   the Julia implementation (validate.jl detrend_quadratic).

x = x(:);
n = numel(x);
if n < 3
    return;
end
t = (1:n)';
A = [ones(n,1), t, t.^2];
x = x - A * (A \ x);
end
