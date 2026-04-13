function x = detrend(x, degree)
% DETREND  Remove polynomial trend via least-squares fit.
%
%   x = sigmatau.util.detrend(x, degree)
%
%   degree = 1 → subtract [1, t] fit (linear)
%   degree = 2 → subtract [1, t, t^2] fit (quadratic)
%
%   Does NOT use MATLAB's built-in detrend() to avoid toolbox dependencies
%   and to match the Julia implementation (validate.jl).

x = x(:);
n = numel(x);
if n < degree + 1
    return;
end
t = (1:n)';
A = ones(n, degree + 1);
for k = 1:degree
    A(:, k + 1) = t.^k;
end
x = x - A * (A \ x);
end
