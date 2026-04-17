function result = mhdev(x, tau0, m_list, varargin)
% MHDEV  Modified Hadamard deviation (MHDEV).
%
%   result = sigmatau.dev.mhdev(x, tau0)
%   result = sigmatau.dev.mhdev(x, tau0, m_list)
%   result = sigmatau.dev.mhdev(x, tau0, m_list, 'data_type', 'freq')
%
%   Third differences with moving average (cumsum trick).
%   Analogous to MDEV vs ADEV.

if nargin < 3, m_list = []; end

params = struct( ...
    'name',       'mhdev', ...
    'min_factor', 4,       ...
    'd',          3,       ...
    'F_fn',       @(m) 1,  ...
    'dmin',       0,       ...
    'dmax',       2,       ...
    'total_type', '',      ...
    'bias_type',  ''       ...
);

result = sigmatau.dev.engine(x, tau0, m_list, @mhdev_kernel, params, varargin{:});
end

function [v, neff] = mhdev_kernel(x, m, tau0, x_cs)
% Optimized MHDEV kernel using 4th-difference identity on prefix sums.
% Identity: s4 - 3s3 + 3s2 - s1 = x_cs(i+4m) - 4x_cs(i+3m) + 6x_cs(i+2m) - 4x_cs(i+m) + x_cs(i)
N  = numel(x);
Ne = N - 4*m + 1;
if Ne <= 0
    v = NaN; neff = 0;
    return;
end
d  = x_cs(1+4*m:Ne+4*m) - 4*x_cs(1+3*m:Ne+3*m) + 6*x_cs(1+2*m:Ne+2*m) - 4*x_cs(1+m:Ne+m) + x_cs(1:Ne);
v  = sum(d.^2) / (Ne * 6 * m^4 * tau0^2);
neff = Ne;
end
