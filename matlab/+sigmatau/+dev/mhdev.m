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
    'dmax',       2        ...
);

result = sigmatau.dev.engine(x, tau0, m_list, @mhdev_kernel, params, varargin{:});
end

function [v, neff] = mhdev_kernel(x, m, tau0)
% Third differences + moving average via cumsum trick.
N  = numel(x);
Ne = N - 4*m + 1;
if Ne <= 0
    v = NaN; neff = 0;
    return;
end
% Third differences of the phase data
d3 = x(1:Ne) - 3*x(1+m:Ne+m) + 3*x(1+2*m:Ne+2*m) - x(1+3*m:Ne+3*m);
% Moving average via cumsum (length-m windows over d3)
S   = cumsum([0; d3]);
avg = S(m+1:end) - S(1:end-m);   % length Ne+1-m
v    = sum(avg.^2) / (numel(avg) * 6 * m^4 * tau0^2);
neff = Ne;
end
