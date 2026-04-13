function result = totdev(x, tau0, m_list, varargin)
% TOTDEV  Total deviation.
%
%   result = sigmatau.dev.totdev(x, tau0)
%   result = sigmatau.dev.totdev(x, tau0, m_list)
%   result = sigmatau.dev.totdev(x, tau0, m_list, 'data_type', 'freq')
%
%   Extends data by symmetric reflection to reduce endpoint effects, then
%   computes overlapping second differences. SP1065 §5.11.
%
%   Algorithm: linear detrend, build 3N-4 extended array by symmetric
%   reflection about each endpoint, compute second differences at all N
%   center positions.

if nargin < 3, m_list = []; end

params = struct( ...
    'name',       'totdev', ...
    'min_factor', 2,        ...
    'd',          2,        ...
    'F_fn',       @(m) m,   ...
    'dmin',       0,        ...
    'dmax',       2,        ...
    'is_total',   true,     ...
    'total_type', 'totvar', ...
    'needs_bias', true,     ...
    'bias_type',  'totvar'  ...
);

result = sigmatau.dev.engine(x, tau0, m_list, @totdev_kernel, params, varargin{:});
end

function [v, neff] = totdev_kernel(x, m, tau0)
% Linear detrend + symmetric reflection, then overlapping second differences.
% SP1065 §5.11: denominator uses (N-2).
N  = numel(x);
xd = sigmatau.util.detrend(x, 1);
% Symmetric reflection about each endpoint
x_left  = 2*xd(1)   - xd(2:N-1);           % length N-2
x_right = 2*xd(end) - xd(N-1:-1:2);         % length N-2
x_star  = [x_left; xd; x_right];            % length 3N-4
off     = N - 2;   % x_star(off+i) == xd(i) for i=1:N

D     = 0;
count = 0;
for i = 1:N
    lo = off + i;
    hi = off + i + 2*m;
    if hi > numel(x_star), continue; end
    d2 = x_star(hi) - 2*x_star(off + i + m) + x_star(lo);
    D  = D + d2^2;
    count = count + 1;
end
if count == 0
    v = NaN; neff = 0;
    return;
end
% SP1065 denominator uses (N-2), not count
v    = D / (2 * (N-2) * (m*tau0)^2);
neff = count;
end
