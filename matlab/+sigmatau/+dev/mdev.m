function result = mdev(x, tau0, m_list, varargin)
% MDEV  Modified Allan deviation (MDEV).
%
%   result = sigmatau.dev.mdev(x, tau0)
%   result = sigmatau.dev.mdev(x, tau0, m_list)
%   result = sigmatau.dev.mdev(x, tau0, m_list, 'data_type', 'freq')
%
%   Uses cumsum prefix sums for O(N) computation per m.
%   SP1065 Eq. 15: MVAR(tau) = sum((s3 - 2s2 + s1)^2) / (Ne * 2 * m^2 * tau0^2)
%
%   Output: result struct (see sigmatau.dev.engine)

if nargin < 3, m_list = []; end

params = struct( ...
    'name',       'mdev', ...
    'min_factor', 3,      ...
    'd',          2,      ...
    'F_fn',       @(m) 1, ...
    'dmin',       0,      ...
    'dmax',       2,      ...
    'total_type', '',     ...
    'bias_type',  ''      ...
);

result = sigmatau.dev.engine(x, tau0, m_list, @mdev_kernel, params, varargin{:});
end

function [v, neff] = mdev_kernel(x, m, tau0, x_cs)
% MDEV_KERNEL  Modified Allan variance kernel for sigmatau.dev.engine.
%
%   Uses cumsum prefix sums for O(N) computation per m.
%   SP1065 Eq. 15: MVAR(tau) = sum((s3 - 2s2 + s1)^2) / (Ne * 2 * m^2 * tau0^2)
N  = numel(x);
Ne = N - 3*m + 1;
if Ne <= 0
    v = NaN; neff = 0;
    return;
end
s1 = x_cs(1+m:Ne+m)     - x_cs(1:Ne);
s2 = x_cs(1+2*m:Ne+2*m) - x_cs(1+m:Ne+m);
s3 = x_cs(1+3*m:Ne+3*m) - x_cs(1+2*m:Ne+2*m);
d  = (s3 - 2*s2 + s1);
v  = sum(d.^2) / (Ne * 2 * m^4 * tau0^2);
neff = Ne;
end
