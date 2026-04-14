function result = hdev(x, tau0, m_list, varargin)
% HDEV  Overlapping Hadamard deviation (OHDEV).
%
%   result = sigmatau.dev.hdev(x, tau0)
%   result = sigmatau.dev.hdev(x, tau0, m_list)
%   result = sigmatau.dev.hdev(x, tau0, m_list, 'data_type', 'freq')
%
%   SP1065 HVAR: HVAR(tau) = mean(d3^2) / (6*m^2*tau0^2)
%   Third differences suppress linear frequency drift.

if nargin < 3, m_list = []; end

params = struct( ...
    'name',       'hdev', ...
    'min_factor', 4,      ...
    'd',          3,      ...
    'F_fn',       @(m) m, ...
    'dmin',       0,      ...
    'dmax',       2,      ...
    'total_type', '',     ...
    'bias_type',  ''      ...
);

result = sigmatau.dev.engine(x, tau0, m_list, @hdev_kernel, params, varargin{:});
end

function [v, neff] = hdev_kernel(x, m, tau0)
% SP1065 HVAR: HVAR(tau) = mean(d3^2) / (6*m^2*tau0^2)
N = numel(x);
L = N - 3*m;
if L <= 0
    v = NaN; neff = 0;
    return;
end
d3   = x(1+3*m:N) - 3*x(1+2*m:N-m) + 3*x(1+m:N-2*m) - x(1:L);
v    = sum(d3.^2) / (L * 6 * m^2 * tau0^2);
neff = L;
end
