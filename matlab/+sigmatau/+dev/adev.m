function result = adev(x, tau0, m_list, varargin)
% ADEV  Overlapping Allan deviation (OADEV).
%
%   result = sigmatau.dev.adev(x, tau0)
%   result = sigmatau.dev.adev(x, tau0, m_list)
%   result = sigmatau.dev.adev(x, tau0, m_list, 'data_type', 'freq')
%
%   SP1065 Eq. 14: AVAR(tau) = mean((x(n+2m) - 2x(n+m) + x(n))^2) / (2*m^2*tau0^2)
%
%   Inputs:
%     x      – phase data (or freq data with data_type='freq')
%     tau0   – sampling interval (s)
%     m_list – averaging factors ([] for auto-generated octave-spaced)
%
%   Output:
%     result – struct: tau, deviation, edf, ci, alpha, neff, tau0, N, method, confidence

if nargin < 3, m_list = []; end

params = struct( ...
    'name',       'adev', ...
    'min_factor', 2,      ...
    'd',          2,      ...
    'F_fn',       @(m) m, ...
    'dmin',       0,      ...
    'dmax',       2,      ...
    'total_type', '',     ...
    'bias_type',  ''      ...
);

result = sigmatau.dev.engine(x, tau0, m_list, @sigmatau.dev.adev_kernel, params, varargin{:});
end
