function result = mhtotdev(x, tau0, m_list, varargin)
% MHTOTDEV  Modified Hadamard total deviation.
%
%   result = sigmatau.dev.mhtotdev(x, tau0)
%   result = sigmatau.dev.mhtotdev(x, tau0, m_list)
%   result = sigmatau.dev.mhtotdev(x, tau0, m_list, 'data_type', 'freq')
%
%   For each N-4m+1 subsegment of phase length 3m+1: linear detrend,
%   symmetric reflection, third differences, moving average. SP1065/FCS 2001.

if nargin < 3, m_list = []; end

params = struct( ...
    'name',       'mhtotdev', ...
    'min_factor', 4,          ...
    'd',          3,          ...
    'F_fn',       @(m) 1,     ...
    'dmin',       0,          ...
    'dmax',       2           ...
);

result = sigmatau.dev.engine(x, tau0, m_list, @mhtotdev_kernel, params, varargin{:});
end

function [v, neff] = mhtotdev_kernel(x, m, tau0)
% Linear detrend per phase segment, symmetric reflection, third diffs + moving avg.
N     = numel(x);
nsubs = N - 4*m + 1;
if nsubs < 1
    v = NaN; neff = 0;
    return;
end

Lp        = 3*m + 1;   % phase segment length
total_sum = 0.0;

for n = 1:nsubs
    phase_seg = x(n : n + 3*m);
    pd = sigmatau.util.detrend(phase_seg, 1);   % length Lp

    % Symmetric reflection: [rev(pd); pd; rev(pd)]
    ext = [pd(end:-1:1); pd; pd(end:-1:1)];

    % Third differences on extended array
    L3 = numel(ext) - 3*m;
    if L3 <= 0, continue; end
    d3_vec = ext(1:L3) - 3*ext(1+m:L3+m) + 3*ext(1+2*m:L3+2*m) - ext(1+3*m:L3+3*m);

    % Moving average via cumsum (length-m windows)
    if numel(d3_vec) >= m
        S     = cumsum([0; d3_vec]);
        n_avg = numel(S) - m;
        avg   = S(m+1:end) - S(1:end-m);
        block_var = sum(avg.^2) / (n_avg * 6 * m^2);
    else
        block_var = 0;
    end

    total_sum = total_sum + block_var;
end

v    = total_sum / (nsubs * (m*tau0)^2);
neff = nsubs;
end
