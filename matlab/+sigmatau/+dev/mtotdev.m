function result = mtotdev(x, tau0, m_list, varargin)
% MTOTDEV  Modified total deviation.
%
%   result = sigmatau.dev.mtotdev(x, tau0)
%   result = sigmatau.dev.mtotdev(x, tau0, m_list)
%   result = sigmatau.dev.mtotdev(x, tau0, m_list, 'data_type', 'freq')
%
%   For each N-3m+1 subsegment of length 3m: half-average detrend,
%   symmetric reflection, modified ADEV (cumsum second differences).
%   SP1065 §5.12.

if nargin < 3, m_list = []; end

params = struct( ...
    'name',       'mtotdev', ...
    'min_factor', 3,         ...
    'd',          2,         ...
    'F_fn',       @(m) 1,    ...
    'dmin',       0,         ...
    'dmax',       2,         ...
    'total_type', 'mtot',    ...
    'bias_type',  ''         ...
);

result = sigmatau.dev.engine(x, tau0, m_list, @mtotdev_kernel, params, varargin{:});
end

function [v, neff] = mtotdev_kernel(x, m, tau0)
% Accumulate variance over all N-3m+1 subsegments.
% Each segment: half-average detrend → symmetric reflection → cumsum second-diff.
N     = numel(x);
nsubs = N - 3*m + 1;
if nsubs < 1
    v = NaN; neff = 0;
    return;
end

seg_len   = 3*m;
half_n    = seg_len / 2;
outer_sum = 0.0;

CX      = cumsum([0; x]);
p_range = (0:seg_len)';
T2      = p_range .* (p_range - 1) / 2;
j_range = (0:3*m)';

for n = 1:nsubs
    % Half-average detrend without slicing
    if m == 1
        slope = (x(n+2) - x(n)) / (2*tau0);
    else
        hi = floor(half_n);
        s1 = (CX(n+hi) - CX(n)) / hi;
        s2 = (CX(n+seg_len) - CX(n+hi)) / (seg_len - hi);
        slope = (s2 - s1) / (half_n * tau0);
    end

    % SumS_vec(p+1) = sum_{i=1}^p (x(n+i-1) - slope*tau0*(i-1))
    SumS_vec = (CX(n+p_range) - CX(n)) - slope * tau0 * T2;

    % Reflection: cs(1:3m+1) is rev(seq_det), cs(3m+1:6m+1) is seq_det
    % cs(k+1) = SumS(seg_len) - SumS(seg_len-k) for k=0:seg_len
    % cs(k+1) = SumS(seg_len) + SumS(k-seg_len) for k=seg_len:2*seg_len
    cs = [SumS_vec(end) - SumS_vec(end:-1:2); SumS_vec(end) + SumS_vec];

    % d2 calculation using cs (length 6m+1)
    a1 = (cs(j_range + m + 1)   - cs(j_range + 1))       / m;
    a2 = (cs(j_range + 2*m + 1) - cs(j_range + m + 1))   / m;
    a3 = (cs(j_range + 3*m + 1) - cs(j_range + 2*m + 1)) / m;
    d2 = a3 - 2*a2 + a1;

    outer_sum = outer_sum + sum(d2.^2) / (6*m);
end

v    = outer_sum / (2 * (m*tau0)^2 * nsubs);
neff = nsubs;
end

