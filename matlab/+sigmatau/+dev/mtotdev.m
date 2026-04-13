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
    'dmax',       2          ...
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

for n = 1:nsubs
    seq = x(n : n + seg_len - 1);

    % Half-average detrend (matches Julia _mtotdev_kernel)
    if m == 1
        slope = (seq(3) - seq(1)) / (2*tau0);
    else
        hi = floor(half_n);
        s1 = sum(seq(1:hi)) / hi;
        s2 = sum(seq(hi+1:seg_len)) / (seg_len - hi);
        slope = (s2 - s1) / (half_n * tau0);
    end
    seq_det = seq - slope * tau0 * (0:seg_len-1)';

    % Symmetric reflection: [rev(seq_det); seq_det; rev(seq_det)]
    ext = [seq_det(end:-1:1); seq_det; seq_det(end:-1:1)];

    % Cumsum of extended sequence. Loop range 0:3m (3m+1 positions) matches
    % Julia _mtotdev_kernel: for j in 0:(6m - 3m) == 0:3m
    cs = cumsum([0; ext]);
    j_range = (0:3*m)';       % 3m+1 indices (0-based)
    j1 = j_range + 1;         % 1-based index into cs (cs(j+1) in Julia)
    a1 = (cs(j1 + m)   - cs(j1))     / m;
    a2 = (cs(j1 + 2*m) - cs(j1 + m)) / m;
    a3 = (cs(j1 + 3*m) - cs(j1 + 2*m)) / m;
    d2 = a3 - 2*a2 + a1;

    outer_sum = outer_sum + sum(d2.^2) / (6*m);
end

v    = outer_sum / (2 * (m*tau0)^2 * nsubs);
neff = nsubs;
end
