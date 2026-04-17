function result = htotdev(x, tau0, m_list, varargin)
% HTOTDEV  Hadamard total deviation.
%
%   result = sigmatau.dev.htotdev(x, tau0)
%   result = sigmatau.dev.htotdev(x, tau0, m_list)
%   result = sigmatau.dev.htotdev(x, tau0, m_list, 'data_type', 'freq')
%
%   Uses frequency data segments with half-average detrend + symmetric
%   reflection + Hadamard cumsum differences. SP1065 §5.13.
%
%   CRITICAL: when m==1, uses overlapping HDEV (third differences on phase)
%   instead of the total deviation algorithm. (CLAUDE.md critical rule.)

if nargin < 3, m_list = []; end

params = struct( ...
    'name',       'htotdev', ...
    'min_factor', 3,         ...
    'd',          3,         ...
    'F_fn',       @(m) m,    ...
    'dmin',       0,         ...
    'dmax',       2,         ...
    'total_type', 'htot',    ...
    'bias_type',  'htot'     ...
);

result = sigmatau.dev.engine(x, tau0, m_list, @htotdev_kernel, params, varargin{:});
end

function [v, neff] = htotdev_kernel(x, m, tau0, ~)
% m==1: HDEV third differences on phase (CLAUDE.md critical rule).
% m>1:  frequency segment algorithm with half-average detrend + reflection.
N = numel(x);

if m == 1
    % Use hdev formula directly — CLAUDE.md critical rule
    L = N - 3;
    if L <= 0
        v = NaN; neff = 0;
        return;
    end
    d3   = x(4:N) - 3*x(3:N-1) + 3*x(2:N-2) - x(1:L);
    v    = sum(d3.^2) / (L * 6 * tau0^2);
    neff = L;
    return;
end

y      = diff(x) / tau0;   % fractional frequency data
Ny     = numel(y);
n_iter = Ny - 3*m + 1;
if n_iter < 1
    v = NaN; neff = 0;
    return;
end

seg_len = 3*m;
dev_sum = 0.0;

CY      = cumsum([0; y]);
p_range = (0:seg_len)';
mid     = floor(seg_len / 2);
T_H     = p_range .* (p_range - 1) / 2 - p_range * mid;
j_range = (0:6*m-1)';

for i = 0:(n_iter - 1)
    n = i + 1;
    % Half-average detrend on frequency segment
    hi       = floor(seg_len / 2);
    lo_start = ceil(seg_len / 2) + 1;
    m1 = (CY(n+hi) - CY(n)) / hi;
    m2 = (CY(n+seg_len) - CY(n+lo_start-1)) / (seg_len - lo_start + 1);

    if mod(seg_len, 2) == 1   % seg_len is odd
        slope = (m2 - m1) / (0.5*(seg_len-1) + 1);
    else
        slope = (m2 - m1) / (0.5*seg_len);
    end

    % SumX0_vec(p+1) = sum_{k=1}^p (y(n+k-1) - slope * (k-1-mid))
    SumX0_vec = (CY(n+p_range) - CY(n)) - slope * T_H;

    % Reflection: cs has 3 parts [rev(x0); x0; rev(x0)]
    % SR(p+1) = cumsum([0; rev(x0)], p)
    S_end = SumX0_vec(end);
    SR = S_end - SumX0_vec(end:-1:1);
    cs = [SR; S_end + SumX0_vec(2:end); 2*S_end + SR(2:end)];

    % Hadamard cumsum differences
    h1 = (cs(j_range+m+1)   - cs(j_range+1))    / m;
    h2 = (cs(j_range+2*m+1) - cs(j_range+m+1))  / m;
    h3 = (cs(j_range+3*m+1) - cs(j_range+2*m+1)) / m;
    H  = h3 - 2*h2 + h1;

    dev_sum = dev_sum + sum(H.^2) / (6*m);
end

v    = dev_sum / (6 * n_iter);
neff = n_iter;
end
