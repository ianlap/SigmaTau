function result = totdev(x, tau0, m_list, varargin)
% TOTDEV  Total deviation.
%
%   result = sigmatau.dev.totdev(x, tau0)
%   result = sigmatau.dev.totdev(x, tau0, m_list)
%   result = sigmatau.dev.totdev(x, tau0, m_list, 'data_type', 'freq')
%
%   Extends data by symmetric reflection to reduce endpoint effects, then
%   computes overlapping second differences. SP1065 §5.2.11 Eq. 25.
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
    'total_type', 'totvar', ...
    'bias_type',  'totvar'  ...
);

result = sigmatau.dev.engine(x, tau0, m_list, @totdev_kernel, params, varargin{:});
end

function [v, neff] = totdev_kernel(x, m, tau0, ~)
% Linear detrend + symmetric reflection, then overlapping second differences.
% SP1065 §5.2.11 Eq. 25: denominator uses 2τ²(N-2) for phase form.
N  = numel(x);
if N < 2
    v = NaN; neff = 0;
    return;
end
xd = sigmatau.util.detrend(x, 1);

off = N - 2;
i   = (1:N)';
lo_idx  = off + i;
mid_idx = off + i + m;
hi_idx  = off + i + 2*m;

% Filter valid indices (within 3N-4)
valid = hi_idx <= (3*N - 4);
lo  = lo_idx(valid);
mid = mid_idx(valid);
hi  = hi_idx(valid);

if isempty(hi)
    v = NaN; neff = 0;
    return;
end

% Compute values at reflected indices without allocating full 3N-4 array
v_lo  = get_val(lo,  xd, N);
v_mid = get_val(mid, xd, N);
v_hi  = get_val(hi,  xd, N);

d2 = v_hi - 2*v_mid + v_lo;
D  = sum(d2.^2);
count = numel(d2);

% SP1065 §5.2.11 Eq. 25: phase form uses 2τ²(N-2)
v    = D / (2 * (N-2) * (m*tau0)^2);
neff = count;
end

function vals = get_val(k, xd, N)
% Simulated indexing into x_star = [x_left; xd; x_right]
% x_left(j)  = 2*xd(1) - xd(j+1) for j=1:N-2
% x_right(j) = 2*xd(N) - xd(N-j) for j=1:N-2
vals = zeros(size(k));

% Left reflection: 1 .. N-2
mask1 = (k <= N-2);
if any(mask1)
    vals(mask1) = 2*xd(1) - xd(k(mask1) + 1);
end

% Center: N-1 .. 2N-2
mask2 = (k >= N-1) & (k <= 2*N-2);
if any(mask2)
    vals(mask2) = xd(k(mask2) - (N-2));
end

% Right reflection: 2N-1 .. 3N-4
mask3 = (k >= 2*N-1);
if any(mask3)
    vals(mask3) = 2*xd(end) - xd(N - (k(mask3) - (2*N-2)));
end
end

