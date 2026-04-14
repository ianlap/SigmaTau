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
    'dmax',       2,          ...
    'total_type', 'mhtot',    ...
    'bias_type',  ''          ...
);

result = sigmatau.dev.engine(x, tau0, m_list, @mhtotdev_kernel, params, varargin{:});
end

function [v, neff] = mhtotdev_kernel(x, m, tau0)
% Linear detrend per phase segment, symmetric reflection, third diffs + moving avg.
assert(m >= 1, 'SigmaTau:mhtotdev', 'averaging factor m must be >= 1');
N     = numel(x);
nsubs = N - 4*m + 1;
if nsubs < 1
    v = NaN; neff = 0;
    return;
end

Lp        = 3*m + 1;   % phase segment length
total_sum = 0.0;

CX  = cumsum([0; x(:)]);
CXT = cumsum([0; x(:) .* (1:N)']);

% Precompute indexing ranges
p_range = (0:Lp)';
T1 = p_range;
T2 = p_range .* (p_range + 1) / 2;
j_range = (1:5*m+4)';

for n = 1:nsubs
    % Linear detrend without slicing: using centered coordinates for stability
    % t_center = (1:Lp) - (Lp+1)/2
    sx  = CX(n+Lp) - CX(n);
    sxt = (CXT(n+Lp) - CXT(n)) - (n-1) * sx;
    
    t_mid = (Lp + 1) / 2;
    sxt_center = sxt - t_mid * sx;
    
    % Beta for centered coordinates: [1, t_center] are orthogonal
    % sum(t_center) = 0, sum(t_center^2) = Lp*(Lp^2-1)/12
    b1 = sx / Lp;
    b2 = sxt_center / (Lp * (Lp^2 - 1) / 12);
    
    % pd(k) = x(n+k-1) - (b1 + b2*(k - t_mid))
    %       = x(n+k-1) - ((b1 - b2*t_mid) + b2*k)
    intercept = b1 - b2 * t_mid;
    slope     = b2;

    % SumPD_vec(p+1) = sum_{k=1}^p (x(n+k-1) - (intercept + slope*k))
    SumPD_vec = (CX(n+p_range) - CX(n)) - (intercept * T1 + slope * T2);

    % Reflection: CE has 3 parts [rev(pd); pd; rev(pd)]
    % SR(p+1) = cumsum([0; rev(pd)], p)
    S_end = SumPD_vec(end);
    SR = S_end - SumPD_vec(end:-1:1);
    CE = [SR; S_end + SumPD_vec(2:end); 2*S_end + SR(2:end)];

    % avg(i) = sum_{k=0}^{m-1} d3_vec(i+k)
    % Through 4th-order differences of CE:
    % avg = -CE(i) + 4*CE(i+m) - 6*CE(i+2m) + 4*CE(i+3m) - CE(i+4m)
    avg = -CE(j_range) + 4*CE(j_range+m) - 6*CE(j_range+2*m) + 4*CE(j_range+3*m) - CE(j_range+4*m);

    block_var = sum(avg.^2) / (numel(avg) * 6 * m^2);
    total_sum = total_sum + block_var;
end

v    = total_sum / (nsubs * (m*tau0)^2);
neff = nsubs;
end
