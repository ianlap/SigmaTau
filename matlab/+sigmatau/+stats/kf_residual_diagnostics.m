function diag = kf_residual_diagnostics(innovations, residuals, varargin)
% KF_RESIDUAL_DIAGNOSTICS  Three diagnostics for KF residuals: whiteness + bias.
%
%   diag = sigmatau.stats.kf_residual_diagnostics(innovations, residuals)
%   diag = sigmatau.stats.kf_residual_diagnostics(innovations, residuals, ...
%             'normalized_innovations', nu_norm, 'lag', L, 'significance', alpha)
%
%   Diagnostics:
%     1. Raw innovation whiteness (Ljung-Box on innovations).
%        Pass iff p > significance.
%     2. Normalized innovation whiteness (Ljung-Box on nu/sqrt(S); optional).
%        Fields are NaN when normalized_innovations not supplied.
%     3. Posterior-residual bias under naive iid SE: |mean| < 3*sigma/sqrt(N).
%        Coarse health-check; not rigorous when residuals are autocorrelated.
%
%   Inputs:
%     innovations             – Nx1 raw innovations (z - H*x_prior)
%     residuals               – Nx1 posterior residuals (z - H*x_post)
%     normalized_innovations  – Nx1 (optional, default [])
%     lag                     – Ljung-Box lag (default min(20, floor(N/5)))
%     significance            – test threshold (default 0.05)
%
%   Output struct fields:
%     innov_lb_pvalue, innov_lb_passed              – D1
%     norm_innov_lb_pvalue, norm_innov_lb_passed    – D2 (NaN when N/A)
%     resid_mean, resid_se, resid_bias_passed       – D3
%     lag, significance, n                          – echo back
%
%   References:
%     Anderson & Moore, "Optimal Filtering" (1979) §5
%     Ljung & Box, Biometrika 65(2):297-303 (1978)
%
%   Requires Statistics and Machine Learning Toolbox (chi2cdf). The Ljung-Box
%   statistic is computed inline rather than via Econometrics Toolbox `lbqtest`,
%   keeping the helper toolbox-light (Econometrics not assumed installed).

innov = innovations(:);
resid = residuals(:);
n     = numel(innov);

if n < 2
    error('sigmatau:stats:kf_residual_diagnostics:tooShort', ...
        'innovations must have length >= 2 (got %d)', n);
end
if numel(resid) ~= n
    error('sigmatau:stats:kf_residual_diagnostics:lenMismatch', ...
        'residuals length %d != innovations length %d', numel(resid), n);
end

p = inputParser;
p.addParameter('normalized_innovations', [], @(v) isempty(v) || isnumeric(v));
p.addParameter('lag',                    min(20, floor(n/5)), ...
    @(v) isnumeric(v) && isscalar(v) && v >= 1);
p.addParameter('significance',           0.05, ...
    @(v) isnumeric(v) && isscalar(v) && v > 0 && v < 1);
p.parse(varargin{:});
opts = p.Results;

lag = round(opts.lag);
if lag >= n
    error('sigmatau:stats:kf_residual_diagnostics:lagTooLarge', ...
        'lag must be < length(innovations)=%d (got %d)', n, lag);
end

% D1: Ljung-Box on raw innovations
p_innov    = ljung_box_pvalue(innov, lag);
pass_innov = p_innov > opts.significance;

% D2: Ljung-Box on normalized innovations (optional)
if isempty(opts.normalized_innovations)
    p_norm    = NaN;
    pass_norm = NaN;
else
    norm_innov = opts.normalized_innovations(:);
    if numel(norm_innov) ~= n
        error('sigmatau:stats:kf_residual_diagnostics:normLenMismatch', ...
            'normalized_innovations length %d != innovations length %d', ...
            numel(norm_innov), n);
    end
    p_norm    = ljung_box_pvalue(norm_innov, lag);
    pass_norm = p_norm > opts.significance;
end

% D3: posterior-residual bias under naive iid SE
mu        = mean(resid);
sig       = std(resid);
se        = 3.0 * sig / sqrt(n);
pass_bias = abs(mu) < se;

diag = struct( ...
    'innov_lb_pvalue',         p_innov, ...
    'innov_lb_passed',         pass_innov, ...
    'norm_innov_lb_pvalue',    p_norm, ...
    'norm_innov_lb_passed',    pass_norm, ...
    'resid_mean',              mu, ...
    'resid_se',                se, ...
    'resid_bias_passed',       pass_bias, ...
    'lag',                     lag, ...
    'significance',            opts.significance, ...
    'n',                       n);
end

% ── Helpers ───────────────────────────────────────────────────────────────────

function pval = ljung_box_pvalue(x, h)
% LJUNG_BOX_PVALUE  Q = n(n+2) sum_{k=1..h} rho_k^2 / (n-k); p = 1 - chi2cdf(Q,h).
% rho_k uses the biased autocorrelation (n cancels in numerator/denominator).
x  = x(:);
n  = length(x);
xc = x - mean(x);
c0 = sum(xc .* xc);
if c0 == 0
    pval = 1.0;   % constant series — no autocorrelation to detect
    return;
end

acc = 0.0;
for k = 1:h
    rho_k = sum(xc(1:n-k) .* xc(k+1:n)) / c0;
    acc   = acc + rho_k^2 / (n - k);
end
Q    = n * (n + 2) * acc;
pval = 1 - chi2cdf(Q, h);
end
