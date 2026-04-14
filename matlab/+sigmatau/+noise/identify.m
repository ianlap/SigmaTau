function alpha_list = identify(x, m_list, data_type, dmin, dmax)
% IDENTIFY  Dominant power-law noise estimator. Returns alpha per m.
%
%   alpha_list = sigmatau.noise.identify(x, m_list, data_type, dmin, dmax)
%
%   Dispatch (SP1065 §5.6 / Stable32 manual):
%     1. N_eff >= NEFF_RELIABLE : lag-1 ACF (primary)
%     2. N_eff <  NEFF_RELIABLE : B1-ratio / R(n) (reliable once m^2 scaled)
%     3. Both above return NaN  : carry forward the most recent reliable alpha
%
%   Carry-forward is the Stable32 manual's "use the previous noise type
%   estimate at the longest averaging time" rule — the last-resort when
%   neither ACF nor B1/R(n) can produce an estimate for this tau.
%
%   Inputs:
%     x         – phase or frequency data (column vector)
%     m_list    – averaging factors
%     data_type – 'phase' (default) or 'freq'
%     dmin      – minimum differencing depth (default 0)
%     dmax      – maximum differencing depth (default 2)
%
%   Output:
%     alpha_list – estimated alpha values (NaN where estimation fails)

% NEFF_RELIABLE: SP1065 §5.6 cites 30 as the theoretical lag-1 ACF minimum,
% but the estimator is still high-variance just above that. 50 is a
% conservative working threshold that stabilises the long-τ tail.
NEFF_RELIABLE = 50;

if nargin < 3 || isempty(data_type), data_type = 'phase'; end
if nargin < 4 || isempty(dmin),      dmin = 0; end
if nargin < 5 || isempty(dmax),      dmax = 2; end

x_clean       = preprocess_x(x);
alpha_list    = NaN(size(m_list));
last_reliable = NaN;

for k = 1:numel(m_list)
    m     = m_list(k);
    N_eff = floor(numel(x_clean) / m);
    alpha = NaN;
    try
        if N_eff >= NEFF_RELIABLE
            alpha = identify_lag1acf(x_clean, m, data_type, dmin, dmax);
        else
            alpha = identify_b1rn(x_clean, m, data_type);
        end
    catch err
        if strcmp(err.identifier, 'SigmaTau:identify')
            warning('SigmaTau:identify', 'Estimation failed for m=%d: %s', m, err.message);
        else
            rethrow(err);
        end
    end

    if ~isnan(alpha)
        alpha_list(k) = alpha;
        last_reliable = alpha;
    elseif ~isnan(last_reliable)
        alpha_list(k) = last_reliable;   % last-resort: carry forward
    end
end
end

% ── Lag-1 ACF method ──────────────────────────────────────────────────────────

function alpha = identify_lag1acf(x, m, data_type, dmin, dmax)
% SP1065 §5.6 lag-1 autocorrelation method.
if strcmpi(data_type, 'phase')
    if m > 1
        x = x(1:m:end);
    end
    x = sigmatau.util.detrend(x, 2);
elseif strcmpi(data_type, 'freq')
    N = floor(numel(x) / m) * m;
    x = mean(reshape(x(1:N), m, []), 1)';
    x = sigmatau.util.detrend(x, 1);
else
    error('SigmaTau:identify', 'data_type must be ''phase'' or ''freq''');
end

d = 0;
while true
    r1  = lag1_acf(x);
    rho = r1 / (1 + r1);

    if d >= dmin && (rho < 0.25 || d >= dmax)
        p     = -2 * (rho + d);
        alpha = p + 2 * strcmpi(data_type, 'phase');
        return;
    end
    x = diff(x);
    d = d + 1;
    if numel(x) < 5
        error('SigmaTau:identify', 'Data too short after differencing');
    end
end
end

function r1 = lag1_acf(x)
x   = x(:) - mean(x);
ssx = sum(x.^2);
% Guard for constant input: detrend residuals are O(eps)*N rather than
% exact zeros, so use a tolerance rather than ==0.
if ssx < eps * numel(x)
    r1 = NaN;
    return;
end
r1 = sum(x(1:end-1) .* x(2:end)) / ssx;
end

% ── B1-ratio / R(n) fallback ──────────────────────────────────────────────────

function alpha_int = identify_b1rn(x, m, data_type)
% B1-ratio and R(n) fallback for small N_eff (SP1065 §5.6).
% Note: the isnan/avar guard is an explicit `if`, not an operator expression.
% This avoids the Julia operator-precedence bug fixed in PR #7.
if strcmpi(data_type, 'phase')
    x_dec = x(1:m:end);
    x_dec = sigmatau.util.detrend(x_dec, 2);
    % AVAR at tau = m*tau0. Computed from decimated phase so the detrend
    % above carries through; the m^2 factor corrects simple_avar(..., 1)
    % to SP1065 Eq. 14's m^2*tau0^2 denominator.
    avar_val   = sigmatau.dev.adev_kernel(x_dec, 1, 1.0) / double(m)^2;
    N_avar     = numel(x_dec) - 2;

    dx  = diff(x);
    Nd  = floor(numel(dx) / m) * m;
    if Nd < m
        alpha_int = NaN;   % signal failure to caller
        return;
    end
    dx        = dx(1:Nd);
    y_blocks  = reshape(dx, m, []);
    y_avg     = mean(y_blocks, 1)';
    var_class = var(y_avg, 0);   % normalise by N (corrected=false in Julia)

elseif strcmpi(data_type, 'freq')
    N = floor(numel(x) / m) * m;
    if N < 2*m
        alpha_int = NaN;   % signal failure to caller
        return;
    end
    y_avg     = mean(reshape(x(1:N), m, []), 1)';
    y_avg     = sigmatau.util.detrend(y_avg, 1);
    dy        = diff(y_avg);
    var_class = var(y_avg, 0);
    avar_val  = sum(dy.^2) / (2 * (numel(y_avg) - 1));
    N_avar    = numel(y_avg);
else
    error('SigmaTau:identify', 'data_type must be ''phase'' or ''freq''');
end

% Guard: if avar is NaN or non-positive, signal failure to caller.
if isnan(avar_val) || avar_val <= 0
    alpha_int = NaN;
    return;
end
B1_obs = var_class / avar_val;

mu_list    = [1, 0, -1, -2];
alpha_list = [-2, -1, 0, 2];
b1_vals    = arrayfun(@(mu) b1_theory(N_avar, mu), mu_list);

mu_best   = mu_list(end);
alpha_int = alpha_list(end);

for i = 1:numel(mu_list)-1
    boundary = sqrt(b1_vals(i) * b1_vals(i+1));
    if B1_obs > boundary
        mu_best   = mu_list(i);
        alpha_int = alpha_list(i);
        break;
    end
end

% Refine alpha=2 vs alpha=1 using R(n) for White PM vs Flicker PM
if mu_best == -2 && strcmpi(data_type, 'phase')
    adev_val = sqrt(avar_val);
    mdev_val = sqrt(sigmatau.dev.mdev_kernel(x, m, 1.0));
    if ~isnan(mdev_val) && adev_val > 0
        Rn_obs = (mdev_val / adev_val)^2;
        R_hi   = rn_theory(m, 0);    % alpha=2 (White PM)
        R_lo   = rn_theory(m, -1);   % alpha=1 (Flicker PM)
        if Rn_obs > sqrt(R_hi * R_lo)
            alpha_int = 1;
        else
            alpha_int = 2;
        end
    end
end
end
