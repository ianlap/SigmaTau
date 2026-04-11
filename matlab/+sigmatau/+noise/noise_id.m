function alpha_list = noise_id(x, m_list, data_type, dmin, dmax)
% NOISE_ID  Dominant power-law noise estimator. Returns alpha per m.
%
%   alpha_list = sigmatau.noise.noise_id(x, m_list, data_type, dmin, dmax)
%
%   For N_eff >= 30: uses lag-1 ACF method (SP1065 §5.6).
%   For N_eff <  30: falls back to B1-ratio / R(n) method.
%
%   Inputs:
%     x         – phase data (column vector, already validated)
%     m_list    – averaging factors
%     data_type – 'phase' (default) or 'freq'
%     dmin      – minimum differencing depth (default 0)
%     dmax      – maximum differencing depth (default 2)
%
%   Output:
%     alpha_list – estimated alpha values (NaN where estimation fails)

if nargin < 3 || isempty(data_type), data_type = 'phase'; end
if nargin < 4 || isempty(dmin),      dmin = 0; end
if nargin < 5 || isempty(dmax),      dmax = 2; end

x_clean    = preprocess_x(x);
alpha_list = NaN(size(m_list));

for k = 1:numel(m_list)
    m     = m_list(k);
    N_eff = floor(numel(x_clean) / m);
    try
        if N_eff >= 30
            alpha_list(k) = noise_id_lag1acf(x_clean, m, data_type, dmin, dmax);
        else
            alpha_list(k) = noise_id_b1rn(x_clean, m, data_type);
        end
    catch err
        warning('SigmaTau:noise_id', 'Estimation failed for m=%d: %s', m, err.message);
    end
end
end

% ── Preprocessing ─────────────────────────────────────────────────────────────

function x_out = preprocess_x(x)
% Remove >5σ outliers then linear detrend (mirrors Julia _preprocess).
x = x(:);
x_mean = mean(x);
x_std  = std(x);
if x_std < eps
    x_out = sigmatau.util.detrend_linear(x);
    return;
end
z     = abs((x - x_mean) / x_std);
x_out = sigmatau.util.detrend_linear(x(z < 5.0));
end

% ── Lag-1 ACF method ──────────────────────────────────────────────────────────

function alpha = noise_id_lag1acf(x, m, data_type, dmin, dmax)
% SP1065 §5.6 lag-1 autocorrelation method.
if strcmpi(data_type, 'phase')
    if m > 1
        x = x(1:m:end);
    end
    x = sigmatau.util.detrend_quadratic(x);
elseif strcmpi(data_type, 'freq')
    N = floor(numel(x) / m) * m;
    x = mean(reshape(x(1:N), m, []), 1)';
    x = sigmatau.util.detrend_linear(x);
else
    error('SigmaTau:noise_id', 'data_type must be ''phase'' or ''freq''');
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
        error('SigmaTau:noise_id', 'Data too short after differencing');
    end
end
end

function r1 = lag1_acf(x)
x = x(:) - mean(x);
if all(x == 0)
    r1 = NaN;
    return;
end
r1 = sum(x(1:end-1) .* x(2:end)) / sum(x.^2);
end

% ── B1-ratio / R(n) fallback ──────────────────────────────────────────────────

function alpha_int = noise_id_b1rn(x, m, data_type)
% B1-ratio and R(n) fallback for small N_eff (SP1065 §5.6).
if strcmpi(data_type, 'phase')
    x_dec = x(1:m:end);
    x_dec = sigmatau.util.detrend_quadratic(x_dec);
    avar_val   = simple_avar(x_dec, 1);
    N_avar     = numel(x_dec) - 2;

    dx  = diff(x);
    Nd  = floor(numel(dx) / m) * m;
    if Nd < m
        alpha_int = 0;
        return;
    end
    dx        = dx(1:Nd);
    y_blocks  = reshape(dx, m, []);
    y_avg     = mean(y_blocks, 1)';
    var_class = var(y_avg, 0);   % corrected=false in Julia → var with 0 flag in MATLAB

elseif strcmpi(data_type, 'freq')
    N = floor(numel(x) / m) * m;
    if N < 2*m
        alpha_int = 0;
        return;
    end
    y_avg     = mean(reshape(x(1:N), m, []), 1)';
    y_avg     = sigmatau.util.detrend_linear(y_avg);
    dy        = diff(y_avg);
    var_class = var(y_avg, 0);
    avar_val  = sum(dy.^2) / (2 * (numel(y_avg) - 1));
    N_avar    = numel(y_avg);
else
    error('SigmaTau:noise_id', 'data_type must be ''phase'' or ''freq''');
end

if isnan(avar_val) || avar_val <= 0
    alpha_int = 0;
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
    mdev_val = simple_mdev(x, m, 1.0);
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

% ── B1 / R(n) theory ─────────────────────────────────────────────────────────

function B1 = b1_theory(N, mu)
switch mu
    case 2;  B1 = N*(N+1)/6;
    case 1;  B1 = N/2;
    case 0;  B1 = N*log(N) / (2*(N-1)*log(2));
    case -1; B1 = 1.0;
    case -2; B1 = (N^2 - 1) / (1.5 * N * (N-1));
    otherwise
        B1 = (N * (1 - N^mu)) / (2 * (N-1) * (1 - 2^mu));
end
end

function Rn = rn_theory(af, b)
switch b
    case 0
        Rn = 1.0 / af;
    case -1
        avar = (1.038 + 3*log(2*pi*0.5*af)) / (4*pi^2);
        mvar = 3*log(256/27) / (8*pi^2);
        Rn   = mvar / avar;
    otherwise
        Rn = 1.0;
end
end

% ── Helpers ───────────────────────────────────────────────────────────────────

function v = simple_avar(x, m)
% Basic overlapping Allan variance at averaging factor m. SP1065 Eq. 10.
N = numel(x);
L = N - 2*m;
if L <= 0
    v = NaN;
    return;
end
d2 = x(1+2*m:N) - 2*x(1+m:N-m) + x(1:L);
v  = sum(d2.^2) / (L * 2 * m^2);
end

function md = simple_mdev(x, m, tau0)
% Basic modified Allan deviation via prefix-sum (no noise ID).
N  = numel(x);
Ne = N - 3*m + 1;
if Ne <= 0
    md = NaN;
    return;
end
S  = cumsum([0; x]);
s1 = S(1+m:Ne+m)   - S(1:Ne);
s2 = S(1+2*m:Ne+2*m) - S(1+m:Ne+m);
s3 = S(1+3*m:Ne+3*m) - S(1+2*m:Ne+2*m);
d  = (s3 - 2*s2 + s1) / m;
md = sqrt(sum(d.^2) / (Ne * 2 * m^2 * tau0^2));
end
