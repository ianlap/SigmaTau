function result = engine(x, tau0, m_list, kernel, params, varargin)
% ENGINE  Shared deviation computation engine.
%
%   result = sigmatau.dev.engine(x, tau0, m_list, kernel, params, ...)
%
%   Inputs:
%     x       – phase data (or frequency data if data_type='freq')
%     tau0    – sampling interval (s)
%     m_list  – averaging factors [], or empty [] for auto-generated octave-spaced
%     kernel  – function handle @(x, m, tau0) → [variance, neff]
%               kernels return VARIANCE; engine takes sqrt
%     params  – struct with fields:
%                 .name        – deviation identifier ('adev', 'mdev', ...)
%                 .min_factor  – N/m ratio for default m_list (2, 3, or 4)
%                 .d           – difference order (2=Allan, 3=Hadamard)
%                 .F_fn        – @(m) → F (m for unmodified, 1 for modified)
%                 .dmin, .dmax – noise_id differencing bounds
%               Total EDF/bias behavior is inferred from params.name.
%
%   Name-Value:
%     'data_type' – 'phase' (default) or 'freq'
%
%   Output:
%     result – struct with fields: tau, deviation, edf, ci, alpha, neff,
%              tau0, N, method, confidence

CONFIDENCE_DEFAULT = 0.683;   % 1σ (68.3%) — SP1065 default confidence level

% Parse name-value options
p = inputParser;
addParameter(p, 'data_type', 'phase');
parse(p, varargin{:});
data_type = p.Results.data_type;

% Frequency-to-phase conversion — CLAUDE.md §Architecture
% cumsum(y)*tau0 produces phase in seconds from fractional-frequency samples.
if strcmpi(data_type, 'freq')
    x = cumsum(x(:)) * tau0;
elseif ~strcmpi(data_type, 'phase')
    error('SigmaTau:engine', 'data_type must be ''phase'' or ''freq'', got ''%s''.', data_type);
end

x    = sigmatau.util.validate_phase_data(x);
tau0 = sigmatau.util.validate_tau0(tau0);
N    = numel(x);

if isempty(m_list)
    m_list = sigmatau.util.default_mlist(N, params.min_factor);
end
m_list = m_list(m_list >= 1);   % Step 5: Filter invalid averaging factors
m_list = m_list(:)';            % row vector

if isempty(m_list)
    result = empty_result(params.name, tau0, N);
    return;
end

% Noise identification (returns NaN where estimation fails)
alpha_float = sigmatau.noise.noise_id(x, m_list, 'phase', params.dmin, params.dmax);

tau  = m_list * tau0;
L    = numel(m_list);
dev  = NaN(1, L);
neff = zeros(1, L);
edf  = NaN(1, L);

T_rec = (N - 1) * tau0;   % record duration for total deviation EDF

for k = 1:L
    m = m_list(k);
    [var_val, n] = kernel(x, m, tau0);

    if n <= 0 || isnan(var_val)
        dev(k)  = NaN;
        neff(k) = 0;
        edf(k)  = NaN;
        continue;
    end

    % Step 1: Guard against negative variance from catastrophic cancellation.
    % Small negative values (rounding) are clamped to 0; significant negatives → NaN.
    % Threshold: -eps * N (scale of sum operations).
    if var_val < -eps * N
        dev(k) = NaN;
    else
        dev(k) = sqrt(max(var_val, 0));
    end
    neff(k) = n;

    % EDF computation
    a = alpha_float(k);
    if isnan(a)
        alpha_k = 0;
    else
        alpha_k = round(a);
    end

    total_type = total_type_for_name(params.name);
    if ~isempty(total_type)
        edf(k) = sigmatau.stats.totaldev_edf(total_type, alpha_k, T_rec, tau(k));
    else
        F = params.F_fn(m);
        edf(k) = sigmatau.stats.calculate_edf(alpha_k, params.d, m, F, 1, N);
    end
end

% Round alpha to integers (NaN → 0)
alpha_int = zeros(1, L);
for k = 1:L
    a = alpha_float(k);
    if ~isnan(a)
        alpha_int(k) = round(a);
    end
end

result = struct( ...
    'tau',        tau,        ...
    'deviation',  dev,        ...
    'edf',        edf,        ...
    'ci',         NaN(L, 2),  ...
    'alpha',      alpha_int,  ...
    'neff',       neff,       ...
    'tau0',       tau0,       ...
    'N',          N,          ...
    'method',     params.name,...
    'confidence', CONFIDENCE_DEFAULT ...
);

% Bias correction (applied in-place; only for total deviations) — must run
% before CI so intervals bracket the bias-corrected deviation.
bias_type = bias_type_for_name(params.name);
if ~isempty(bias_type)
    result = apply_bias(result, bias_type, T_rec);
end

result.ci = sigmatau.stats.ci(result);
end

% ── Helpers ───────────────────────────────────────────────────────────────────

function total_type = total_type_for_name(name)
switch name
    case 'totdev'
        total_type = 'totvar';
    case 'mtotdev'
        total_type = 'mtot';
    case 'htotdev'
        total_type = 'htot';
    case 'mhtotdev'
        total_type = 'mhtot';
    otherwise
        total_type = '';
end
end

function bias_type = bias_type_for_name(name)
switch name
    case 'totdev'
        bias_type = 'totvar';
    case 'htotdev'
        bias_type = 'htot';
    otherwise
        bias_type = '';
end
end

function result = empty_result(name, tau0, N)
CONFIDENCE_DEFAULT = 0.683;
result = struct( ...
    'tau',        zeros(1,0), ...
    'deviation',  zeros(1,0), ...
    'edf',        zeros(1,0), ...
    'ci',         zeros(0,2), ...
    'alpha',      zeros(1,0), ...
    'neff',       zeros(1,0), ...
    'tau0',       tau0,       ...
    'N',          N,          ...
    'method',     name,       ...
    'confidence', CONFIDENCE_DEFAULT ...
);
end

function result = apply_bias(result, bias_type, T_rec)
% Divide deviation by bias factor B(alpha). CI is filled after this step,
% so no rescaling of CI is needed here.
B = sigmatau.stats.bias(result.alpha, bias_type, result.tau, T_rec);
result.deviation = result.deviation ./ B(:)';
end
