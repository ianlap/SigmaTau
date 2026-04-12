%TEST_DEVIATIONS  Smoke tests + white FM slope tests for all 10 deviation wrappers.
%
% Part 1: Structural smoke tests — output struct has expected fields,
%         tau > 0, at least one finite positive deviation value.
%
% Part 2: White FM slope test — generates phase data from white FM frequency
%         noise and verifies that each deviation's log-log slope is within 5%
%         of the theoretical value.  SP1065 Table 1 / IEEE 1139-2022.
%
%   White FM (alpha=0) expected slopes:
%     adev, mdev, hdev, mhdev, totdev, mtotdev, htotdev, mhtotdev  →  -1/2
%     tdev = tau·MDEV/√3,  ldev = tau·MHDEV/√(10/3)               →  +1/2

% ── Part 1: Structural smoke tests ───────────────────────────────────────────
rng(42);
N    = 500;
tau0 = 1.0;
x    = cumsum(randn(N,1));   % random walk phase (RWFM-like)

m_list = [1, 2, 4, 8, 16];

devfns = {
    @sigmatau.dev.adev,    'adev';
    @sigmatau.dev.mdev,    'mdev';
    @sigmatau.dev.tdev,    'tdev';
    @sigmatau.dev.hdev,    'hdev';
    @sigmatau.dev.mhdev,   'mhdev';
    @sigmatau.dev.ldev,    'ldev';
    @sigmatau.dev.totdev,  'totdev';
    @sigmatau.dev.mtotdev, 'mtotdev';
    @sigmatau.dev.htotdev, 'htotdev';
    @sigmatau.dev.mhtotdev,'mhtotdev';
};

for k = 1:size(devfns, 1)
    fn   = devfns{k,1};
    name = devfns{k,2};

    result = fn(x, tau0, m_list);

    % Required fields
    assert(isfield(result, 'tau'),       sprintf('%s: missing .tau', name));
    assert(isfield(result, 'deviation'), sprintf('%s: missing .deviation', name));
    assert(isfield(result, 'edf'),       sprintf('%s: missing .edf', name));
    assert(isfield(result, 'ci'),        sprintf('%s: missing .ci', name));
    assert(isfield(result, 'alpha'),     sprintf('%s: missing .alpha', name));
    assert(isfield(result, 'neff'),      sprintf('%s: missing .neff', name));
    assert(isfield(result, 'method'),    sprintf('%s: missing .method', name));
    assert(strcmp(result.method, name),  sprintf('%s: wrong method name', name));

    % Length consistency
    L = numel(result.tau);
    assert(L > 0,                        sprintf('%s: empty tau', name));
    assert(numel(result.deviation) == L, sprintf('%s: deviation length mismatch', name));
    assert(isequal(size(result.ci), [L 2]), sprintf('%s: ci size wrong', name));

    % tau values
    assert(all(result.tau > 0),          sprintf('%s: tau should be positive', name));

    % deviation values (may have some NaN at large m, but at least one finite)
    assert(any(isfinite(result.deviation) & result.deviation > 0), ...
           sprintf('%s: no finite positive deviations', name));

    fprintf('  OK: %s\n', name);
end

fprintf('test_deviations Part 1 (smoke): all assertions passed\n');

% ── Part 2: White FM slope tests ─────────────────────────────────────────────
% White FM: frequency samples y ~ i.i.d. N(0,1); phase = [0; cumsum(y)]*tau0.
% N=4096 gives Neff > 3500 at all taus, so each deviation estimate has
% relative std < 2.5%, making the log-log slope accurate to < 0.01.
% 5% tolerance (±0.025 on slope magnitude 0.5) gives ample margin.

rng(12345);
N_s   = 4096;
tau0  = 1.0;
y_wfm = randn(N_s - 1, 1);              % white FM frequency samples
x_s   = [0; cumsum(y_wfm)] * tau0;      % phase = running integral of frequency

% Standard deviations (adev/hdev/totdev/htotdev) have exact slope -0.5 for
% all m; start from m=1.
m_std  = 2.^(0:5);   % [1 2 4 8 16 32]

% Modified estimators (mdev/mhdev/mtotdev/mhtotdev) and their derived forms
% (tdev/ldev) use a moving-average construction that only reaches its
% asymptotic slope of -0.5 for m >= 4.  Starting from m=1 introduces a
% systematic slope bias of ~10-14%, failing the 5% check.  At m=4 the bias
% is < 2% (derived from the exact spectral covariance formula for white FM).
m_mod  = 2.^(2:7);   % [4 8 16 32 64 128]  — fast O(N) modified wrappers
m_mods = 2.^(2:6);   % [4 8 16 32  64]     — total modified (O(N·m), speed cap)

TOL_rel = 0.05;   % 5% relative tolerance on log-log slope

%                    col 1: function handle          col 2: name       col 3: m_list  col 4: expected slope
slope_tests = {
    @sigmatau.dev.adev,      'adev',      m_std,   -0.5;
    @sigmatau.dev.mdev,      'mdev',      m_mod,   -0.5;
    @sigmatau.dev.hdev,      'hdev',      m_std,   -0.5;
    @sigmatau.dev.mhdev,     'mhdev',     m_mod,   -0.5;
    @sigmatau.dev.tdev,      'tdev',      m_mod,   +0.5;   % TDEV = tau·MDEV/sqrt(3)
    @sigmatau.dev.ldev,      'ldev',      m_mod,   +0.5;   % LDEV = tau·MHDEV/sqrt(10/3)
    @sigmatau.dev.totdev,    'totdev',    m_std,   -0.5;
    @sigmatau.dev.mtotdev,   'mtotdev',   m_mods,  -0.5;
    @sigmatau.dev.htotdev,   'htotdev',   m_std,   -0.5;
    @sigmatau.dev.mhtotdev,  'mhtotdev',  m_mods,  -0.5;
};

for k = 1:size(slope_tests, 1)
    fn        = slope_tests{k,1};
    name      = slope_tests{k,2};
    ml        = slope_tests{k,3};
    exp_slope = slope_tests{k,4};

    result = fn(x_s, tau0, ml);

    % Need at least 3 finite points for a meaningful log-log linear fit.
    valid = isfinite(result.deviation) & result.deviation > 0;
    assert(sum(valid) >= 3, ...
        sprintf('%s: need ≥3 finite points for slope fit, got %d', name, sum(valid)));

    % Log-log linear fit: log10(dev) = slope*log10(tau) + intercept.
    lt        = log10(result.tau(valid));
    ld        = log10(result.deviation(valid));
    p         = polyfit(lt, ld, 1);
    slope_obs = p(1);

    % 5% relative tolerance: for |expected| = 0.5 this gives ±0.025.
    tol = TOL_rel * abs(exp_slope);
    assert(abs(slope_obs - exp_slope) <= tol, ...
        sprintf('%s: slope %.4f outside 5%% of expected %.1f (±%.4f)', ...
                name, slope_obs, exp_slope, tol));

    fprintf('  slope OK: %-10s  observed=%+.3f  expected=%+.1f\n', ...
            name, slope_obs, exp_slope);
end

fprintf('test_deviations Part 2 (white FM slopes): all slope checks passed\n');
fprintf('test_deviations: all assertions passed\n');
