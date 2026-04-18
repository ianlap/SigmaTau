%TEST_KF_DIAGNOSTICS  Unit tests for sigmatau.stats.kf_residual_diagnostics.
%
% Three controlled inputs, one per diagnostic:
%   white noise           -> passes D1, D2 (when supplied), D3
%   AR(1) rho=0.5         -> fails D1 (autocorrelated)
%   biased N(1.0, 1.0)    -> fails D3

% ── Test 1: white noise passes all three ───────────────────────────────────
fprintf('Test 1: white noise passes D1+D2+D3 ... ');
rng(42);
N          = 1000;
innov      = randn(N, 1);
resid      = randn(N, 1);
norm_innov = randn(N, 1);

d = sigmatau.stats.kf_residual_diagnostics(innov, resid, ...
        'normalized_innovations', norm_innov);

assert(d.innov_lb_passed,           'D1: whiteness should pass for white noise');
assert(d.norm_innov_lb_passed == true, 'D2: normalized whiteness should pass for white noise');
assert(d.resid_bias_passed,         'D3: bias should pass for zero-mean white noise');
assert(d.n == N,                    'n field mismatch');
assert(d.lag == min(20, floor(N/5)), 'lag default mismatch');
assert(d.significance == 0.05,      'significance default mismatch');
fprintf('PASSED (p_innov=%.3f, p_norm=%.3f, |mu|=%.3e)\n', ...
    d.innov_lb_pvalue, d.norm_innov_lb_pvalue, abs(d.resid_mean));

% ── Test 2: AR(1) rho=0.5 fails whiteness ────────────────────────────────
fprintf('Test 2: AR(1) rho=0.5 fails D1 ... ');
rng(43);
ar1 = zeros(N, 1);
ar1(1) = randn();
for k = 2:N
    ar1(k) = 0.5 * ar1(k-1) + randn();
end

d2 = sigmatau.stats.kf_residual_diagnostics(ar1, randn(N, 1));

assert(~d2.innov_lb_passed,         'D1: should fail for AR(1) rho=0.5');
assert(d2.innov_lb_pvalue < 0.05,   'D1 p-value should be small for AR(1)');
assert(isnan(d2.norm_innov_lb_pvalue), 'D2 should be NaN when no normalized innov supplied');
fprintf('PASSED (p=%.3e)\n', d2.innov_lb_pvalue);

% ── Test 3: biased residuals fail bias ────────────────────────────────────
fprintf('Test 3: biased N(1.0, 1.0) fails D3 ... ');
rng(44);
biased = 1.0 + randn(N, 1);

d3 = sigmatau.stats.kf_residual_diagnostics(randn(N, 1), biased);

assert(~d3.resid_bias_passed,       'D3: should fail for biased residuals');
assert(d3.resid_mean > 0.5,         'D3: mean should be positive (sanity)');
assert(d3.innov_lb_passed,          'D1: white innovations still pass whiteness');
fprintf('PASSED (mu=%.3f > se=%.3e)\n', d3.resid_mean, d3.resid_se);

% ── Test 4: argument validation ──────────────────────────────────────────
fprintf('Test 4: argument validation errors ... ');
try
    sigmatau.stats.kf_residual_diagnostics(1.0, 1.0);
    error('expected error: too short');
catch err
    assert(contains(err.identifier, 'tooShort'), 'wrong error id (too short): %s', err.identifier);
end
try
    sigmatau.stats.kf_residual_diagnostics(randn(10,1), randn(11,1));
    error('expected error: length mismatch');
catch err
    assert(contains(err.identifier, 'lenMismatch'), 'wrong error id (len): %s', err.identifier);
end
try
    sigmatau.stats.kf_residual_diagnostics(randn(10,1), randn(10,1), 'lag', 10);
    error('expected error: lag too large');
catch err
    assert(contains(err.identifier, 'lagTooLarge'), 'wrong error id (lag): %s', err.identifier);
end
fprintf('PASSED\n');

fprintf('test_kf_diagnostics: all assertions passed\n');
