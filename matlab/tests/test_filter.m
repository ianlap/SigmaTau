%TEST_FILTER  Unit tests for MATLAB Kalman Filter implementation.

% ── Setup synthetic data ──────────────────────────────────────────────────
rng(2024);
N    = 1000;
tau0 = 1.0;
x_wfm = cumsum(randn(N, 1));

% ── Test 1: basic execution ────────────────────────────────────────────────
fprintf('Test 1: kalman_filter runs on white FM ... ');
config = struct( ...
    'q_wpm',  0,      ...
    'q_wfm',  1.0,    ...
    'q_rwfm', 0,      ...
    'R',      1.0,    ...
    'nstates', 3,     ...
    'tau',     tau0,  ...
    'g_p',     0,     ...
    'g_i',     0,     ...
    'g_d',     0,     ...
    'P0',      1e6    ...
);

result = sigmatau.kf.kalman_filter(x_wfm, config);

assert(numel(result.phase_est) == N, 'phase_est length mismatch');
assert(numel(result.freq_est) == N, 'freq_est length mismatch');
assert(numel(result.residuals) == N, 'residuals length mismatch');
assert(all(isfinite(result.phase_est)), 'non-finite phase_est');
fprintf('PASSED\n');

% ── Test 2: residual statistics ────────────────────────────────────────────
fprintf('Test 2: residuals have zero mean ... ');
mu = mean(result.residuals);
sig = std(result.residuals);
% Tol: 3 standard errors
assert(abs(mu) < 3 * sig / sqrt(N), 'residuals biased');
fprintf('PASSED\n');

% ── Test 3: covariance convergence ─────────────────────────────────────────
fprintf('Test 3: covariance P(1,1) converges ... ');
p11_final = result.P_history(1, 1, end);
assert(p11_final < config.P0 / 1000, 'covariance did not converge');
assert(p11_final > 0, 'covariance not positive');
fprintf('PASSED\n');

% ── Test 4: nstates=2 variant ──────────────────────────────────────────────
fprintf('Test 4: nstates=2 variant ... ');
config2 = config;
config2.nstates = 2;
result2 = sigmatau.kf.kalman_filter(x_wfm, config2);
assert(all(result2.drift_est == 0), 'drift_est should be zero for nstates=2');
fprintf('PASSED\n');

% ── Test 5: optimization ───────────────────────────────────────────────────
fprintf('Test 5: optimize runs without error ... ');
opt_res = sigmatau.kf.optimize(x_wfm, config);
assert(isfinite(opt_res.q_wfm), 'optimized q_wfm is NaN');
assert(opt_res.q_wfm > 0, 'optimized q_wfm <= 0');
fprintf('PASSED\n');

fprintf('test_filter: all assertions passed\n');
