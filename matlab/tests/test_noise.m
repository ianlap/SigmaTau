%TEST_NOISE  Unit tests for +noise/identify and +noise/generate.

% ── generate: basic sanity checks ─────────────────────────────────────────────

rng(42);

% generate returns a column vector of the requested length
x = sigmatau.noise.generate(0, 1024);
assert(isvector(x) && numel(x) == 1024, 'generate: wrong output length');
assert(all(isfinite(x)), 'generate: output contains non-finite values');

% N must be even
try
    sigmatau.noise.generate(0, 101);
    error('generate: should have thrown for odd N');
catch err
    assert(~isempty(strfind(err.message, 'even')) || contains(err.message, 'even'), ...
           'generate: wrong error for odd N');
end

% tau0 scaling: std of second-differences scales with tau0
rng(1);
x1 = sigmatau.noise.generate(0, 2048, 1.0);
rng(1);
x2 = sigmatau.noise.generate(0, 2048, 2.0);
assert(abs(std(diff(diff(x2))) / std(diff(diff(x1))) - 2.0) < 0.1, ...
       'generate: tau0 scaling error');

% ── generate: noise slope check ───────────────────────────────────────────────
% Coarse tolerance (±0.20) — statistical; seed is fixed for reproducibility.

rng(99);
N_gen = 8192;
tau0  = 1.0;
TOL   = 0.20;

configs = {
    2,  -1.0,  'White PM  (alpha=2)';
    0,  -0.5,  'White FM  (alpha=0)';
   -2,   0.5,  'RWFM      (alpha=-2)';
};
m_list = 2.^(0:8);

for k = 1:size(configs, 1)
    alpha_true = configs{k, 1};
    slope_exp  = configs{k, 2};
    label      = configs{k, 3};

    x    = sigmatau.noise.generate(alpha_true, N_gen, tau0);
    res  = sigmatau.dev.adev(x, tau0, m_list);

    valid = isfinite(res.deviation) & res.deviation > 0;
    assert(sum(valid) >= 3, ...
           sprintf('generate slope test: too few valid points for %s', label));

    p     = polyfit(log10(res.tau(valid)), log10(res.deviation(valid)), 1);
    slope = p(1);
    fprintf('  generate+adev %s: slope=%.3f expected≈%.1f\n', label, slope, slope_exp);
    assert(abs(slope - slope_exp) < TOL, ...
           sprintf('generate slope: %s got %.3f, expected %.1f±%.2f', ...
                   label, slope, slope_exp, TOL));
end

% ── identify: lag-1 ACF path (N_eff >= 30) ────────────────────────────────────

rng(7);
N_id = 4096;

% White FM phase data → expect alpha ≈ 0
x_wfm = sigmatau.noise.generate(0, N_id);
a_wfm = sigmatau.noise.noise_id(x_wfm, [1], 'phase');
assert(~isnan(a_wfm), 'noise_id: returned NaN for WFM');
assert(round(a_wfm) == 0, sprintf('noise_id: WFM alpha=%.2f, expected ~0', a_wfm));

% RWFM phase data → expect alpha ≈ -2
x_rwfm = sigmatau.noise.generate(-2, N_id);
a_rwfm = sigmatau.noise.noise_id(x_rwfm, [1], 'phase');
assert(~isnan(a_rwfm), 'noise_id: returned NaN for RWFM');
assert(round(a_rwfm) == -2, sprintf('noise_id: RWFM alpha=%.2f, expected ~-2', a_rwfm));

% ── identify: B1/Rn fallback (N_eff < 30) ─────────────────────────────────────

% Provide 60 points; use m=3 → N_eff=20 < 30 → B1/Rn path
rng(13);
x_small = sigmatau.noise.generate(0, 512);
a_small = sigmatau.noise.noise_id(x_small, [20], 'phase');
assert(~isnan(a_small), 'noise_id B1/Rn: should return a value');
assert(isnumeric(a_small) && isscalar(a_small), 'noise_id B1/Rn: wrong output type');

% ── identify: edge cases ──────────────────────────────────────────────────────

% Constant data → should return NaN (not error)
x_const = ones(100, 1);
a_const = sigmatau.noise.noise_id(x_const, [1]);
assert(isnan(a_const), 'noise_id: constant data should yield NaN');

% ── identify: no spurious α=2 at long τ (regression test) ────────────────────
% Before the B1/R(n) m² fix and carry-forward policy, every noise type except
% WHPM spuriously returned α=2 once N_eff dropped below the lag-1 ACF threshold
% (B1's R(n) discrimination degenerated). Assert that red-spectrum inputs
% (WHFM, RWFM) do not default to +2 at the tail.

rng(50);
N_cf = 2^14;                                         % 16384
ms   = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];  % tail N_eff: 128 → 16

% RWFM: tail α must be firmly negative (bug would give α=+2)
x_rwfm_cf = sigmatau.noise.generate(-2, N_cf);
a_rwfm_cf = sigmatau.noise.noise_id(x_rwfm_cf, ms, 'phase');
assert(round(a_rwfm_cf(end)) <= -1, ...
       sprintf('regression: RWFM tail α=%.2f, expected ≤-1', a_rwfm_cf(end)));

% WHFM: tail α must be ≤0 (WHFM or FLFM — both defensible at the boundary)
x_wfm_cf  = sigmatau.noise.generate(0, N_cf);
a_wfm_cf  = sigmatau.noise.noise_id(x_wfm_cf, ms, 'phase');
assert(round(a_wfm_cf(end)) <= 0, ...
       sprintf('regression: WFM tail α=%.2f, expected ≤0', a_wfm_cf(end)));

fprintf('test_noise: all assertions passed\n');
