%TEST_STATS  Unit tests for +stats functions.

% calculate_edf — WHFM (alpha=0), ADEV (d=2, F=m, S=1)
edf = sigmatau.stats.calculate_edf(0, 2, 1, 1, 1, 1000);
assert(isfinite(edf) && edf > 0, 'calculate_edf: WHFM should be positive finite');

% Invalid parameter: alpha+2d <= 1
edf_bad = sigmatau.stats.calculate_edf(-5, 2, 1, 1, 1, 100);
assert(isnan(edf_bad), 'calculate_edf: invalid alpha should return NaN');

% Not enough data
edf_short = sigmatau.stats.calculate_edf(0, 2, 100, 1, 1, 10);
assert(isnan(edf_short), 'calculate_edf: N<L should return NaN');

% totaldev_edf
edf_tot = sigmatau.stats.totaldev_edf('totvar', 0, 1000, 10);
assert(isfinite(edf_tot) && edf_tot > 0, 'totaldev_edf: totvar WHFM should be positive');
edf_mtot = sigmatau.stats.totaldev_edf('mtot', -1, 1000, 10);
assert(isfinite(edf_mtot) && edf_mtot > 0, 'totaldev_edf: mtot FLFM should be positive');
edf_htot = sigmatau.stats.totaldev_edf('htot', -2, 1000, 10);
assert(isfinite(edf_htot) && edf_htot > 0, 'totaldev_edf: htot RWFM should be positive');

% Unknown alpha → NaN
edf_unk = sigmatau.stats.totaldev_edf('totvar', 99, 1000, 10);
assert(isnan(edf_unk), 'totaldev_edf: unknown alpha should return NaN');

% bias — TOTVAR, MTOT
B_totvar = sigmatau.stats.bias([-1, -2, 0], 'totvar', [10, 10, 10], 1000);
assert(numel(B_totvar) == 3 && all(isfinite(B_totvar)), 'bias: totvar failed');
assert(B_totvar(3) == 1, 'bias: alpha=0 totvar should be 1');

B_mtot = sigmatau.stats.bias([0, -1, -2], 'mtot', [1,1,1], 100);
assert(all(B_mtot > 1), 'bias: mtot should be >1');

% edf() — result-struct dispatcher
result_edf = struct( ...
    'method',   'adev', ...
    'alpha',    [0, 0], ...
    'tau',      [1.0, 2.0], ...
    'tau0',     1.0,    ...
    'N',        1000    ...
);
edf_res = sigmatau.stats.edf(result_edf);
assert(isvector(edf_res) && numel(edf_res) == 2, 'edf: wrong output size');
assert(all(isfinite(edf_res) & edf_res > 0), 'edf: adev WHFM should be positive');

% edf() for total deviation methods
result_tot = struct( ...
    'method', 'totdev', 'alpha', [0], ...
    'tau', [10.0], 'tau0', 1.0, 'N', 1001 ...
);
edf_tot = sigmatau.stats.edf(result_tot);
assert(isfinite(edf_tot) && edf_tot > 0, 'edf: totdev should be positive');

% edf() unknown method → NaN (with warning)
result_unk = struct('method','unknown','alpha',[0],'tau',[1],'tau0',1,'N',100);
w = warning('off', 'SigmaTau:edf');
edf_unk = sigmatau.stats.edf(result_unk);
warning(w);
assert(isnan(edf_unk), 'edf: unknown method should return NaN');

% ci() — smoke test with edf field
result_ci = struct( ...
    'deviation',  [1e-12, 5e-13], ...
    'alpha',      [0, 0],         ...
    'edf',        [NaN, NaN],     ...
    'N',          1000,           ...
    'confidence', 0.683           ...
);
ci_mat = sigmatau.stats.ci(result_ci);
assert(isequal(size(ci_mat), [2, 2]), 'ci: wrong CI size');
assert(all(ci_mat(:,1) < [1e-12; 5e-13]), 'ci: lower CI should be < dev');
assert(all(ci_mat(:,2) > [1e-12; 5e-13]), 'ci: upper CI should be > dev');

% ci() with finite EDF (chi-squared path if toolbox available)
result_ci2 = struct( ...
    'deviation',  [1e-12], ...
    'alpha',      [0],     ...
    'edf',        [50],    ...
    'N',          1000,    ...
    'confidence', 0.683    ...
);
ci_mat2 = sigmatau.stats.ci(result_ci2);
assert(isequal(size(ci_mat2), [1, 2]), 'ci: chi2 path wrong size');
assert(ci_mat2(1,1) > 0 && ci_mat2(1,1) < 1e-12 && ci_mat2(1,2) > 1e-12, ...
       'ci: chi2 path interval order wrong');

fprintf('test_stats: all assertions passed\n');
