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

% bias_correction
B_totvar = sigmatau.stats.bias_correction([-1, -2, 0], 'totvar', [10, 10, 10], 1000);
assert(numel(B_totvar) == 3 && all(isfinite(B_totvar)), 'bias_correction: totvar failed');
assert(B_totvar(3) == 1, 'bias_correction: alpha=0 totvar should be 1');

B_mtot = sigmatau.stats.bias_correction([0, -1, -2], 'mtot', [1,1,1], 100);
assert(all(B_mtot > 1), 'bias_correction: mtot should be >1');

% compute_ci — smoke test via a simple result struct
result = struct( ...
    'deviation',  [1e-12, 5e-13], ...
    'alpha',      [0, 0],         ...
    'N',          1000,           ...
    'confidence', 0.683           ...
);
ci = sigmatau.stats.compute_ci(result);
assert(isequal(size(ci), [2, 2]), 'compute_ci: wrong CI size');
assert(all(ci(:,1) < [1e-12; 5e-13]), 'compute_ci: lower CI should be < dev');
assert(all(ci(:,2) > [1e-12; 5e-13]), 'compute_ci: upper CI should be > dev');

fprintf('test_stats: all assertions passed\n');
