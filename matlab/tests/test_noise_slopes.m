%TEST_NOISE_SLOPES  Verify ADEV noise slopes match theory.
%
% White PM (alpha=2): ADEV slope ~ tau^(-1)    → log-log slope ≈ -1
% White FM (alpha=0): ADEV slope ~ tau^(-1/2)  → log-log slope ≈ -0.5
% RWFM    (alpha=-2): ADEV slope ~ tau^(+1/2)  → log-log slope ≈ +0.5
%
% Tolerance: ±0.15 on log-log slope (statistical noise).

rng(12345);
N    = 16384;
tau0 = 1.0;
TOL  = 0.15;

configs = {
    2,  -1.0,  'White PM  (alpha=2)';
    0,  -0.5,  'White FM  (alpha=0)';
   -2,   0.5,  'RWFM      (alpha=-2)';
};

m_list = 2.^(0:8);

for k = 1:size(configs,1)
    alpha_true = configs{k,1};
    slope_exp  = configs{k,2};
    label      = configs{k,3};

    x = sigmatau.noise.generate(alpha_true, N, tau0);

    result = sigmatau.dev.adev(x, tau0, m_list);

    % Log-log linear fit on finite values
    valid = isfinite(result.deviation) & result.deviation > 0;
    if sum(valid) < 3
        error('test_noise_slopes: too few valid points for %s', label);
    end
    lt = log10(result.tau(valid));
    ld = log10(result.deviation(valid));
    p  = polyfit(lt, ld, 1);
    slope_obs = p(1);

    fprintf('  %s: slope observed = %.3f, expected ≈ %.1f\n', label, slope_obs, slope_exp);
    assert(abs(slope_obs - slope_exp) < TOL, ...
           sprintf('%s: slope %.3f not within %.2f of expected %.1f', ...
                   label, slope_obs, TOL, slope_exp));
end

fprintf('test_noise_slopes: all slope checks passed\n');
