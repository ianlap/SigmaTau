%TEST_NOISE_SLOPES  Verify ADEV noise slopes match theory.
%
% White PM (alpha=2): ADEV slope ~ tau^(-1)    → log-log slope ≈ -1
% White FM (alpha=0): ADEV slope ~ tau^(-1/2)  → log-log slope ≈ -0.5
% RWFM    (alpha=-2): ADEV slope ~ tau^(+1/2)  → log-log slope ≈ +0.5
%
% Tolerance: ±0.15 on log-log slope (statistical noise).

rng(12345);
N    = 8192;
tau0 = 1.0;
TOL  = 0.15;

% Power-law noise generator using the legacy function
% Since the legacy function uses rng('shuffle'), we use a simpler approach.
function y = gen_noise(alpha, N, tau0)
% Generate N-point frequency data with PSD ~ f^alpha using Kasdin method.
% Returns phase data (cumsum of frequency data).
Nf = N;
f  = (1:Nf/2)';
S  = f .^ (alpha/2);
phase = 2*pi*rand(Nf/2-1, 1);
half  = zeros(Nf/2+1, 1);
half(2:Nf/2) = S(1:end-1) .* exp(1j*phase);
full = [half; conj(flipud(half(2:Nf/2)))];
x_freq = real(ifft(full));
x_freq = (x_freq - mean(x_freq)) / std(x_freq);
y = cumsum(x_freq * tau0);   % phase
end

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

    x = gen_noise(alpha_true, N, tau0);

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
