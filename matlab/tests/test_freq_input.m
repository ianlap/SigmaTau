%TEST_FREQ_INPUT  Verify frequency data input gives same result as phase input.
%
% Conversion: phase = cumsum(freq) * tau0
% So: adev(freq, tau0, m, 'data_type','freq') == adev(cumsum(freq)*tau0, tau0, m)

rng(7);
N    = 512;
tau0 = 0.1;
freq = randn(N, 1);   % white FM
phase = cumsum(freq) * tau0;

m_list = [1, 2, 4, 8];

devfns = {
    @sigmatau.dev.adev,    'adev';
    @sigmatau.dev.mdev,    'mdev';
    @sigmatau.dev.tdev,    'tdev';
    @sigmatau.dev.hdev,    'hdev';
    @sigmatau.dev.mhdev,   'mhdev';
    @sigmatau.dev.ldev,    'ldev';
};

for k = 1:size(devfns,1)
    fn   = devfns{k,1};
    name = devfns{k,2};

    r_phase = fn(phase, tau0, m_list);
    r_freq  = fn(freq,  tau0, m_list, 'data_type', 'freq');

    % Compare deviation values
    err = abs(r_phase.deviation - r_freq.deviation);
    rel = err ./ max(abs(r_phase.deviation), 1e-30);

    assert(all(rel < 1e-12 | isnan(r_phase.deviation)), ...
           sprintf('%s: freq input differs from phase input (max rel err = %.2e)', ...
                   name, max(rel(~isnan(rel)))));
    fprintf('  OK: %s freq/phase consistency (max rel err = %.2e)\n', ...
            name, max(rel(~isnan(rel))));
end

fprintf('test_freq_input: all assertions passed\n');
