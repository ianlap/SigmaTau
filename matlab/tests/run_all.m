%RUN_ALL  Run all SigmaTau MATLAB tests and report results.
%
%   cd matlab && matlab -batch "addpath(genpath('.')); run('tests/run_all.m')"

addpath(genpath(fileparts(mfilename('fullpath')) + "/../"));

tests = {
    'test_util',
    'test_stats',
    'test_noise',
    'test_deviations',
    'test_noise_slopes',
    'test_freq_input',
    'test_crossval_julia',
};

passed = 0;
failed = 0;
failed_names = {};

for k = 1:numel(tests)
    name = tests{k};
    fprintf('\n=== %s ===\n', name);
    try
        run(name);
        fprintf('PASS: %s\n', name);
        passed = passed + 1;
    catch err
        fprintf('FAIL: %s\n  %s\n', name, err.message);
        failed = failed + 1;
        failed_names{end+1} = name;
    end
end

fprintf('\n========================================\n');
fprintf('Results: %d passed, %d failed\n', passed, failed);
if failed > 0
    fprintf('Failed tests:\n');
    for k = 1:numel(failed_names)
        fprintf('  - %s\n', failed_names{k});
    end
    error('SigmaTau:test', '%d test(s) failed.', failed);
end
fprintf('All tests passed.\n');
