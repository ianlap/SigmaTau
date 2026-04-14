% basic_usage.m — Simple example of using SigmaTau for stability analysis.

% 1. Setup environment
addpath(genpath('matlab'));
rng(42);

% 2. Generate sample phase data (White FM: phase is a random walk)
N    = 1024;
tau0 = 1.0;
y    = randn(N-1, 1);           % white frequency noise
x    = [0; cumsum(y)] * tau0;    % integrate to phase

fprintf('SigmaTau Basic Usage (MATLAB)\n');
fprintf('----------------------------\n');
fprintf('Generated %d samples (tau0 = %.1fs)\n\n', N, tau0);

% 3. Compute Overlapping Allan Deviation (ADEV)
fprintf('Computing ADEV...\n');
res_adev = sigmatau.dev.adev(x, tau0);

% 4. Compute Modified Allan Deviation (MDEV)
fprintf('Computing MDEV...\n');
res_mdev = sigmatau.dev.mdev(x, tau0);

% 5. Display results
fprintf('%10s | %15s | %15s | %10s\n', 'Tau [s]', 'ADEV', 'MDEV', 'Alpha (ID)');
fprintf('%s\n', repmat('-', 1, 60));

for i = 1:numel(res_adev.tau)
    % Both use the same default m_list, so we can align them
    fprintf('%10.1f | %15.6e | %15.6e | %10d\n', ...
            res_adev.tau(i), res_adev.deviation(i), res_mdev.deviation(i), res_adev.alpha(i));
end

% 6. Accessing Confidence Intervals
fprintf('\nConfidence Intervals (ADEV, 68.3%%):\n');
for i = 1:min(3, numel(res_adev.tau))
    fprintf('  tau = %5.1f: [%.4e, %.4e]\n', ...
            res_adev.tau(i), res_adev.ci(i, 1), res_adev.ci(i, 2));
end

fprintf('\nDone.\n');
