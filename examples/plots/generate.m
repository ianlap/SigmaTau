% Generate a reference white-FM phase series and compute ADEV with sigmatau.
% Dumps x and (tau, adev, ci_lo, ci_hi) so a Python script can overlay
% allantools oadev on the same data.
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'matlab')));

rng(12345);
N    = 8192;
tau0 = 1.0;
y    = randn(N-1, 1);                     % white FM frequency
x    = [0; cumsum(y)] * tau0;             % phase

m   = 2.^(0:floor(log2(N/8)));            % up to N/8 for stable adev
res = sigmatau.dev.adev(x, tau0, m);

out_dir = fileparts(mfilename('fullpath'));
writematrix(x,                           fullfile(out_dir, 'wfm_phase.csv'));
writematrix([res.tau(:), res.deviation(:), res.ci(:,1), res.ci(:,2)], ...
            fullfile(out_dir, 'sigmatau_adev.csv'));
fprintf('wrote wfm_phase.csv (N=%d) and sigmatau_adev.csv (%d taus)\n', ...
        N, numel(res.tau));
