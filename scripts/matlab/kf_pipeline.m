function results = kf_pipeline(filename, tau0)
% KF_PIPELINE  MATLAB Kalman filter pipeline.
%
%   results = kf_pipeline(filename, tau0)
%
%   Stages:
%     1. Load phase data.
%     2. Characterize noise with MHDEV.
%     3. Run initial KF.
%     4. Optimize Q components.
%     5. Final KF run and reporting.

    addpath(genpath('matlab'));
    
    if nargin < 2 || isempty(tau0)
        tau0 = 1.0;
    end
    
    %-- Load data
    fprintf('\nSigmaTau: Kalman Filter Pipeline (MATLAB)\n');
    fprintf('File: %s\n', filename);
    try
        x = dlmread(filename);
        x = x(:);
        N = numel(x);
    catch ME
        error('Failed to read file: %s', ME.message);
    end
    fprintf('  Loaded %d phase samples  (tau0 = %.3f s)\n', N, tau0);

    %-- Stage 2: Characterize noise
    fprintf('Stage 2: Characterizing noise (MHDEV) ... ');
    mh = sigmatau.dev.mhdev(x, tau0);
    fprintf('done.\n');
    
    %-- Stage 3: Initial parameters
    % Simple fallback for demonstration
    q_wpm0  = mh.deviation(1)^2;
    q_wfm0  = q_wpm0 * 1e-4;
    q_rwfm0 = q_wpm0 * 1e-8;
    
    fprintf('  Initial parameters:\n');
    fprintf('    q_wpm  = %.4e\n', q_wpm0);
    fprintf('    q_wfm  = %.4e\n', q_wfm0);
    fprintf('    q_rwfm = %.4e\n', q_rwfm0);

    %-- Stage 4: Run initial KF
    config = struct( ...
        'q_wpm',  q_wpm0,  ...
        'q_wfm',  q_wfm0,  ...
        'q_rwfm', q_rwfm0, ...
        'R',      q_wpm0,  ...
        'nstates', 3,      ...
        'tau',     tau0,   ...
        'g_p',     0,      ...
        'g_i',     0,      ...
        'g_d',     0,      ...
        'P0',      1e6     ...
    );
    
    fprintf('Stage 3: Running initial KF ... ');
    kf_res = sigmatau.kf.kalman_filter(x, config);
    fprintf('done.\n');
    
    rms_init = sqrt(mean(kf_res.innovations(round(0.1*N):end).^2));
    fprintf('  Initial Innovation RMS: %.4e\n', rms_init);

    %-- Stage 5: Optimization
    fprintf('Stage 4: Optimizing parameters (Nelder-Mead) ... ');
    opt_res = sigmatau.kf.optimize(x, config);
    fprintf('done.\n');
    
    fprintf('  Optimized parameters:\n');
    fprintf('    q_wpm  = %.4e\n', opt_res.q_wpm);
    fprintf('    q_wfm  = %.4e\n', opt_res.q_wfm);
    fprintf('    q_rwfm = %.4e\n', opt_res.q_rwfm);

    %-- Stage 6: Final KF
    config.q_wpm  = opt_res.q_wpm;
    config.q_wfm  = opt_res.q_wfm;
    config.q_rwfm = opt_res.q_rwfm;
    config.R      = opt_res.q_wpm;
    
    fprintf('Stage 5: Running final KF ... ');
    kf_final = sigmatau.kf.kalman_filter(x, config);
    fprintf('done.\n');
    
    rms_final = sqrt(mean(kf_final.innovations(round(0.1*N):end).^2));
    fprintf('  Final Innovation RMS: %.4e\n', rms_final);
    
    %-- Results
    results = struct( ...
        'initial_q', [q_wpm0, q_wfm0, q_rwfm0], ...
        'final_q',   [opt_res.q_wpm, opt_res.q_wfm, opt_res.q_rwfm], ...
        'rms_init',  rms_init, ...
        'rms_final', rms_final, ...
        'kf_res',    kf_final ...
    );
end
