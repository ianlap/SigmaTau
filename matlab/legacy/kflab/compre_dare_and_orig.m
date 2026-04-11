% Test both methods with identical parameters
noise_params = struct('q_wpm', 100, 'q_wfm', 0.01, 'q_rwfm', 1e-6, 'q_irwfm', 0);
kf_params = struct('nstates', 3, 'maturity', 5000, 'max_horizon', 1000);
pred_params = struct('g_p', 0, 'g_i', 0, 'g_d', 0, 'init_cov', 1e30, 'verbose', true);
tau=1;
% Run both
tic;
results_original = kf_predict(phase_data, tau, noise_params, kf_params, pred_params);
toc;
tic;
results_dare = kf_predict_dare(phase_data, tau, noise_params, kf_params, pred_params);
toc;
% Compare RMSE at key horizons
horizons = [10, 100, 1000];
for h = horizons
    idx_orig = find(results_original.rms_stats.horizon == h);
    idx_dare = find(results_dare.rms_stats.horizon == h);
    
    if ~isempty(idx_orig) && ~isempty(idx_dare)
        rms_orig = results_original.rms_stats.rms_error(idx_orig);
        rms_dare = results_dare.rms_stats.rms_error(idx_dare);
        fprintf('Horizon %d: Original=%.3e, DARE=%.3e, Ratio=%.3f\n', ...
                h, rms_orig, rms_dare, rms_dare/rms_orig);
    end
end