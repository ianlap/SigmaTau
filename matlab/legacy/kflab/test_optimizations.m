%% COMPARE_OPTIMIZATION_METHODS - Test script to compare optimize_kf vs optimize_kf_dare
%
% This script compares the original optimize_kf with the new optimize_kf_dare
% to verify they give similar results and measure performance differences.

clear; clc;

fprintf('=== OPTIMIZATION METHOD COMPARISON ===\n');
fprintf('Comparing optimize_kf (original) vs optimize_kf_dare (DARE method)\n\n');

%% Load or generate test data
% Replace this section with your actual data loading
if exist('phase_data', 'var') && exist('tau', 'var')
    fprintf('Using existing phase_data and tau from workspace\n');
else
    % Generate synthetic data if real data not available
    fprintf('No phase_data found - generating synthetic test data\n');
    tau = 1.0;  % 1 second sampling
    N = 100000;  % 100k samples
    t = (0:N-1) * tau;
    
    % Simple synthetic phase data with noise
    phase_data = 0.1*randn(N,1) + 0.01*cumsum(randn(N,1)) + 0.001*cumsum(cumsum(randn(N,1)));
    
    fprintf('Generated %d samples with tau=%.1f seconds\n', N, tau);
end

fprintf('Data length: %d samples (%.1f hours)\n', length(phase_data), length(phase_data)*tau/3600);

%% Define test parameters
q_initial = struct();
q_initial.q_wpm = 100;      % Fixed measurement noise
q_initial.q_wfm = 0.01;     % Initial guess for WFM
q_initial.q_rwfm = 1e-6;    % Initial guess for RWFM
q_initial.q_irwfm = 1e-9;   % Initial guess for IRWFM (small but non-zero)

% Test configuration - start with small grid for speed
opt_config = struct();
opt_config.search_range = 1;           % ±1 decade (faster test)
opt_config.n_grid_per_decade = 1;     % 1 points per decade (faster test)
opt_config.target_horizons = [10 100 1000];
opt_config.horizon_weights = [1 1 1]; % Equal weights
opt_config.nstates = 3;               % Use 3-state filter
opt_config.maturity = 5000;           % Shorter maturity for speed
opt_config.method = 'grid';           % Use grid search
opt_config.verbose = true;

fprintf('\nTest configuration:\n');
fprintf('  Search range: ±%d decades\n', opt_config.search_range);
fprintf('  Grid density: %d points per decade\n', opt_config.n_grid_per_decade);
fprintf('  Target horizons: [%s] samples\n', num2str(opt_config.target_horizons));
fprintf('  States: %d\n', opt_config.nstates);
fprintf('  Maturity: %d samples\n', opt_config.maturity);

total_points = (2 * opt_config.search_range * opt_config.n_grid_per_decade + 1)^2;
if q_initial.q_irwfm > 0
    total_points = total_points * (2 * opt_config.search_range * opt_config.n_grid_per_decade + 1);
end
fprintf('  Total grid points: %d\n', total_points);

%% Run original optimization
fprintf('\n--- RUNNING ORIGINAL OPTIMIZE_KF ---\n');
tic;
try
    [q_opt_original, results_original] = optimize_kf(phase_data, tau, q_initial, opt_config);
    time_original = toc;
    original_success = true;
    fprintf('Original optimization completed in %.1f seconds\n', time_original);
catch ME
    time_original = toc;
    original_success = false;
    fprintf('Original optimization FAILED after %.1f seconds: %s\n', time_original, ME.message);
    q_opt_original = struct();
    results_original = struct();
end

%% Run DARE optimization
fprintf('\n--- RUNNING DARE OPTIMIZE_KF_DARE ---\n');
tic;
try
    [q_opt_dare, results_dare] = optimize_kf_dare(phase_data, tau, q_initial, opt_config);
    time_dare = toc;
    dare_success = true;
    fprintf('DARE optimization completed in %.1f seconds\n', time_dare);
catch ME
    time_dare = toc;
    dare_success = false;
    fprintf('DARE optimization FAILED after %.1f seconds: %s\n', time_dare, ME.message);
    q_opt_dare = struct();
    results_dare = struct();
end

%% Compare results
fprintf('\n=== COMPARISON RESULTS ===\n');

if original_success && dare_success
    fprintf('\n--- PERFORMANCE COMPARISON ---\n');
    fprintf('Original time:  %.1f seconds\n', time_original);
    fprintf('DARE time:      %.1f seconds\n', time_dare);
    if time_original > time_dare
        speedup = time_original / time_dare;
        fprintf('DARE is %.1fx FASTER\n', speedup);
    else
        slowdown = time_dare / time_original;
        fprintf('DARE is %.1fx SLOWER\n', slowdown);
    end
    
    fprintf('\n--- OPTIMAL PARAMETERS ---\n');
    fprintf('Parameter     Original        DARE           Ratio\n');
    fprintf('q_wfm         %.3e      %.3e      %.3f\n', ...
            q_opt_original.q_wfm, q_opt_dare.q_wfm, q_opt_dare.q_wfm/q_opt_original.q_wfm);
    fprintf('q_rwfm        %.3e      %.3e      %.3f\n', ...
            q_opt_original.q_rwfm, q_opt_dare.q_rwfm, q_opt_dare.q_rwfm/q_opt_original.q_rwfm);
    fprintf('q_irwfm       %.3e      %.3e      %.3f\n', ...
            q_opt_original.q_irwfm, q_opt_dare.q_irwfm, q_opt_dare.q_irwfm/q_opt_original.q_irwfm);
    
    fprintf('\n--- COST FUNCTION VALUES ---\n');
    fprintf('Original weighted RMS: %.3e\n', results_original.weighted_rms);
    fprintf('DARE weighted RMS:     %.3e\n', results_dare.weighted_rms);
    fprintf('Ratio (DARE/Original): %.3f\n', results_dare.weighted_rms / results_original.weighted_rms);
    
    if isfield(results_original, 'rms_opt') && isfield(results_dare, 'rms_opt')
        fprintf('\n--- RMS AT EACH HORIZON ---\n');
        fprintf('Horizon    Original        DARE           Ratio\n');
        for i = 1:length(opt_config.target_horizons)
            h = opt_config.target_horizons(i);
            rms_orig = results_original.rms_opt(i);
            rms_dare = results_dare.rms_opt(i);
            fprintf('%7d    %.3e      %.3e      %.3f\n', ...
                    h, rms_orig, rms_dare, rms_dare/rms_orig);
        end
    end
    
    fprintf('\n--- EVALUATION COUNT ---\n');
    fprintf('Original evaluations: %d\n', results_original.n_evaluations);
    fprintf('DARE evaluations:     %d\n', results_dare.n_evaluations);
    
    fprintf('\n--- AGREEMENT ASSESSMENT ---\n');
    param_agreement = abs(log10(q_opt_dare.q_wfm / q_opt_original.q_wfm)) < 0.1 && ...
                     abs(log10(q_opt_dare.q_rwfm / q_opt_original.q_rwfm)) < 0.1;
    cost_agreement = abs(results_dare.weighted_rms / results_original.weighted_rms - 1) < 0.05;
    
    if param_agreement && cost_agreement
        fprintf('✓ METHODS AGREE: Parameters and costs are very similar\n');
    elseif param_agreement
        fprintf('△ PARTIAL AGREEMENT: Parameters similar, but costs differ\n');
    elseif cost_agreement
        fprintf('△ PARTIAL AGREEMENT: Costs similar, but parameters differ\n');
    else
        fprintf('✗ METHODS DISAGREE: Significant differences in results\n');
    end
    
elseif original_success
    fprintf('Only ORIGINAL method succeeded\n');
    fprintf('Original time: %.1f seconds\n', time_original);
    fprintf('Optimal q_wfm: %.3e, q_rwfm: %.3e, q_irwfm: %.3e\n', ...
            q_opt_original.q_wfm, q_opt_original.q_rwfm, q_opt_original.q_irwfm);
    
elseif dare_success
    fprintf('Only DARE method succeeded\n');
    fprintf('DARE time: %.1f seconds\n', time_dare);
    fprintf('Optimal q_wfm: %.3e, q_rwfm: %.3e, q_irwfm: %.3e\n', ...
            q_opt_dare.q_wfm, q_opt_dare.q_rwfm, q_opt_dare.q_irwfm);
    
else
    fprintf('Both methods FAILED\n');
    fprintf('Check your data and function implementations\n');
end

%% Test individual prediction methods for verification
if original_success || dare_success
    fprintf('\n=== PREDICTION METHOD VERIFICATION ===\n');
    
    % Use the better result (or original if both succeeded)
    if original_success
        test_q = q_opt_original;
        fprintf('Testing prediction methods with original optimal parameters\n');
    else
        test_q = q_opt_dare;
        fprintf('Testing prediction methods with DARE optimal parameters\n');
    end
    
    % Set up prediction test
    noise_test = test_q;
    kf_test = struct('nstates', opt_config.nstates, 'maturity', opt_config.maturity, 'max_horizon', 1000);
    pred_test = struct('g_p', 0, 'g_i', 0, 'g_d', 0, 'init_cov', 1e30, 'verbose', false);
    
    try
        fprintf('Running kf_predict (original)...\n');
        tic;
        results_kf_orig = kf_predict(phase_data, tau, noise_test, kf_test, pred_test);
        time_kf_orig = toc;
        fprintf('  Completed in %.2f seconds\n', time_kf_orig);
        
        fprintf('Running kf_predict_dare...\n');
        tic;
        results_kf_dare = kf_predict_dare(phase_data, tau, noise_test, kf_test, pred_test);
        time_kf_dare = toc;
        fprintf('  Completed in %.2f seconds\n', time_kf_dare);
        
        % Compare prediction RMS
        test_horizons = [10, 100, 1000];
        fprintf('\nPrediction RMS comparison:\n');
        fprintf('Horizon    kf_predict      kf_predict_dare   Ratio\n');
        for h = test_horizons
            idx_orig = find(results_kf_orig.rms_stats.horizon == h);
            idx_dare = find(results_kf_dare.rms_stats.horizon == h);
            if ~isempty(idx_orig) && ~isempty(idx_dare)
                rms_orig = results_kf_orig.rms_stats.rms_error(idx_orig);
                rms_dare = results_kf_dare.rms_stats.rms_error(idx_dare);
                fprintf('%7d    %.3e        %.3e       %.3f\n', ...
                        h, rms_orig, rms_dare, rms_dare/rms_orig);
            end
        end
        
        if time_kf_orig > time_kf_dare
            pred_speedup = time_kf_orig / time_kf_dare;
            fprintf('\nkf_predict_dare is %.1fx faster for prediction\n', pred_speedup);
        else
            pred_slowdown = time_kf_dare / time_kf_orig;
            fprintf('\nkf_predict_dare is %.1fx slower for prediction\n', pred_slowdown);
        end
        
    catch ME
        fprintf('Prediction verification failed: %s\n', ME.message);
    end
end

fprintf('\n=== SUMMARY ===\n');
if original_success && dare_success
    fprintf('Both methods completed successfully\n');
    if time_dare < time_original
        fprintf('DARE method is faster and should be preferred for optimization\n');
    else
        fprintf('Original method is faster - investigate DARE implementation\n');
    end
else
    fprintf('One or both methods failed - check implementations\n');
end

fprintf('\nComparison complete!\n');