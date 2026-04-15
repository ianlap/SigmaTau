function result = kalman_filter(data, config)
% KALMAN_FILTER Run the Kalman filter on phase data.
% Struct In, Struct Out pattern.

    if nargin < 2
        config = struct();
    end
    
    data = data(:); % Ensure column vector
    
    config = apply_defaults(config);
    validate_config(config);
    
    if isempty(config.x0)
        config = initialize_state(config, data);
    end
    
    N = length(data);
    ns = config.nstates;
    tau = config.tau;
    R = config.R;
    g_p = config.g_p;
    g_i = config.g_i;
    g_d = config.g_d;
    period = config.period;
    twopi = 2 * pi;
    
    x = config.x0(:);
    
    if isscalar(config.P0)
        P = config.P0 * eye(ns);
    else
        P = config.P0;
    end
    P = (P + P') / 2.0; % enforce symmetry initially
    
    H = zeros(1, ns);
    H(1, 1) = 1.0; % measures phase only
    
    pid_state = zeros(2, 1); % [sumx; last_steer]
    
    phase_est = zeros(N, 1);
    freq_est = zeros(N, 1);
    drift_est = zeros(N, 1);
    residuals_v = zeros(N, 1);
    innov_v = zeros(N, 1);
    steers_v = zeros(N, 1);
    sumsteers_v = zeros(N, 1);
    sum2steer_v = zeros(N, 1);
    P_history = cell(N, 1);
    
    phase = data; 
    
    for k = 1:N
        Phi = sigmatau.kf.build_phi(ns, tau);
        Q = sigmatau.kf.build_Q(ns, config.q_wfm, config.q_rwfm, config.q_irwfm, config.q_diurnal, tau);
        
        if ns == 5
            H(1, 4) = sin(twopi * k / period);
            H(1, 5) = cos(twopi * k / period);
        end
        
        if k > 1
            % Predict step
            x = sigmatau.kf.predict_state(x, Phi, pid_state(2), tau, ns);
            P = sigmatau.kf.predict_covariance(P, Phi, Q);
            
            % Update working phase reference
            phase(k) = phase(k-1) + data(k) - data(k-1) + sumsteers_v(k-1);
        end
        
        % Innovation
        innov = phase(k) - H * x;
        
        % Kalman update
        S = H * P * H' + R;
        K = (P * H') / S;
        
        x = x + K * innov;
        P = (eye(ns) - K * H) * P;
        P = (P + P') / 2.0;
        
        % Guard diagonal against numerical drift
        for idx = 1:ns
            P(idx, idx) = safe_sqrt(P(idx, idx))^2;
        end
        
        % Posterior residual
        resid = phase(k) - x(1);
        
        % PID steering update
        [steer, pid_state] = sigmatau.kf.update_pid(pid_state, x, ns, g_p, g_i, g_d);
        
        % Cumulative steering
        if k == 1
            sumsteers_v(1) = pid_state(2);
            sum2steer_v(1) = sumsteers_v(1);
        else
            sumsteers_v(k) = sumsteers_v(k-1) + pid_state(2);
            sum2steer_v(k) = sum2steer_v(k-1) + sumsteers_v(k);
        end
        
        % Store results
        phase_est(k) = x(1);
        if ns >= 2, freq_est(k) = x(2); end
        if ns >= 3, drift_est(k) = x(3); end
        
        residuals_v(k) = resid;
        innov_v(k) = innov;
        steers_v(k) = steer;
        P_history{k} = P;
    end
    
    result.phase_est = phase_est;
    result.freq_est = freq_est;
    result.drift_est = drift_est;
    result.residuals = residuals_v;
    result.innovations = innov_v;
    result.steers = steers_v;
    result.sumsteers = sumsteers_v;
    result.sum2steers = sum2steer_v;
    result.P_history = P_history;
    result.config = config;
end

function x_out = safe_sqrt(v)
    if abs(v) < 1e-10
        x_out = 0.0;
    elseif v >= 0.0
        x_out = sqrt(v);
    else
        x_out = -sqrt(-v);
    end
end

function A = build_design_matrix(t, nstates, period)
    n = length(t);
    A = ones(n, nstates);
    if nstates >= 2
        A(:, 2) = t;
    end
    if nstates >= 3
        A(:, 3) = (t .^ 2) ./ 2.0;
    end
    if nstates == 5
        A(:, 4) = sin((2 * pi / period) .* t);
        A(:, 5) = cos((2 * pi / period) .* t);
    end
end

function config = initialize_state(config, data)
    N = length(data);
    n_fit = min(100, N - 1);
    n_fit = max(n_fit, config.nstates);
    
    if n_fit >= N
        error('Not enough data to initialize: need > %d samples', n_fit);
    end
    
    t = (0:n_fit-1)' .* config.tau;
    y = data(1:n_fit);
    
    A = build_design_matrix(t, config.nstates, config.period);
    
    coeffs = A \ y;
    config.x0 = coeffs(1:config.nstates);
    
    resid = y - A * coeffs;
    v = var(resid);
    
    config.P0 = v * inv(A' * A);
end

function config = apply_defaults(config)
    if ~isfield(config, 'q_wpm'), config.q_wpm = 100.0; end
    if ~isfield(config, 'q_wfm'), config.q_wfm = 0.01; end
    if ~isfield(config, 'q_rwfm'), config.q_rwfm = 1e-6; end
    if ~isfield(config, 'q_irwfm'), config.q_irwfm = 0.0; end
    if ~isfield(config, 'q_diurnal'), config.q_diurnal = 0.0; end
    if ~isfield(config, 'R'), config.R = 100.0; end
    if ~isfield(config, 'g_p'), config.g_p = 0.1; end
    if ~isfield(config, 'g_i'), config.g_i = 0.01; end
    if ~isfield(config, 'g_d'), config.g_d = 0.05; end
    if ~isfield(config, 'nstates'), config.nstates = 3; end
    if ~isfield(config, 'tau'), config.tau = 1.0; end
    if ~isfield(config, 'P0'), config.P0 = 1e6; end
    if ~isfield(config, 'x0'), config.x0 = []; end
    if ~isfield(config, 'period'), config.period = 86400.0; end
end

function validate_config(config)
    if ~ismember(config.nstates, [2, 3, 5])
        error('nstates must be 2, 3, or 5');
    end
    if ~isempty(config.x0) && length(config.x0) ~= config.nstates
        error('x0 length does not match nstates');
    end
    if length(config.P0(:)) > 1 && ~isequal(size(config.P0), [config.nstates, config.nstates])
        error('P0 matrix size != (nstates, nstates)');
    end
    if config.q_wpm <  0, error('q_wpm < 0'); end
    if config.q_wfm <  0, error('q_wfm < 0'); end
    if config.q_rwfm <  0, error('q_rwfm < 0'); end
    if config.q_irwfm <  0, error('q_irwfm < 0'); end
    if config.q_diurnal <  0, error('q_diurnal < 0'); end
    if config.q_diurnal > 0 && config.nstates ~= 5, error('q_diurnal > 0 requires nstates=5'); end
    if config.tau <= 0, error('tau must be > 0'); end
end
