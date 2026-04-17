---
name: kalman-filter
description: >
  Use when building, modifying, or debugging the Kalman filter, prediction,
  optimization, or pipeline functions. Trigger when working in +sigmatau/+kf/
  or julia/src/filter.jl, predict.jl, optimize.jl, pipeline.jl.
---

# Kalman Filter Architecture

## Design: Struct In, Struct Out

Legacy code has 14 input args and 14 return values. Refactored version uses structs.

### Config Struct
```matlab
cfg = struct(...
    'q_wfm', 0.01, ...      % White frequency modulation
    'q_rwfm', 1e-6, ...     % Random walk frequency modulation
    'R', 100, ...            % Measurement noise (= q_wpm)
    'g_p', 0.1, ...          % PID proportional gain
    'g_i', 0.01, ...         % PID integral gain
    'g_d', 0.05, ...         % PID derivative gain
    'nstates', 3, ...        % 2, 3, or 5
    'tau', 1.0, ...          % Sampling interval [s]
    'P0', 1e6, ...           % Initial covariance (scalar or matrix)
    'x0', [], ...            % Initial state ([] = auto-init)
    'q_irwfm', 0, ...        % Optional: integrated RWFM
    'q_diurnal', 0, ...      % Optional: diurnal (requires nstates=5)
    'period', 86400 ...      % Optional: diurnal period [s]
);
```

### Results Struct
```matlab
results.phase_est     % Estimated phase
results.freq_est      % Estimated frequency
results.drift_est     % Estimated drift
results.residuals     % Measurement residuals
results.innovations   % Kalman innovations
results.steers        % PID steering corrections
results.sumsteers     % Cumulative frequency steering
results.sum2steers    % Cumulative phase steering
results.P             % Covariance history (N x nstates x nstates)
results.config        % Echo of input config
```

## Pipeline: Composable Functions

The 47KB monolith `main_kf_pipeline_unified.m` becomes:
```matlab
function results = pipeline(mode, data_cfg, opt_cfg, out_cfg)
    data   = sigmatau.util.load_data(mode, data_cfg);
    dev    = sigmatau.dev.engine(data.phase, data.tau0, [], ...);
    noise  = sigmatau.noise.fit(dev);
    kf0    = sigmatau.kf.filter(data.phase, noise_to_config(noise));
    kf_opt = sigmatau.kf.optimize(data.phase, noise, opt_cfg);
    results = package(data, dev, noise, kf0, kf_opt);
end
```

The 800-line interactive prompting system → `examples/interactive_pipeline.m`.

## Gotchas
- PID integral accumulates phase error: `sumx += x(1)`, not frequency. Don't change.
- Covariance update uses `P = (I - K*H) * P` not full Joseph form. Matches MATLAB legacy.
- `safe_sqrt` handles negative diagonal elements from numerical drift — preserve this.
- Steering is applied AFTER Phi prediction, BEFORE P update.
- Q matrix elements have specific tau power dependencies (tau, tau^3/3, tau^5/20, etc.)
  from continuous-time noise model integration. These are exact — don't approximate.
