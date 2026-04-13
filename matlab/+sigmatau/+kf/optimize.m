function [q_opt, results] = optimize(data, cfg)
% OPTIMIZE  Optimize KF process noise parameters via NLL + Nelder-Mead.
%
% Minimizes the negative log-likelihood (NLL) of the observed phase data
% under the linear Gaussian state-space model, using the innovation sequence:
%
%   NLL = 0.5 * sum_k [ log(S_k) + nu_k^2 / S_k ]
%
% where nu_k = z_k - H*x_{k|k-1} is the innovation and
%       S_k  = H*P_{k|k-1}*H' + R is the innovation variance.
%
% Optimization is over log10-space {q_wfm, q_rwfm[, q_irwfm]} using
% MATLAB's fminsearch (Nelder-Mead simplex). q_wpm (= R) is held fixed.
%
% Syntax:
%   [q_opt, results] = sigmatau.kf.optimize(data, cfg)
%
% Inputs:
%   data - phase measurements [N x 1 or 1 x N]
%   cfg  - struct with fields:
%          .q_wpm    - measurement noise variance R (fixed)
%          .q_wfm    - initial WFM guess
%          .q_rwfm   - initial RWFM guess
%          .q_irwfm  - initial IRWFM guess (default: 0; 0 = not optimized)
%          .nstates  - KF state dimension: 2 or 3 (default: 3)
%          .tau      - sampling interval [s]
%          .verbose  - print progress (default: true)
%          .max_iter - max fminsearch iterations (default: 500)
%          .tol      - convergence tolerance (default: 1e-6)
%
% Outputs:
%   q_opt   - struct: .q_wpm, .q_wfm, .q_rwfm, .q_irwfm (optimal values)
%   results - struct: .nll, .n_evals, .exitflag, .elapsed
%
% Example:
%   cfg = struct('q_wpm', 100, 'q_wfm', 0.01, 'q_rwfm', 1e-6, 'tau', 1.0);
%   [q_opt, res] = sigmatau.kf.optimize(phase_data, cfg);
%
% See also: sigmatau.kf.filter

cfg  = apply_defaults(cfg);
data = data(:);  % ensure column vector

if cfg.verbose
    fprintf('\n=== KF NLL OPTIMIZATION (Nelder-Mead) ===\n');
    fprintf('  q_wpm   = %.3e  (fixed, R)\n', cfg.q_wpm);
    fprintf('  q_wfm0  = %.3e\n',              cfg.q_wfm);
    fprintf('  q_rwfm0 = %.3e\n',              cfg.q_rwfm);
    if cfg.q_irwfm > 0
        fprintf('  q_irwfm0= %.3e\n',          cfg.q_irwfm);
    end
end

% Initial guess in log10 space
theta0 = [log10(cfg.q_wfm); log10(cfg.q_rwfm)];
if cfg.q_irwfm > 0
    theta0 = [theta0; log10(cfg.q_irwfm)];
end

obj  = @(th) kf_nll(th, data, cfg);
opts = optimset('MaxIter',      cfg.max_iter,         ...
                'MaxFunEvals',  cfg.max_iter * 200,   ...
                'TolX',         cfg.tol,              ...
                'TolFun',       cfg.tol,              ...
                'Display',      'off');

tic;
[theta_opt, nll_opt, exitflag, output] = fminsearch(obj, theta0, opts);
elapsed = toc;

q_opt.q_wpm   = cfg.q_wpm;
q_opt.q_wfm   = 10^theta_opt(1);
q_opt.q_rwfm  = 10^theta_opt(2);
if numel(theta_opt) >= 3
    q_opt.q_irwfm = 10^theta_opt(3);
else
    q_opt.q_irwfm = 0;
end

results.nll      = nll_opt;
results.n_evals  = output.funcCount;
results.exitflag = exitflag;
results.elapsed  = elapsed;

if cfg.verbose
    fprintf('  NLL = %.6f  (%d evals, %.2fs)\n', ...
            nll_opt, output.funcCount, elapsed);
    fprintf('  q_wfm   = %.3e\n', q_opt.q_wfm);
    fprintf('  q_rwfm  = %.3e\n', q_opt.q_rwfm);
    if q_opt.q_irwfm > 0
        fprintf('  q_irwfm = %.3e\n', q_opt.q_irwfm);
    end
end

end

% ── Local functions ───────────────────────────────────────────────────────────

function cfg = apply_defaults(cfg)
if ~isfield(cfg, 'q_irwfm'),  cfg.q_irwfm  = 0;    end
if ~isfield(cfg, 'nstates'),  cfg.nstates  = 3;    end
if ~isfield(cfg, 'verbose'),  cfg.verbose  = true; end
if ~isfield(cfg, 'max_iter'), cfg.max_iter = 500;  end
if ~isfield(cfg, 'tol'),      cfg.tol      = 1e-6; end

required = {'tau', 'q_wpm', 'q_wfm', 'q_rwfm'};
for i = 1:numel(required)
    f = required{i};
    if ~isfield(cfg, f)
        error('sigmatau:optimize:MissingField', 'cfg.%s is required', f);
    end
end

if ~ismember(cfg.nstates, [2, 3])
    error('sigmatau:optimize:InvalidNstates', 'cfg.nstates must be 2 or 3');
end
end

function nll = kf_nll(theta, data, cfg)
% Evaluate NLL of phase data under KF with log10-space parameters theta.
% No steering — innovations reflect pure filter fit to data.
N   = numel(data);
ns  = cfg.nstates;
tau = cfg.tau;
R   = cfg.q_wpm;

q_wfm   = 10^theta(1);
q_rwfm  = 10^theta(2);
q_irwfm = 0;
if numel(theta) >= 3
    q_irwfm = 10^theta(3);
end

% State-transition matrix Phi — matches filter.jl build_phi!
Phi = eye(ns);
if ns >= 2, Phi(1,2) = tau;         end
if ns >= 3, Phi(1,3) = tau^2 / 2;
            Phi(2,3) = tau;         end

H = zeros(1, ns);
H(1) = 1;  % observe phase only

% Process noise Q — SP1065 continuous-time model, matches filter.jl build_Q!
Q = build_Q_mat(ns, q_wfm, q_rwfm, q_irwfm, tau);

% LS initialization on first min(100, N-1) samples
n_fit = max(ns, min(100, N - 1));
t_fit = (0 : n_fit - 1)' * tau;
A_fit = build_A_mat(t_fit, ns);
x     = A_fit \ data(1:n_fit);

P   = 1e6 * eye(ns);
nll = 0;

for k = 1:N
    if k > 1
        x = Phi * x;
        P = Phi * P * Phi' + Q;
    end

    nu = data(k) - H * x;           % innovation (scalar)
    S  = H * P * H' + R;            % innovation variance (scalar)

    if S <= 0
        nll = 1e15;
        return;
    end

    % SP1065 Gaussian NLL contribution
    nll = nll + 0.5 * (log(S) + nu^2 / S);

    K = (P * H') / S;               % Kalman gain [ns x 1]
    x = x + K * nu;
    P = (eye(ns) - K * H) * P;
end
end

function Q = build_Q_mat(ns, q_wfm, q_rwfm, q_irwfm, tau)
% SP1065: continuous-time noise model integrated over tau.
% Matches filter.jl build_Q! exactly.
Q  = zeros(ns, ns);
t2 = tau^2; t3 = tau^3; t4 = tau^4; t5 = tau^5;

Q(1,1) = q_wfm*tau + q_rwfm*t3/3 + q_irwfm*t5/20;
if ns >= 2
    Q(1,2) = q_rwfm*t2/2 + q_irwfm*t4/8;
    Q(2,1) = Q(1,2);
    Q(2,2) = q_rwfm*tau  + q_irwfm*t3/3;
end
if ns >= 3
    Q(1,3) = q_irwfm*t3/6;
    Q(3,1) = Q(1,3);
    Q(2,3) = q_irwfm*t2/2;
    Q(3,2) = Q(2,3);
    Q(3,3) = q_irwfm*tau;
end
end

function A = build_A_mat(t, ns)
% Design matrix [1, t, t^2/2] for LS state initialization.
A = ones(numel(t), ns);
if ns >= 2, A(:,2) = t;         end
if ns >= 3, A(:,3) = t.^2 / 2; end
end
