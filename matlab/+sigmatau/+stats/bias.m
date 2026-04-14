function B = bias(alpha, var_type, tau, T)
% BIAS  Bias factor B(alpha) for TOTVAR, MTOT, and HTOT corrections.
%
%   B = sigmatau.stats.bias(alpha, var_type, tau, T)
%
%   Inputs:
%     alpha    – noise exponent (scalar or vector)
%     var_type – 'totvar' | 'mtot' | 'htot'
%     tau      – averaging time (scalar or vector, same size as alpha)
%     T        – record duration (s), (N-1)*tau0 (scalar)
%
%   Output:
%     B – bias factor for each tau. Divide deviation by B for unbiased estimate.
%
%   Reference: SP1065 §§5.2.11–5.2.13; FCS 2001 Table 1.

% SP1065 Table 11 — MTOT bias factors, indexed by alpha ∈ {-2,-1,0,1,2}
MTOT_B = [1.31, 1.30, 1.27, 1.17, 1.06];

% FCS 2001 Table 1 — HTOT a(alpha), indexed by alpha ∈ {-4,-3,-2,-1,0}; B = 1/(1+a)
HTOT_A = [-0.321, -0.283, -0.229, -0.149, -0.005];

FLICKER_FM_A = 1 / (3 * log(2));   % ≈ 0.481 — SP1065 Eq. 39, alpha=-1
RWFM_A       = 0.75;               % SP1065 Eq. 39, alpha=-2

alpha = double(alpha(:));
tau   = double(tau(:));
B     = ones(size(alpha));

switch lower(var_type)
    case 'totvar'
        for k = 1:numel(alpha)
            if alpha(k) == -1
                B(k) = 1 - FLICKER_FM_A * (tau(k) / T);
            elseif alpha(k) == -2
                B(k) = 1 - RWFM_A * (tau(k) / T);
            end
        end

    case 'mtot'
        for k = 1:numel(alpha)
            a_round = round(alpha(k));
            if a_round < -2 || a_round > 2
                warning('sigmatau:stats:bias:OutOfBounds', ...
                        'alpha=%g out of MTOT bounds [-2..2]. Clamping.', a_round);
                a_round = max(-2, min(2, a_round));
            end
            idx = a_round + 3;   % alpha ∈ {-2..2} → idx ∈ {1..5}
            B(k) = MTOT_B(idx);
        end

    case 'htot'
        for k = 1:numel(alpha)
            a_round = round(alpha(k));
            if a_round < -4 || a_round > 0
                warning('sigmatau:stats:bias:OutOfBounds', ...
                        'alpha=%g out of HTOT bounds [-4..0]. Clamping.', a_round);
                a_round = max(-4, min(0, a_round));
            end
            idx = a_round + 5;   % alpha ∈ {-4..0} → idx ∈ {1..5}
            B(k) = 1 / (1 + HTOT_A(idx));
        end

    otherwise
        error('SigmaTau:bias', ...
              'var_type must be ''totvar'', ''mtot'', or ''htot'', got ''%s''.', var_type);
end
end
