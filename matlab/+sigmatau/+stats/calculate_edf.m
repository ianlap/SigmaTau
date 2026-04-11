function edf = calculate_edf(alpha, d, m, F, S, N)
% CALCULATE_EDF  Equivalent degrees of freedom for overlapping variance estimator.
%
%   edf = sigmatau.stats.calculate_edf(alpha, d, m, F, S, N)
%
%   Inputs:
%     alpha – power-law noise exponent (-4 to 2)
%     d     – phase difference order (2 = Allan, 3 = Hadamard)
%     m     – averaging factor tau/tau0
%     F     – filter factor (m for unmodified, 1 for modified)
%     S     – stride (1 overlapping, m non-overlapping)
%     N     – number of phase data points
%
%   Reference: Greenhall & Riley, PTTI 2003, Eq. 4–10.

if alpha + 2*d <= 1
    edf = NaN;
    return;
end

L = m/F + m*d;          % filter length
if N < L
    edf = NaN;
    return;
end

M = 1 + floor(S*(N - L)/m);     % number of summands
J = min(M, (d + 1)*S);          % truncation parameter

sz0 = compute_sz(0, F, alpha, d);
if isnan(sz0)
    edf = NaN;
    return;
end

basic_sum = compute_basic_sum(J, M, S, F, alpha, d, sz0);

if basic_sum <= 0 || isnan(basic_sum)
    edf = NaN;
    return;
end

edf = M * sz0^2 / basic_sum;
end

% ── Helper functions ──────────────────────────────────────────────────────────

function sw = compute_sw(t, alpha)
t_abs = abs(t);
switch alpha
    case  2;  sw = -t_abs;
    case  1;  sw = t^2 * log(max(t_abs, eps));
    case  0;  sw = t_abs^3;
    case -1;  sw = -t^4 * log(max(t_abs, eps));
    case -2;  sw = -t_abs^5;
    case -3;  sw = t^6 * log(max(t_abs, eps));
    case -4;  sw = t_abs^7;
    otherwise; sw = NaN;
end
end

function sx = compute_sx(t, F, alpha)
if F > 100 && alpha <= 0
    sx = compute_sw(t, alpha + 2);
else
    sx = F^2 * (2*compute_sw(t, alpha) - ...
                compute_sw(t - 1/F, alpha) - ...
                compute_sw(t + 1/F, alpha));
end
end

function sz = compute_sz(t, F, alpha, d)
switch d
    case 1
        sz = 2*compute_sx(t,F,alpha) - compute_sx(t-1,F,alpha) - compute_sx(t+1,F,alpha);
    case 2
        sz = 6*compute_sx(t,F,alpha) ...
           - 4*compute_sx(t-1,F,alpha) - 4*compute_sx(t+1,F,alpha) ...
           + compute_sx(t-2,F,alpha)   + compute_sx(t+2,F,alpha);
    case 3
        sz = 20*compute_sx(t,  F,alpha) ...
           - 15*compute_sx(t-1,F,alpha) - 15*compute_sx(t+1,F,alpha) ...
           +  6*compute_sx(t-2,F,alpha) +  6*compute_sx(t+2,F,alpha) ...
           -    compute_sx(t-3,F,alpha) -    compute_sx(t+3,F,alpha);
    otherwise
        sz = NaN;
end
end

function bsum = compute_basic_sum(J, M, S, F, alpha, d, sz0)
bsum = sz0^2;
for j = 1:(J-1)
    szj  = compute_sz(j/S, F, alpha, d);
    bsum = bsum + 2*(1 - j/M)*szj^2;
end
if J <= M
    szJ  = compute_sz(J/S, F, alpha, d);
    bsum = bsum + (1 - J/M)*szJ^2;
end
end
