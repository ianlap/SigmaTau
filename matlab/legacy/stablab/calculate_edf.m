function edf = calculate_edf(alpha, d, m, F, S, N)
% CALCULATE_EDF Computes equivalent degrees of freedom for stability variances
%
% Inputs:
%   alpha - frequency noise exponent (-4 to 2)
%           2: WHPM, 1: FLPM, 0: WHFM, -1: FLFM, -2: RWFM, -3: FWFM, -4: RRFM
%   d     - order of phase difference (1, 2, or 3)
%           2: Allan variance, 3: Hadamard variance
%   m     - averaging factor (tau/tau0), positive integer
%   F     - filter factor (1: modified variance, m: unmodified variance)
%   S     - stride factor (1: nonoverlapped, m: overlapped)
%   N     - number of phase data points
%
% Output:
%   edf   - equivalent degrees of freedom

% Check restriction
if alpha + 2*d <= 1
    error('Invalid parameters: alpha + 2d must be > 1');
end

% Initial steps
L = m/F + m*d;  % Filter length
if N < L
    error('Not enough data: N must be >= L');
end

M = 1 + floor(S*(N - L)/m);  % Number of summands
J = min(M, (d + 1)*S);       % Truncation parameter

% Compute sz(0, F, alpha, d)
sz0 = compute_sz(0, F, alpha, d);

% Compute BasicSum (simplified version)
basic_sum = compute_basic_sum(J, M, S, F, alpha, d);

% Calculate EDF
edf = M * sz0^2 / basic_sum;

end

function sw = compute_sw(t, alpha)
% Compute sw function from Table (7) in the paper
    t_abs = abs(t);
    
    switch alpha
        case 2
            sw = -t_abs;
        case 1
            sw = t^2 * log(max(t_abs, eps));  % Avoid log(0)
        case 0
            sw = t_abs^3;
        case -1
            sw = -t^4 * log(max(t_abs, eps));
        case -2
            sw = -t_abs^5;
        case -3
            sw = t^6 * log(max(t_abs, eps));
        case -4
            sw = t_abs^7;
        otherwise
            error('Invalid alpha value');
    end
end

function sx = compute_sx(t, F, alpha)
% Compute sx function from equation (8)
    if F == inf || (alpha <= 0 && F > 100)
        % Use limiting form for large F
        sx = compute_sw(t, alpha + 2);
    else
        sx = F^2 * (2*compute_sw(t, alpha) - ...
                    compute_sw(t - 1/F, alpha) - ...
                    compute_sw(t + 1/F, alpha));
    end
end

function sz = compute_sz(t, F, alpha, d)
% Compute sz function from equation (9)
    switch d
        case 1
            sz = 2*compute_sx(t, F, alpha) - ...
                 compute_sx(t-1, F, alpha) - ...
                 compute_sx(t+1, F, alpha);
            
        case 2
            sz = 6*compute_sx(t, F, alpha) - ...
                 4*compute_sx(t-1, F, alpha) - ...
                 4*compute_sx(t+1, F, alpha) + ...
                 compute_sx(t-2, F, alpha) + ...
                 compute_sx(t+2, F, alpha);
            
        case 3
            sz = 20*compute_sx(t, F, alpha) - ...
                 15*compute_sx(t-1, F, alpha) - ...
                 15*compute_sx(t+1, F, alpha) + ...
                 6*compute_sx(t-2, F, alpha) + ...
                 6*compute_sx(t+2, F, alpha) - ...
                 compute_sx(t-3, F, alpha) - ...
                 compute_sx(t+3, F, alpha);
            
        otherwise
            error('Invalid d value: must be 1, 2, or 3');
    end
end

function basic_sum = compute_basic_sum(J, M, S, F, alpha, d)
% Compute BasicSum from equation (10)
    sz0 = compute_sz(0, F, alpha, d);
    
    % First term - SQUARED
    basic_sum = sz0^2;
    
    % Summation terms - SQUARED
    for j = 1:(J-1)
        szj = compute_sz(j/S, F, alpha, d);
        basic_sum = basic_sum + 2 * (1 - j/M) * szj^2;  % Note: sz squared
    end
    
    % Last term (trapezoidal correction) - SQUARED
    if J <= M
        szJ = compute_sz(J/S, F, alpha, d);
        basic_sum = basic_sum + (1 - J/M) * szJ^2;  % Note: sz squared
    end
end

% Example usage:
% edf = calculate_edf(0, 2, 1, 1, 1, 1000);  % Modified Allan variance, WHFM
% edf = calculate_edf(0, 2, 1, 1, 1, 1000);  % Overlapped Allan variance, WHFM