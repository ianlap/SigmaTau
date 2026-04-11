function alpha_list = noise_id(x, m_list, data_type, dmin, dmax)
%NOISE_ID  Dominant power-law noise estimator from time series data
%
%   alpha_list = noise_id(x, m_list, data_type, dmin, dmax)
%
%   Inputs:
%     x         – phase or freq data (column vector)
%     m_list    – list of averaging factors (τ = m·τ₀)
%     data_type – 'phase' or 'freq'
%     dmin      – minimum differencing depth (default = 0)
%     dmax      – maximum differencing depth (default = 2)
%
%   Output:
%     alpha_list – estimated α values at each τ
%
%   Method: for each m, use lag-1 autocorrelation estimator when N_eff ≥ 30,
%           otherwise use fallback via B1 ratio and R(n) test.
%
%   See also: lag1ACF, B1 ratio method, SP1065

    import allanlab.*

    if nargin < 4, dmin = 0; end
    if nargin < 5, dmax = 1; end
    x = preprocess_x(x);  % clean raw data: outlier removal + detrending
    alpha_list = NaN(size(m_list));

    for k = 1:length(m_list)
        m = m_list(k);

        % Estimate number of usable points after averaging
        N_eff = floor(length(x) / m);

        try
            if N_eff >= 30
                % Use lag-1 ACF method
                [alpha, ~, ~, ~] = noiseID_lag1acf(x, m, data_type, dmin, dmax);
            else
                % Use B1 ratio + R(n) fallback method
                [alpha, ~, b1_obs] = noiseID_B1Rn(x, m, data_type);
            end
            alpha_list(k) = round(alpha);
        catch err
            warning("Noise ID failed for m = %d: %s", m, err.message);
            alpha_list(k) = NaN;
        end
    end
end

% -- LOCAL FUNCTION: Remove outliers and detrend --------------------------
function x_out = preprocess_x(x)
    x = x(:);
    z = (x - mean(x)) / std(x);
    x_clean = x(abs(z) < 5);     % remove >5σ outliers
    x_out = detrend(x_clean, 1); % remove linear trend (frequency drift)
end

% -- LOCAL FUNCTION: Lag-1 ACF method -------------------------------------
function [alpha, alpha_int, d, rho] = noiseID_lag1acf(x, m, data_type, dmin, dmax)
    if nargin < 4, dmin = 0; end
    if nargin < 5, dmax = 2; end

    % Step 1: preprocess by type
    if strcmpi(data_type, 'phase')
        if m > 1, x = x(1:m:end); end           % decimate
        x = detrend(x, 2);                      % remove drift
    elseif strcmpi(data_type, 'freq')
        N = floor(length(x) / m) * m;
        x = reshape(x(1:N), m, []);
        x = mean(x, 1)';
        x = detrend(x, 1);
    else
        error('data_type must be ''phase'' or ''freq''.');
    end

    % Step 2: differencing loop
    d = 0;
    while true
        r1 = compute_lag1_acf(x);              % lag-1 autocorrelation
        rho = r1 / (1 + r1);                   % fractional integration index

        if d >= dmin && (rho < 0.25 || d >= dmax)
            p = -2 * (rho + d);                % spectral slope
            alpha = p + 2 * strcmpi(data_type, 'phase');
            alpha_int = round(alpha);
            return;
        else
            x = diff(x);
            d = d + 1;
            if length(x) < 5
                error('Data too short after differencing.');
            end
        end
    end
end

function r1 = compute_lag1_acf(x)
    x = x(:) - mean(x);
    if all(x == 0)
        r1 = NaN;
        return;
    end
    x0 = x(1:end-1);
    x1 = x(2:end);
    r1 = sum(x0 .* x1) / sum(x.^2);
end

% -- LOCAL FUNCTION: B1 ratio and R(n) fallback method ---------------------
function [alpha_int, mu_best, B1_obs] = noiseID_B1Rn(x_full, m, data_type)
    x_full = x_full(:);

    if strcmpi(data_type, 'phase')
        % Decimate and detrend phase data
        x_dec = x_full(1:m:end);
        x_dec = detrend(x_dec, 2);
        tau = m;
        avar_val = allanlab.avar(x_dec, tau);
        N_avar = floor(length(x_dec) - 2);

        % Classical variance of averaged diffs
        dx = diff(x_full);
        N = floor(length(dx) / m) * m;
        if N < m, alpha_int = NaN; mu_best = NaN; B1_obs = NaN; return; end
        dx = dx(1:N);
        y_blocks = reshape(dx, m, []);
        y_avg = mean(y_blocks, 1);
        var_classical = var(y_avg, 0);

    elseif strcmpi(data_type, 'freq')
        N = floor(length(x_full) / m) * m;
        if N < 2*m, alpha_int = NaN; mu_best = NaN; B1_obs = NaN; return; end
        x = reshape(x_full(1:N), m, []);
        y_avg = mean(x, 1)';
        y_avg = detrend(y_avg, 1);
        dy = diff(y_avg);
        var_classical = var(y_avg, 0);
        avar_val = sum(dy.^2) / (2 * (length(y_avg) - 1));
        N_avar = length(y_avg);
    else
        error('Unsupported data_type: use ''phase'' or ''freq''.');
    end

    % Compute observed B1 ratio
    B1_obs = var_classical / avar_val;
    
    % Define noise types based on the complete table
    % Note: Code only handles μ = [-2, -1, 0, 1], not μ = 2 or 3
    mu_list    = [1, 0, -1, -2];  % Ordered from high to low for checking
    alpha_list = [-2, -1, 0, 2];   % RWFM, FLFM, WHFM, WHPM
    noise_types = {'RWFM', 'FLFM', 'WHFM', 'WHPM'};
    
    % Calculate theoretical B1 values
    b1_vals = arrayfun(@(mu) b1_theory(N_avar, mu), mu_list);
    
    % Decision boundaries using geometric means (NIST approach)
    % Check from highest μ downward
    mu_best = mu_list(end);  % Default to lowest μ
    alpha_int = alpha_list(end);
    
    for i = 1:length(mu_list)-1
        boundary = sqrt(b1_vals(i) * b1_vals(i+1));
        
        if B1_obs > boundary
            mu_best = mu_list(i);
            alpha_int = alpha_list(i);
            break;
        end
    end
 
    % Refine α = 2 vs 1 using R(n) when needed (FLPM vs WHPM)
    if mu_best == -2 && strcmpi(data_type, 'phase')
        adev = sqrt(avar_val);
        mdev = simple_mdev(x_full, tau, 1);
        Rn_obs = (mdev / adev)^2;
        R_hi = rn_theory(m, 0);   % α = 2 (WHPM)
        R_lo = rn_theory(m, -1);  % α = 1 (FLPM)
        if Rn_obs > sqrt(R_hi * R_lo)
            alpha_int = 1;  % Flicker PM
        else
            alpha_int = 2;  % White PM
        end
    end
end

% -- LOCAL FUNCTION: Theoretical B1 values from slope μ -------------------
function B1 = b1_theory(N, mu)
    switch mu
        case 2,   B1 = N*(N+1)/6;
        case 1,   B1 = N/2;
        case 0,   B1 = N*log(N)/(2*(N-1)*log(2));
        case -1,  B1 = 1;
        case -2,  B1 = (N^2 - 1) / (1.5 * N * (N - 1));
        otherwise
            B1 = (N * (1 - N^mu)) / (2 * (N - 1) * (1 - 2^mu));
    end
end

% -- LOCAL FUNCTION: Theoretical R(n) values ------------------------------
function Rn = rn_theory(af, b)
    switch b
        case 0
            Rn = af^(-1);  % White PM
        case -1
            avar = (1.038 + 3 * log(2 * pi * 0.5 * af)) / (4 * pi^2);
            mvar = 3 * log(256 / 27) / (8 * pi^2);
            Rn = mvar / avar;  % Flicker PM
        otherwise
            Rn = 1;
    end
end

% -- LOCAL FUNCTION: MDev without calling NoiseID ------------------------
function mdev = simple_mdev(x, m, tau0)
    % Compute non-overlapping MDEV without calling noise_id or full mdev
    N = length(x);
    L = N - 3*m + 1;
    if L <= 0
        mdev = NaN;
        return;
    end

    % Moving averages via prefix sum
    S = cumsum([0; x]);  % prefix sum
    s1 = S(1+m:L+m)     - S(1:L);
    s2 = S(1+2*m:L+2*m) - S(1+m:L+m);
    s3 = S(1+3*m:L+3*m) - S(1+2*m:L+2*m);
    d = s3 - 2*s2 + s1;

    mvar = mean(d.^2) / (2 * m^2 * tau0^2);
    mdev = sqrt(mvar);
end