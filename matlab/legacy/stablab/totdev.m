function [tau, totdev, edf, ci, alpha, Neff] = totdev(x, tau0, m_list)
%TOTDEV   Computes total deviation (TOTDEV) from phase data.
%
%   [tau, totdev, edf, ci, alpha, Neff] = allanlab.totdev(x, tau0, m_list)
%
%   Input:
%     x      – phase data (seconds), sampled at interval τ₀
%     tau0   – basic sampling interval (seconds)
%     m_list – averaging factors (optional). Each m gives τ = m·τ₀.
%
%   Output:
%     tau     – averaging times τ (seconds)
%     totdev  – bias-corrected total deviation σ_tot(τ) (unitless)
%     edf     – equivalent degrees of freedom
%     ci      – confidence intervals: [lo hi] for each τ (default p = 0.683)
%     alpha   – power-law noise type estimate from noise_id
%     Neff    – number of effective points contributing to each τ
%
%   References:
%     W. J. Riley & D. A. Howe, “Handbook of Frequency Stability Analysis,”
%     NIST Special Publication 1065, §5.2.11.
%     https://www.nist.gov/publications/handbook-frequency-stability-analysis
%
%     EDF formulas: Greenhall & Riley (PTTI 2003)
%     https://www.wriley.com/Uncertainty%20of%20Stability%20Variances.pdf

    import allanlab.*  % noise_id, totaldev_edf, compute_ci, bias_correction

    %-- ensure column vector
    if isrow(x), x = x.'; end
    N = numel(x);

    %-- default m list: powers of 2 with ≥2m data
    if nargin < 3 || isempty(m_list)
        m_list = 2 .^ (0:floor(log2(N/2)));
    end

    %-- remove linear frequency drift
    x_drift_removed = detrend(x);

    %-- symmetric reflection about endpoints
    x_left  = 2*x_drift_removed(1) - x_drift_removed(2:N-1);
    x_right = 2*x_drift_removed(end) - x_drift_removed(end-1:-1:2);
    x_star  = [x_left; x_drift_removed; x_right];
    offset  = numel(x_left);

    %-- preallocate
    tau     = m_list * tau0;
    totdev  = NaN(size(tau));
    rawvar  = NaN(size(tau));  % for bias correction
    edf     = NaN(size(tau));
    ci      = NaN(numel(tau), 2);
    alpha   = NaN(size(tau));
    Neff    = NaN(size(tau));

    %-- Warn if estimated runtime is high
    warn_threshold = 30; % seconds
    est_runtime = 2e-5 * N^1.4;
    if est_runtime > warn_threshold
        fprintf(['WARNING: Calculation of TOTDEV may take a long time (estimated %.1f seconds for N = %d).\n'], est_runtime, N);
    end

    %-- compute raw total deviation
    text_progress(0, 'Computing TOTDEV');
    for k = 1:numel(m_list)
        m = m_list(k);
        i_all = 1:(3*N - 2*m - 4);
        center = i_all + m;
        valid = (center >= 1) & (center <= N);
        if ~any(valid), continue, end

        i = i_all(valid);
        d2 = x_star(offset + i + 2*m) ...
           - 2*x_star(offset + i + m) ...
           +     x_star(offset + i);

        D = sum(d2.^2);
        den = 2 * (N - 2) * (m * tau0)^2;
        rawvar(k) = D / den;

        % Number of effective points for CI fallback
        Neff(k) = numel(d2);
	text_progress(k/numel(m_list));
    end
    
    text_progress(1);  % Complete progress

    %-- trim invalid
    valid = ~isnan(rawvar);
    tau     = tau(valid);
    rawvar  = rawvar(valid);
    m_list  = m_list(valid);
    Neff    = Neff(valid);

    %-- noise type estimation (d = 2)
    alpha = noise_id(x, m_list, 'phase', 0, 1);

    %-- EDF
    T = N * tau0;
    for k = 1:numel(m_list)
        try
            edf(k) = totaldev_edf('totvar', alpha(k), T, tau(k));
        catch
            edf(k) = NaN;
        end
    end

    %-- Bias correction (SP1065 §5.11)
    B =  bias_correction(alpha, 'totvar', tau, T);
    totdev = sqrt(rawvar ./ B);  % corrected

    %-- Confidence intervals (default: p = 0.683)
    p = 0.683;
    ci = compute_ci(totdev, edf, p, alpha, Neff);
end
