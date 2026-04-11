function [tau, mhdev, edf, ci, alpha] = mhdev(x, tau0, m_list)
%MHDEV   Computes modified Hadamard deviation (MHDEV) from phase data.
%
%   [tau, mhdev, edf, ci, alpha] = allanlab.mhdev(x, tau0, m_list)
%
%   Input:
%     x      – Phase data (seconds), row or column vector
%     tau0   – Basic sampling interval (seconds)
%     m_list – Averaging factors (optional), defines τ = m·τ₀
%
%   Output:
%     tau   – Averaging times τ = m·τ₀ (seconds)
%     mhdev – Modified Hadamard deviation σ_H,mod(τ), dimensionless
%     edf   – Equivalent degrees of freedom (Greenhall & Riley, 2003)
%     ci    – Confidence interval matrix [ci_lower, ci_upper]
%     alpha – Noise type exponent estimated from lag1ACF, B1 bias, and R(n) bias
%
%   References:
%     W. J. Riley, *Handbook of Frequency Stability Analysis*, NIST SP1065,
%     Section 5.2.10
%     https://www.nist.gov/publications/handbook-frequency-stability-analysis
%
%     C. A. Greenhall and W. J. Riley, *Uncertainty of Stability Variances*,
%     Proceedings of the 35th Annual Precise Time and Time Interval (PTTI) Meeting, 2003.
%     https://www.wriley.com/Uncertainty%20of%20Stability%20Variances.pdf

    import allanlab.*  % noise_id, compute_ci

    %-- Ensure column vector format for input
    if isrow(x), x = x.'; end
    N = numel(x);

    %-- Default m_list: octave-spaced values with ≥4m points
    if nargin < 3 || isempty(m_list)
        m_list = 2.^(0:floor(log2(N/4)));
    end

    %-- Preallocate outputs
    tau   = m_list * tau0;
    mhdev = NaN(size(m_list));
    edf   = NaN(size(m_list));
    ci    = NaN(numel(m_list), 2);
    alpha = NaN(size(m_list));
    Neff  = NaN(size(m_list));  % effective number of samples

    %-- Estimate alpha using lag1ACF, B1 bias, and R(n) bias
    alpha = noise_id(x, m_list, 'phase');

    %-- Loop over averaging factors m
    for k = 1:numel(m_list)
        m = m_list(k);
        N_eff = N - 4*m + 1;  % from Riley (wriley.com/paper4ht.htm)
        if N_eff <= 0, break, end
        
        % Store effective number of samples
        Neff(k) = N_eff;

        % Third difference: x(n) - 3x(n+m) + 3x(n+2m) - x(n+3m)
        d4 = x(1:N_eff) - 3*x(1+m:N_eff+m) + 3*x(1+2*m:N_eff+2*m) - x(1+3*m:N_eff+3*m);

        % Prefix sum for efficient moving average of third differences
        S = cumsum([0; d4]);
        avg = S(m+1:end) - S(1:end-m);  % mean over m-point windows

        % SP1065 §5.2.10: σ²_H,mod(τ) = ⟨(⟨Δ³x⟩_m)²⟩ / (6·m²)
        mhvar = mean(avg.^2) / (6 * m^2);
        mhdev(k) = sqrt(mhvar) / tau(k);

        % EDF from Greenhall & Riley (2003), d = 3, F = 1 (modified), S = m
        try
            edf(k) = calculate_edf(alpha(k), 3, m, 1, m, N);
        catch
            edf(k) = NaN;
        end
    end

    %-- Compute confidence intervals with default p = 0.683
    p = 0.683;
    ci = compute_ci(mhdev, edf, p, alpha, Neff);
end