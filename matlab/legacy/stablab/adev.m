function [tau, adev, edf, ci, alpha] = adev(x, tau0, m_list)
%ADEV   Computes overlapping Allan deviation (ADEV) from phase data.
%
%   [tau, adev, edf, ci, alpha] = allanlab.adev(x, tau0, m_list)
%
%   Input:
%     x      – Phase data (seconds), row or column vector
%     tau0   – Basic sampling interval (seconds)
%     m_list – Averaging factors (optional), defines τ = m·τ₀
%
%   Output:
%     tau   – Averaging times τ = m·τ₀ (seconds)
%     adev  – Overlapping Allan deviation σ_y(τ), dimensionless
%     edf   – Equivalent degrees of freedom (Greenhall & Riley, 2003)
%     ci    – Confidence interval matrix [ci_lower, ci_upper]
%     alpha – Noise type exponent estimated from lag1ACF, B1 bias, and R(n) bias
%
%   References:
%     W. J. Riley & D. A. Howe, “Handbook of Frequency Stability Analysis,”
%     NIST Special Publication 1065, §5.2.4.
%     https://www.nist.gov/publications/handbook-frequency-stability-analysis
%
%     C. A. Greenhall and W. J. Riley, *Uncertainty of Stability Variances*,
%     Proceedings of the 35th Annual Precise Time and Time Interval (PTTI) Meeting, 2003.
%     https://www.wriley.com/Uncertainty%20of%20Stability%20Variances.pdf

    import allanlab.*  % noise_id, compute_ci

    %-- ensure column vector format for input
    if isrow(x), x = x.'; end
    N = numel(x);

    %-- default m_list: octave‑spaced values with ≥2m points available
    if nargin < 3 || isempty(m_list)
        m_list = 2.^(0:floor(log2(N/2)));
    end 

    %-- preallocate outputs
    tau     = m_list * tau0;
    adev    = NaN(size(tau));
    edf     = NaN(size(tau));
    ci      = NaN(numel(tau), 2);
    alpha   = NaN(size(tau));
    Neff    = NaN(size(tau));  % effective number of samples

    %-- estimate alpha using lag1ACF, B1 bias, and R(n) bias
    alpha = noise_id(x, m_list, 'phase');

    %-- loop over averaging factors m to compute ADEV and EDF
    for k = 1:numel(m_list)
        m = m_list(k);
        L = N - 2*m;
        if L <= 0, break, end
        Neff(k) = L;

        % second differences: x(n+2m) - 2x(n+m) + x(n)
        d2 = x(1+2*m:N) - 2*x(1+m:N-m) + x(1:L);

        % SP1065 §5.2.4: σ²_y(τ) = ⟨(Δ²x)²⟩ / (2·m²·τ₀²)
        avar = mean(d2.^2) / (2 * m^2 * tau0^2);
        adev(k) = sqrt(avar);

        % EDF from Greenhall & Riley (2003), with d=2, F=m, S=m
        try
            edf(k) = calculate_edf(alpha(k), 2, m, m, m, N);
        catch
            edf(k) = NaN;
        end
    end

    %-- compute 68.3% confidence intervals using EDF or fallback
    p = 0.683;
    ci = compute_ci(adev, edf, p, alpha, Neff);
end
