function [tau, ldev, edf, ci, alpha] = ldev(x, tau0, m_list)
%LDEV   Computes Lapinski deviation (LDEV) from phase data.
%
%   [tau, ldev, edf, ci, alpha] = allanlab.ldev(x, tau0, m_list)
%
%   Input:
%     x      – Phase data (seconds), row or column vector
%     tau0   – Basic sampling interval (seconds)
%     m_list – Averaging factors (optional), defines τ = m·τ₀ (≥4·m points required)
%
%   Output:
%     tau   – Averaging times τ = m·τ₀ (seconds)
%     ldev  – Lapinski deviation σ_L(τ), in seconds
%     edf   – Equivalent degrees of freedom (Greenhall & Riley, 2003)
%     ci    – Confidence interval matrix [ci_lower, ci_upper], in seconds
%     alpha – Noise type exponent estimated from lag1ACF, B1 bias, and R(n) bias
%
%   Definition:
%     σ_L(τ) = (τ / √(10/3)) · σ_MH(τ), where σ_MH is the modified Hadamard deviation
%
%   References:
%     W. J. Riley, *Handbook of Frequency Stability Analysis*, NIST SP1065.
%     https://www.nist.gov/publications/handbook-frequency-stability-analysis
%     See also NIST SP1065 §5.2.10 (Modified Hadamard Deviation)
%
%     C. A. Greenhall and W. J. Riley, *Uncertainty of Stability Variances*,
%     Proceedings of the 35th Annual Precise Time and Time Interval (PTTI) Meeting, 2003.
%     https://www.wriley.com/Uncertainty%20of%20Stability%20Variances.pdf

    import allanlab.*  % call mhdev

    %-- ensure column vector format
    if isrow(x), x = x.'; end
    N = numel(x);

    %-- default m_list: powers-of-two with ≥4m points
    if nargin < 3 || isempty(m_list)
        m_list = 2.^(0:floor(log2(N/4)));
    end

    %-- compute Modified Hadamard Deviation (σ_MH) and metadata
    [tau, mhdev, edf, ci_mh, alpha] = allanlab.mhdev(x, tau0, m_list);

    %-- apply LDEV scaling: σ_L(τ) = τ / √(10/3) · σ_MH(τ)
    scale = tau / sqrt(10/3);
    % Ensure all vectors are the same length
    if numel(scale) ~= numel(mhdev)
        warning('Size mismatch between tau and mhdev — truncating to common length');
        n = min(numel(scale), numel(mhdev));
        scale = scale(1:n);
        mhdev = mhdev(1:n);
        ci_mh = ci_mh(1:n,:);
    end

    ldev = scale .* mhdev;
    ci = ci_mh .* scale(:);
end
