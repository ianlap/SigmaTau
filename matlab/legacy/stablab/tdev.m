function [tau, tdev, edf, ci, alpha] = tdev(x, tau0, m_list)
%TDEV   Computes time deviation (TDEV) from phase data.
%
%   [tau, tdev, edf, ci, alpha] = allanlab.tdev(x, tau0, m_list)
%
%   Input:
%     x      – Phase data (seconds), row or column vector
%     tau0   – Basic sampling interval (seconds)
%     m_list – Averaging factors (optional), defines τ = m·τ₀
%
%   Output:
%     tau   – Averaging times τ = m·τ₀ (seconds)
%     tdev  – Time deviation σ_x(τ), in seconds
%     edf   – Equivalent degrees of freedom (Greenhall & Riley, 2003)
%     ci    – Confidence interval matrix [ci_lower, ci_upper], in seconds
%     alpha – Noise type exponent estimated from lag1ACF, B1 bias, and R(n) bias
%
%   References:
%     W. J. Riley, *Handbook of Frequency Stability Analysis*, NIST SP1065,
%     Section 5.2.6
%     https://www.nist.gov/publications/handbook-frequency-stability-analysis
%
%     C. A. Greenhall and W. J. Riley, *Uncertainty of Stability Variances*,
%     Proceedings of the 35th Annual Precise Time and Time Interval (PTTI) Meeting, 2003.
%     https://www.wriley.com/Uncertainty%20of%20Stability%20Variances.pdf

    import allanlab.*  % mdev returns tau, mdev, edf, ci, alpha

     %-- ensure column vector
    if isrow(x), x = x.'; end
    N = numel(x);

    %-- default m_list
    if nargin < 3 || isempty(m_list)
        m_list = 2.^(0:floor(log2(N/3)));
    end

    %-- compute modified Allan deviation
    [tau, mdev, edf, ci_mdev, alpha] = allanlab.mdev(x, tau0, m_list);

    %-- convert MDEV to time deviation: TDEV = τ · MDEV / √3
    tdev = tau .* mdev / sqrt(3);

    %-- scale CI bounds accordingly: TDEV_CI = τ · MDEV_CI / √3
    ci = tau(:) ./ sqrt(3) .* ci_mdev;  % elementwise scaling of both columns
end
