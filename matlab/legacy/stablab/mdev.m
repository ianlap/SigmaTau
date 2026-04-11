function [tau, mdev, edf, ci, alpha] = mdev(x, tau0, m_list)
%MDEV   Computes modified Allan deviation (MDEV) from phase data.
% ...
%     https://www.wriley.com/Uncertainty%20of%20Stability%20Variances.pdf

    import allanlab.*

    if isrow(x), x = x.'; end
    N = numel(x);

    if nargin < 3 || isempty(m_list)
        m_list = 2.^(0:floor(log2(N/3)));
    end

    tau    = m_list * tau0;
    mdev   = NaN(size(m_list));
    edf    = NaN(size(m_list));
    ci     = NaN(numel(m_list), 2);
    alpha  = NaN(size(m_list));
    N_eff  = NaN(size(m_list));  % <-- NEW

    alpha = noise_id(x, m_list, 'phase');
    x_cumsum = cumsum([0; x]);

    for k = 1:numel(m_list)
        m = m_list(k);
        N_eff_k = N - 3*m + 1;
        N_eff(k) = N_eff_k;  % <-- Store for CI calc

        if N_eff_k <= 0, break, end

        s1 = x_cumsum(1+m : N_eff_k+m)     - x_cumsum(1:N_eff_k);
        s2 = x_cumsum(1+2*m : N_eff_k+2*m) - x_cumsum(1+m : N_eff_k+m);
        s3 = x_cumsum(1+3*m : N_eff_k+3*m) - x_cumsum(1+2*m : N_eff_k+2*m);
        d = (s3 - 2*s2 + s1)/m;

        mvar = mean(d.^2) / (2 * m^2 * tau0^2);
        mdev(k) = sqrt(mvar);

        try
            edf(k) = calculate_edf(alpha(k), 2, m, 1, m, N);
        catch
            edf(k) = NaN;
        end
    end

    %-- Compute confidence intervals with valid N_eff vector
    p = 0.683;
    ci = compute_ci(mdev, edf, p, alpha, N_eff);  % <-- FIXED
end
