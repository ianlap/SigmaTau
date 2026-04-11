function [tau, hdev, edf, ci, alpha] = hdev(x, tau0, m_list)
%HDEV   Computes overlapping Hadamard deviation (HDEV) from phase data.
%
%   [tau, hdev, edf, ci, alpha] = hdev(x, tau0, m_list)
%
%   Input:
%     x      - Phase data (seconds), row or column vector
%     tau0   - Basic sampling interval (seconds)
%     m_list - Averaging factors (optional), defines τ = m · τ₀
%
%   Output:
%     tau   - Averaging times τ = m · τ₀ (seconds)
%     hdev  - Overlapping Hadamard deviation σ_H(τ), unitless
%     edf   - Equivalent degrees of freedom (SP1065-based)
%     ci    - Confidence intervals on σ_H(τ), default p = 0.683
%     alpha - Estimated noise type exponent from lag1ACF, B1, and R(n)
%
%   References:
%     W. J. Riley, *Handbook of Frequency Stability Analysis*, NIST SP1065,
%     Sections 5.2.8–5.2.9
%     https://www.nist.gov/publications/handbook-frequency-stability-analysis

    import allanlab.*  % noise_id, compute_ci

    %-- ensure column vector
    if isrow(x), x = x.'; end
    N = numel(x);

    %-- default octave m list ensuring at least 3m data points
    if nargin < 3 || isempty(m_list)
        m_list = 2.^(0:floor(log2(N/4)));
    end

    %-- preallocate
    tau   = m_list * tau0;
    hdev  = NaN(size(m_list));
    edf   = NaN(size(m_list));
    alpha = NaN(size(m_list));
    ci    = NaN(numel(m_list), 2);
    Neff  = NaN(size(m_list));  % effective sample count

    %-- compute overlapping HDEV using third differences
    for k = 1:numel(m_list)
        m = m_list(k);
        L = N - 3*m;
        if L <= 0, break, end
        Neff(k) = L;

        % third difference: x(n+3m) - 3x(n+2m) + 3x(n+m) - x(n)
        d3 = x(1+3*m:N) - 3*x(1+2*m:N-m) + 3*x(1+m:N-2*m) - x(1:L);

        % SP1065: σ²_H(τ) = ⟨(Δ³x)²⟩ / (6·τ²)
        hvar = mean(d3.^2) / (6 * tau(k)^2);
        hdev(k) = sqrt(hvar);
    end

    %-- estimate alpha using lag1ACF, B1, and R(n)
    alpha = noise_id(x, m_list, 'phase');

    %-- compute EDF using SP1065 formulation
    for k = 1:numel(m_list)
        m = m_list(k);
        try
            % d = 3 (3rd difference), F = m, S = m (overlapping)
            edf(k) = calculate_edf(alpha(k), 3, m, m, m, N);
        catch
            edf(k) = NaN;
        end
    end

    %-- compute confidence intervals using chi-squared or fallback
    p = 0.683;
    ci = compute_ci(hdev, edf, p, alpha, Neff);
end
