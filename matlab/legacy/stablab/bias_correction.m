function B = bias_correction(alpha, varType, tau, T)
%BIAS_CORRECTION   Bias factor B(α) for TOTVAR, MTOT, and HTOT corrections.
%
%   B = bias_correction(alpha, varType, tau, T)
%
%   Inputs:
%     alpha   – noise exponent α (scalar or vector)
%     varType – 'totvar' | 'mtot' | 'htot'
%     tau     – averaging time τ (same shape as alpha)
%     T       – record duration T = N·τ₀ (scalar)
%
%   Output:
%     B       – bias factor for each τ, same size as alpha
%
%   References:
%     NIST SP1065 §§5.11–5.13
%     Greenhall & Riley, “Uncertainty of Stability Variances”, PTTI 2003
%     https://www.wriley.com/Uncertainty%20of%20Stability%20Variances.pdf

    alpha = double(alpha);
    B = ones(size(alpha));

    switch lower(varType)
        case 'totvar'
            % TOTVAR: B(τ) = 1 - a * (τ / T)
            for k = 1:numel(alpha)
                if alpha(k) == -1
                    a = 1 / (3 * log(2));  % ≈ 0.481
                    B(k) = 1 - a * (tau(k) / T);
                elseif alpha(k) == -2
                    a = 0.75;
                    B(k) = 1 - a * (tau(k) / T);
                else
                    B(k) = 1;  % no correction for other α
                end
            end

        case 'mtot'
            % MTOTDEV bias factors from SP1065 Table 11
            B_table = containers.Map( ...
                {'2', '1', '0', '-1', '-2'}, ...
                [1.06, 1.17, 1.27, 1.30, 1.31] ...
            );
            for k = 1:numel(alpha)
                key = num2str(alpha(k));
                if isKey(B_table, key)
                    B(k) = B_table(key);
                else
                    B(k) = 1;
                end
            end

        case 'htot'
            % HTOT bias: B = 1 / (1 + a), using a(α) from Table 1 (FCS 2001)
            a_table = containers.Map( ...
                {'0', '-1', '-2', '-3', '-4'}, ...
                [-0.005, -0.149, -0.229, -0.283, -0.321] ...
            );
            for k = 1:numel(alpha)
                key = num2str(alpha(k));
                if isKey(a_table, key)
                    a = a_table(key);
                    B(k) = 1 / (1 + a);  % bias factor = 1 / (1 + a)
                else
                    B(k) = 1;  % fallback if α not listed
                end
            end

        otherwise
            error('Unknown varType "%s". Use "totvar", "mtot", or "htot".', varType);
    end
end
