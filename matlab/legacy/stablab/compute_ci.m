function ci = compute_ci(dev, edf, p, alpha, N)
%COMPUTE_CI   Confidence intervals for stability deviations
%
%   ci = compute_ci(dev, edf, p, alpha, N)
%
%   Inputs:
%     dev   – deviation values (vector)
%     edf   – equivalent degrees of freedom (vector)
%     p     – confidence level (e.g., 0.683)
%     alpha – noise type estimates (vector, same length as dev)
%     N     – number of raw samples used (vector, same length as dev)
%
%   Output:
%     ci – Nx2 matrix of confidence intervals: [lower, upper]

    % Ensure column vectors
    dev   = dev(:);
    edf   = edf(:);
    alpha = alpha(:);
    N     = N(:);

    L = numel(dev);

    % Dimension checks with explicit error messages
    if numel(edf)   ~= L
        error('edf is not the correct dimensionality. dev vector has length %d, pass an edf vector that is the same length.', L);
    end
    if numel(alpha) ~= L
        error('alpha is not the correct dimensionality. dev vector has length %d, pass an alpha vector that is the same length.', L);
    end
    if numel(N)     ~= L
        error('N is not the correct dimensionality. dev vector has length %d, pass an N vector that is the same length.', L);
    end

    % z-score for confidence level
    z = norminv(1 - (1 - p)/2);

    % Initialize output
    ci = NaN(L, 2);

    % Loop over each point
    for k = 1:L
        if isnan(edf(k)) || edf(k) == 0
            % Fallback: Gaussian CI using Kn and sample count
            Kn = kn_from_alpha(alpha(k));
            margin = Kn * dev(k) * z / sqrt(N(k));
            ci(k,1) = dev(k) - margin;
            ci(k,2) = dev(k) + margin;
        else
            % CI from chi-squared distribution
            alpha_chi = 1 - p;
            chi2_lo = chi2inv(alpha_chi/2, edf(k));
            chi2_hi = chi2inv(1 - alpha_chi/2, edf(k));
            ci(k,1) = dev(k) * sqrt(edf(k) / chi2_hi);
            ci(k,2) = dev(k) * sqrt(edf(k) / chi2_lo);
        end
    end
end

function Kn = kn_from_alpha(alpha)
%KN_FROM_ALPHA   Lookup Kn factor based on rounded α
    switch round(alpha)
        case -2  % Random walk FM
            Kn = 0.75;
        case -1  % Flicker FM
            Kn = 0.77;
        case  0  % White FM
            Kn = 0.87;
        case  1  % Flicker PM
            Kn = 0.99;
        case  2  % White PM
            Kn = 0.99;
        otherwise
            Kn = 1.10;  % Conservative fallback
    end
end
