function ci_mat = ci(result, confidence)
% CI  Compute confidence intervals for a deviation result struct.
%
%   ci_mat = sigmatau.stats.ci(result)
%   ci_mat = sigmatau.stats.ci(result, confidence)
%
%   Uses chi-squared distribution when EDF is finite and positive
%   (requires Statistics and Machine Learning Toolbox for chi2inv).
%   Falls back to Gaussian ±Kn·dev·z/√N when EDF is NaN or zero.
%
%   Inputs:
%     result     – deviation result struct with fields:
%                    .deviation, .alpha, .edf, .N, .confidence
%     confidence – confidence level (default: result.confidence or 0.683)
%
%   Output:
%     ci_mat – Lx2 matrix [lower, upper] per averaging time

if nargin < 2 || isempty(confidence)
    if isfield(result, 'confidence')
        confidence = result.confidence;
    else
        confidence = 0.683;
    end
end

dev  = result.deviation(:);
alph = result.alpha(:);
N    = result.N;
L    = numel(dev);

edf_vec = zeros(L, 1);
if isfield(result, 'edf')
    edf_vec = result.edf(:);
end

ci_mat = NaN(L, 2);
z      = z_from_confidence(confidence);

% Detect if Statistics Toolbox chi2inv is available
have_chi2 = ~isempty(which('chi2inv'));

for k = 1:L
    d = dev(k);
    if isnan(d)
        continue;
    end
    ef = edf_vec(k);
    if isfinite(ef) && ef > 0 && have_chi2
        % Chi-squared CI: dev * sqrt(edf / chi2_{alpha/2}) to sqrt(edf / chi2_{1-alpha/2})
        a_chi = 1 - confidence;
        chi_lo = chi2inv(a_chi / 2,       ef);
        chi_hi = chi2inv(1 - a_chi / 2,   ef);
        ci_mat(k, 1) = d * sqrt(ef / chi_hi);
        ci_mat(k, 2) = d * sqrt(ef / chi_lo);
    else
        % Gaussian fallback: ±Kn * dev * z / sqrt(N)
        Kn = kn_from_alpha(alph(k));
        half = Kn * d * z / sqrt(N);
        ci_mat(k, 1) = d - half;
        ci_mat(k, 2) = d + half;
    end
end
end

% ── Helpers ───────────────────────────────────────────────────────────────────

function z = z_from_confidence(confidence)
% Normal quantile approximation (Abramowitz & Stegun 26.2.17).
% Matches Julia _z_from_confidence in stats.jl. Max error < 4.5e-4.
p = 1 - (1 - confidence) / 2;
t = sqrt(-2 * log(1 - p));
c = [2.515517, 0.802853, 0.010328];
d = [1.432788, 0.189269, 0.001308];
z = t - (c(1) + c(2)*t + c(3)*t^2) / (1 + d(1)*t + d(2)*t^2 + d(3)*t^3);
end

function Kn = kn_from_alpha(alpha)
% SP1065 Kn factors per noise type.
switch round(alpha)
    case -2;  Kn = 0.75;
    case -1;  Kn = 0.77;
    case  0;  Kn = 0.87;
    case  1;  Kn = 0.99;
    case  2;  Kn = 0.99;
    otherwise; Kn = 1.10;
end
end
