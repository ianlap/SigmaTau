function ci = compute_ci(result)
% COMPUTE_CI  Compute Gaussian confidence intervals (Kn fallback method).
%
%   ci = sigmatau.stats.compute_ci(result)
%
%   Inputs:
%     result – DeviationResult struct (from sigmatau.dev.engine)
%
%   Output:
%     ci – Lx2 matrix [lower, upper] per averaging time
%
%   Uses Gaussian fallback: ±Kn * dev * z / sqrt(N).
%   This matches the Julia compute_ci implementation (stats.jl).
%   No Statistics Toolbox chi-squared inversion is used.

dev        = result.deviation(:);
alpha_vals = result.alpha(:);
N          = result.N;
confidence = result.confidence;

L  = numel(dev);
ci = NaN(L, 2);

z = z_from_confidence(confidence);

for k = 1:L
    Kn   = kn_from_alpha(alpha_vals(k));
    half = Kn * dev(k) * z / sqrt(N);
    ci(k,1) = dev(k) - half;
    ci(k,2) = dev(k) + half;
end
end

% ── Helpers ───────────────────────────────────────────────────────────────────

function z = z_from_confidence(confidence)
% Normal quantile approximation (Abramowitz & Stegun 26.2.17).
% Matches Julia _z_from_confidence in stats.jl. Max error < 4.5e-4.
p = 1 - (1 - confidence)/2;
t = sqrt(-2*log(1 - p));
c = [2.515517, 0.802853, 0.010328];
d = [1.432788, 0.189269, 0.001308];
z = t - (c(1) + c(2)*t + c(3)*t^2) / (1 + d(1)*t + d(2)*t^2 + d(3)*t^3);
end

function Kn = kn_from_alpha(alpha)
switch round(alpha)
    case -2;  Kn = 0.75;
    case -1;  Kn = 0.77;
    case  0;  Kn = 0.87;
    case  1;  Kn = 0.99;
    case  2;  Kn = 0.99;
    otherwise; Kn = 1.10;
end
end
