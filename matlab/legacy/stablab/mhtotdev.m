function [tau, mhtotdev, edf, ci, alpha] = mhtotdev(x, tau0, m_list)
%MHTOTDEV Computes Modified Hadamard Total Deviation WITHOUT forcing agreement at tau=1s
%
% [tau, mhtotdev, edf, ci, alpha] = allanlab.mhtotdev(x, tau0, m_list)
%
% Input:
%   x       – Phase data (seconds), row or column vector
%   tau0    – Basic sampling interval (seconds)
%   m_list  – Averaging factors (optional), defines τ = m·τ₀
%
% Output:
%   tau       – Averaging times τ = m·τ₀ (seconds)
%   mhtotdev  – Modified Hadamard total deviation σ_MHtot(τ), unitless
%   edf       – NaN vector (no published EDF model available)
%   ci        – Confidence intervals (default 68.3% method)
%   alpha     – Noise type exponent α estimated from lag1ACF, B1, R(n)
%
% References:
%   W. J. Riley, *Handbook of Frequency Stability Analysis*, NIST SP1065,
%   Sections 5.2.12 & 5.2.14
%   https://www.nist.gov/publications/handbook-frequency-stability-analysis

import allanlab.*

%-- ensure column vector
if isrow(x), x = x.'; end
N = numel(x);

%-- default m list: octave spaced, ensuring at least 4m points
if nargin < 3 || isempty(m_list)
    m_list = 2.^(0:floor(log2(N/4)));
end

%-- initialize outputs
 tau = m_list * tau0;
 mhtotdev = NaN(size(tau));
 edf = NaN(size(tau)); % no published EDF model
 ci = NaN(numel(tau), 2);
 MHvar = NaN(size(tau));
 Neff_list = N - 4*m_list + 1;

%-- estimate alpha from phase data
 alpha = noise_id(x, m_list, 'phase', 0, 1);

%-- Warn if estimated runtime is high
warn_threshold = 30; % seconds
est_runtime = 2e-5 * N^1.4;
if est_runtime > warn_threshold
    fprintf(['WARNING: Calculation of MHTOTDEV may take a long time (estimated %.1f seconds for N = %d).\n'], est_runtime, N);
end

%-- compute MHTOTDEV (no special case for m=1)
for k = 1:numel(m_list)
    m = m_list(k);
    nsubs = Neff_list(k);
    if nsubs < 1, continue, end
    total_sum = 0;
    for n = 1:nsubs
        % Extract phase segment (3m+1 points for 3m frequency samples)
        phase_seg = x(n : n + 3*m);
        % linear detrending of phase data
        t = (0:3*m)';
        phase_detrended = detrend(phase_seg, 'linear');
        % symmetric reflection of phase data
        ext = [phase_detrended(end:-1:1); phase_detrended; phase_detrended(end:-1:1)];
        % third difference on phase data
        L = numel(ext) - 3*m;
        d3 = ext(1:L) - 3*ext(1+m:L+m) + 3*ext(1+2*m:L+2*m) - ext(1+3*m:L+3*m);
        % moving average
        S = cumsum([0; d3]);
        if numel(S) > m
            avg = S(m+1:end) - S(1:end-m);
            block_var = mean(avg.^2) / (6 * m^2);
        else
            block_var = 0;
        end
        total_sum = total_sum + block_var;
    end
    % store average variance
    MHvar(k) = total_sum / nsubs;
    % convert to deviation and normalize by tau
    mhtotdev(k) = sqrt(MHvar(k)) / tau(k);
end

T = N * tau0;
for k = 1:numel(m_list)
    try
        edf(k) = totaldev_edf('mhtot', alpha(k), T, tau(k));
    catch
        edf(k) = NaN;
    end
end



%-- confidence intervals (default: 68.3%)
ci = compute_ci(mhtotdev, edf, 0.683, alpha, Neff_list);

end