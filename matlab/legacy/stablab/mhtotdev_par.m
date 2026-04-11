function [tau, mhtotdev, edf, ci, alpha] = mhtotdev_par(x, tau0, m_list)
%MHTOTDEV_PAR  Modified Hadamard Total Deviation (vectorised & parallel‑ready)
%
%   [tau, mhtotdev, edf, ci, alpha] = mhtotdev_par(x, tau0, m_list)
%
%   Drop‑in replacement for allanlab.mhtotdev with substantially lower run‑time
%   on multi‑core machines. Numerical results are identical to the original
%   routine; only the execution model is different.
%
%   INPUTS
%     x       – phase data (seconds), column or row vector
%     tau0    – basic sampling interval (seconds)
%     m_list  – list of averaging factors; if omitted, defaults to powers of 2
%
%   OUTPUTS
%     tau       – column vector of averaging times τ = m·τ₀ (seconds)
%     mhtotdev  – column vector of σ_MHtot(τ)
%     edf       – column vector (NaN; no published EDF model)
%     ci        – N×2 matrix of 68.3 % confidence intervals
%     alpha     – column vector of noise‑type exponents from noise_id
%
%   References:
%     W. J. Riley, *Handbook of Frequency Stability Analysis*, NIST SP 1065,
%     §§ 5.2.12 & 5.2.14 (formulas only).
%
%   Author: Ian Lapinski
%   Date:   2025‑07‑14
%
%--------------------------------------------------------------------------
import allanlab.*
%% Input handling
if isrow(x), x = x.'; end
N = numel(x);

if nargin < 3 || isempty(m_list)
    m_list = 2.^(0:floor(log2(N/4)));   % octave‑spaced default
end

Nm         = numel(m_list);
% ensure column orientation for all outputs
m_list     = m_list(:);

tau        = m_list * tau0;            % Nm×1 column
mhtotdev   = NaN(Nm,1);
MHvar      = NaN(Nm,1);
edf        = NaN(Nm,1);
Neff_list  = N - 4*m_list + 1;         % Nm×1 column

%% Noise‑type exponent α (needed for CI)
alpha = noise_id(x, m_list, 'phase', 0, 2);
alpha = alpha(:);                       % force column

%% Decide on parallel execution
usePar = license('test','Distrib_Computing_Toolbox') && (Nm > 1);
if usePar
    pool = gcp('nocreate');
    if isempty(pool)
        try, parpool('local'); catch, usePar = false; end
    end
end

%% Core computation – outer m‑loop
if usePar
    parfor k = 1:Nm
        [MHvar(k), mhtotdev(k)] = mh_single_m(m_list(k), x, tau(k), Neff_list(k));
    end
else
    for k = 1:Nm
        [MHvar(k), mhtotdev(k)] = mh_single_m(m_list(k), x, tau(k), Neff_list(k));
    end
end

%% EDF Calculation from Empirical Fit

T = N * tau0;
for k = 1:numel(m_list)
    try
        edf(k) = totaldev_edf('mhtot', alpha(k), T, tau(k));
    catch
        edf(k) = NaN;
    end
end

%% Confidence intervals (68.3 %)
ci = compute_ci(mhtotdev, edf, 0.683, alpha, Neff_list);

end % ============================== END MAIN ==============================


%==========================================================================
function [varhat, devhat] = mh_single_m(m, x, tau_k, Neff)
%MH_SINGLE_M  Modified Hadamard total variance for one averaging factor m.
%
%   This helper is invoked inside the (par)for loop. All temporaries are
%   kept local to the worker to minimise slicing overhead.

if Neff < 1
    varhat = NaN; devhat = NaN; return
end

total = 0;
for n = 1:Neff
    % -- 1. extract phase slice of length 3m+1 --
    idx = n : n + 3*m;
    x_seg = x(idx);

    % -- 2. linear detrend (per slice) --
    x_det = detrend(x_seg, 'linear');

    % -- 3. symmetric reflection (Riley SP 1065 §5.2.12) --
    ext = [x_det(end:-1:1); x_det; x_det(end:-1:1)];

    % -- 4. third difference --
    L  = numel(ext) - 3*m;
    d3 = ext(1:L) - 3*ext(1+m:L+m) + 3*ext(1+2*m:L+2*m) - ext(1+3*m:L+3*m);

    % -- 5. m‑sample moving average via prefix sums --
    S   = cumsum([0; d3]);
    if numel(S) > m
        avg   = S(m+1:end) - S(1:end-m);
        total = total + mean(avg.^2) / (6*m^2);
    end
end

varhat = total / Neff;          % modified Hadamard total *variance*
devhat = sqrt(varhat) / tau_k;  % convert to σ_MHtot

end
