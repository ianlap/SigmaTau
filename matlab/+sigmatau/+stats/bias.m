function B = bias(alpha, var_type, tau, T)
% BIAS  Bias factor B(alpha) for TOTVAR, MTOT, and HTOT corrections.
%
%   B = sigmatau.stats.bias(alpha, var_type, tau, T)
%
%   Canonical name per CLAUDE.md architecture. Delegates to bias_correction.
%
%   Inputs:
%     alpha    – noise exponent (scalar or vector)
%     var_type – 'totvar' | 'mtot' | 'htot'
%     tau      – averaging time (scalar or vector, same size as alpha)
%     T        – record duration (s), (N-1)*tau0 (scalar)
%
%   Output:
%     B – bias factor for each tau. Divide deviation by B for unbiased estimate.
%
%   Reference: SP1065 §§5.11–5.13; FCS 2001 Table 1.

B = sigmatau.stats.bias_correction(alpha, var_type, tau, T);
end
