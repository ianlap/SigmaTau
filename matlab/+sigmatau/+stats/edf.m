function edf_vec = edf(result)
% EDF  Compute equivalent degrees of freedom for a deviation result struct.
%
%   edf_vec = sigmatau.stats.edf(result)
%
%   Dispatches to calculate_edf or totaldev_edf based on result.method.
%   Mirrors Julia's edf_for_result in stats.jl.
%
%   Input:
%     result – deviation result struct with fields:
%                .method    – deviation name ('adev', 'mdev', ...)
%                .alpha     – noise exponents (integer, per tau)
%                .tau       – averaging times (s)
%                .tau0      – basic sampling interval (s)
%                .N         – number of phase data points
%
%   Output:
%     edf_vec – row vector of EDF values, NaN where parameters are invalid

tau   = result.tau(:)';
alpha = result.alpha(:)';
L     = numel(tau);
T_rec = (result.N - 1) * result.tau0;   % record duration (s)

edf_vec = NaN(1, L);

for k = 1:L
    m = round(tau(k) / result.tau0);
    edf_vec(k) = edf_dispatch(result.method, alpha(k), m, tau(k), ...
                               result.tau0, result.N, T_rec);
end
end

% ── Dispatch ─────────────────────────────────────────────────────────────────

function val = edf_dispatch(method, alpha, m, tau, tau0, N, T)
% SP1065 dispatch: d=2 for Allan family, d=3 for Hadamard family.
% F=m for unmodified; F=1 for modified. Stride S=1 (overlapping).
switch lower(method)
    case 'adev'
        val = sigmatau.stats.calculate_edf(alpha, 2, m, m, 1, N);
    case 'mdev'
        val = sigmatau.stats.calculate_edf(alpha, 2, m, 1, 1, N);
    case 'hdev'
        val = sigmatau.stats.calculate_edf(alpha, 3, m, m, 1, N);
    case 'mhdev'
        val = sigmatau.stats.calculate_edf(alpha, 3, m, 1, 1, N);
    case 'totdev'
        val = sigmatau.stats.totaldev_edf('totvar', alpha, T, tau);
    case 'mtotdev'
        val = sigmatau.stats.totaldev_edf('mtot', alpha, T, tau);
    case 'htotdev'
        val = sigmatau.stats.totaldev_edf('htot', alpha, T, tau);
    case 'mhtotdev'
        val = sigmatau.stats.totaldev_edf('mhtot', alpha, T, tau);
    case 'tdev'
        val = sigmatau.stats.calculate_edf(alpha, 2, m, 1, 1, N);  % same as mdev
    case 'ldev'
        val = sigmatau.stats.calculate_edf(alpha, 3, m, 1, 1, N);  % same as mhdev
    otherwise
        warning('SigmaTau:edf', 'Unknown method "%s", returning NaN', method);
        val = NaN;
end
end
