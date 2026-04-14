function result = tdev(x, tau0, m_list, varargin)
% TDEV  Time deviation (TDEV). Wraps mdev and scales by tau/sqrt(3).
%
%   result = sigmatau.dev.tdev(x, tau0)
%   result = sigmatau.dev.tdev(x, tau0, m_list)
%   result = sigmatau.dev.tdev(x, tau0, m_list, 'data_type', 'freq')
%
%   SP1065 §4: TDEV(tau) = tau * MDEV(tau) / sqrt(3)
%   Does NOT call engine directly; derives from MDEV for consistency.

if nargin < 3, m_list = []; end

TDEV_MDEV_PREFACTOR = sqrt(3);   % SP1065 §4: TDEV = tau * MDEV / √3

mr    = sigmatau.dev.mdev(x, tau0, m_list, varargin{:});
scale = mr.tau / TDEV_MDEV_PREFACTOR;   % row vector

result = mr;
result.deviation = scale .* mr.deviation;
result.ci        = mr.ci .* repmat(scale(:), 1, 2);
result.method    = 'tdev';
end
