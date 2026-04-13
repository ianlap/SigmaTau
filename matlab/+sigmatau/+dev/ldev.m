function result = ldev(x, tau0, m_list, varargin)
% LDEV  Lapinski deviation (LDEV). Wraps mhdev, scales by tau/sqrt(10/3).
%
%   result = sigmatau.dev.ldev(x, tau0)
%   result = sigmatau.dev.ldev(x, tau0, m_list)
%   result = sigmatau.dev.ldev(x, tau0, m_list, 'data_type', 'freq')
%
%   LDEV(tau) = tau * MHDEV(tau) / sqrt(10/3)
%   Does NOT call engine directly; derives from MHDEV for consistency.

if nargin < 3, m_list = []; end

LDEV_MHDEV_PREFACTOR = sqrt(10/3);   % LDEV = tau * MHDEV / √(10/3)

mr    = sigmatau.dev.mhdev(x, tau0, m_list, varargin{:});
scale = mr.tau / LDEV_MHDEV_PREFACTOR;   % row vector

result = mr;
result.deviation = scale .* mr.deviation;
result.ci        = mr.ci .* repmat(scale(:), 1, 2);
result.method    = 'ldev';
end
