function alpha_list = noise_id(x, m_list, data_type, dmin, dmax)
% NOISE_ID  Dominant power-law noise estimator. Thin wrapper for identify().
%
%   alpha_list = sigmatau.noise.noise_id(x, m_list, data_type, dmin, dmax)
%
%   See also: sigmatau.noise.identify

if nargin < 3 || isempty(data_type), data_type = 'phase'; end
if nargin < 4 || isempty(dmin),      dmin = 0; end
if nargin < 5 || isempty(dmax),      dmax = 2; end

alpha_list = sigmatau.noise.identify(x, m_list, data_type, dmin, dmax);
end
