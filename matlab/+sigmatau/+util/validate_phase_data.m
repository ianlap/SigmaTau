function x = validate_phase_data(x)
% VALIDATE_PHASE_DATA  Check phase data and return as column Float64 vector.
%
%   x = sigmatau.util.validate_phase_data(x)
%
%   Throws if data contains NaN/Inf or has fewer than 4 points.
%   Returns a double column vector.

x = x(:);
if ~all(isfinite(x))
    error('SigmaTau:validate', 'Phase data must be finite (no NaN or Inf).');
end
if numel(x) < 4
    error('SigmaTau:validate', 'Phase data must have at least 4 points.');
end
x = double(x);
end
