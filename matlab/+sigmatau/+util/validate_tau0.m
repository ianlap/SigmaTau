function tau0 = validate_tau0(tau0)
% VALIDATE_TAU0  Check that tau0 is positive and finite.
%
%   tau0 = sigmatau.util.validate_tau0(tau0)

tau0 = double(tau0);
if ~(isfinite(tau0) && tau0 > 0)
    error('SigmaTau:validate', 'tau0 must be positive and finite, got %g.', tau0);
end
end
