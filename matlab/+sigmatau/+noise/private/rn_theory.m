function Rn = rn_theory(af, b)
% RN_THEORY  Theoretical R(n) = MVAR/AVAR ratio.
%
%   Rn = rn_theory(af, b)
%
%   SP1065 §5.6 / Riley §5.2.6.
%   Used to resolve WHPM (b=0) vs. FLPM (b=-1) after the B1 ratio test.
switch b
    case 0
        Rn = 1.0 / af;                              % WHPM asymptotic: R → 1/m
    case -1
        % FLPM: leading-order MVAR/AVAR ratio
        % Riley §5.2.6 (Eq 5.7/5.8)
        avar = (1.038 + 3*log(2*pi*0.5*af)) / (4*pi^2);
        mvar = 3*log(256/27) / (8*pi^2);
        Rn   = mvar / avar;
    otherwise
        Rn = 1.0;
end
end
