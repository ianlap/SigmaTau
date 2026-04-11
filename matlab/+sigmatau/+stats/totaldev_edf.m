function edf = totaldev_edf(var_type, alpha, T, tau)
% TOTALDEV_EDF  EDF for total deviation variants via lookup-table coefficients.
%
%   edf = sigmatau.stats.totaldev_edf(var_type, alpha, T, tau)
%
%   Inputs:
%     var_type – 'totvar' | 'mtot' | 'htot' | 'mhtot'
%     alpha    – power-law noise exponent
%     T        – record duration (s), (N-1)*tau0
%     tau      – averaging time (s)
%
%   Reference: SP1065 Table 9–10; Greenhall (2003); FCS 2001.

switch lower(var_type)
    case 'totvar'
        [b, c] = coeff_totvar(alpha);
        edf = b*(T/tau) - c;
    case 'mtot'
        [b, c] = coeff_mtot(alpha);
        edf = b*(T/tau) - c;
    case 'htot'
        [b0, b1] = coeff_htot(alpha);
        edf = (T/tau) / (b0 + b1*(tau/T));
    case 'mhtot'
        [b, c] = coeff_mhtot(alpha);
        edf = b*(T/tau) - c;
    otherwise
        edf = NaN;
end
end

function [b,c] = coeff_totvar(alpha)
% SP1065 Table 9
switch alpha
    case  0;  b=1.50; c=0.00;
    case -1;  b=1.17; c=0.22;
    case -2;  b=0.93; c=0.36;
    otherwise; b=NaN; c=NaN;
end
end

function [b,c] = coeff_mtot(alpha)
% SP1065 Table 10
switch alpha
    case  2;  b=1.90; c=2.10;
    case  1;  b=1.20; c=1.40;
    case  0;  b=1.10; c=1.20;
    case -1;  b=0.85; c=0.50;
    case -2;  b=0.75; c=0.31;
    otherwise; b=NaN; c=NaN;
end
end

function [b0,b1] = coeff_htot(alpha)
% Greenhall (2003) Table 1
switch alpha
    case  0;  b0=0.546; b1=1.41;
    case -1;  b0=0.667; b1=2.00;
    case -2;  b0=0.909; b1=1.00;
    otherwise; b0=NaN; b1=NaN;
end
end

function [b,c] = coeff_mhtot(alpha)
% FCS 2001 coefficients (approximate — no published model for mhtotdev)
switch alpha
    case  2;  b=3.904; c=9.640;
    case  1;  b=2.656; c=11.093;
    case  0;  b=2.275; c=8.701;
    case -1;  b=1.964; c=4.908;
    case -2;  b=1.572; c=4.534;
    otherwise; b=NaN; c=NaN;
end
end
