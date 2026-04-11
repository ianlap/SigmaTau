function edf = totaldev_edf(varType, alpha, T, tau)
%TOTALDEV_EDF   EDF calculator for TOTDEV, MTOTDEV, and HTOTDEV
%
%   edf = totaldev_edf(varType, alpha, T, tau)
%
%   Computes the equivalent degrees-of-freedom for total variance estimators:
%     • TOTVAR (standard total deviation)
%     • MTOT   (modified total deviation)
%     • HTOT   (Hadamard total deviation)
%   using formulas and coefficients from:
%     – Frequency Stability Handbook (Riley & Greenhall, §5.4.2–5.4.3)
%     – FCS 2001 IEEE Paper: "Definitions of 'total' estimators..."
%
%   Inputs:
%     varType – 'totvar' | 'mtot' | 'htot'
%     alpha   – power-law noise exponent:
%                +2: WHPM, +1: FLPM, 0: WHFM, -1: FLFM, -2: RWFM
%     T       – record duration [s]
%     tau     – averaging time [s] (i.e. tau = m * tau0)
%
%   Output:
%     edf     – scalar equivalent degrees-of-freedom
%
%   Examples:
%     edf = totaldev_edf('totvar',  0,   10000, 10);   % White FM
%     edf = totaldev_edf('mtot',   -2,   86400, 60);   % RWFM
%     edf = totaldev_edf('htot',   -1,   3600,  30);   % FLFM

varType = lower(varType);

switch varType
    case 'totvar'
        [b, c] = coeff_totvar(alpha);
        edf = b * (T / tau) - c;

    case 'mtot'
        [b, c] = coeff_mtot(alpha);
        edf = b * (T / tau) - c;

    case 'htot'
        [b0, b1] = coeff_htot(alpha);
        edf = (T / tau) / (b0 + b1 * (tau / T));

    case 'mhtot'
        [b,c] = coeff_mhtot(alpha);
        edf = b * (T/tau) - c;
    otherwise
        error('Unsupported varType "%s". Use "totvar", "mtot", or "htot".', varType);
end

if edf <= 0 || isnan(edf)
    %warning('EDF is non-positive or undefined: check alpha, T, tau.');
end
end

% -------------------- Coefficient Tables ----------------------------
function [b,c] = coeff_totvar(alpha)
    switch alpha
        case 0,   b=1.50; c=0.00;   % White FM
        case -1,  b=1.17; c=0.22;   % Flicker FM
        case -2,  b=0.93; c=0.36;   % Random walk FM
        otherwise, b=NaN; c=NaN;
    end
end

function [b,c] = coeff_mtot(alpha)
    switch alpha
        case 2,   b=1.90; c=2.10;   % White PM
        case 1,   b=1.20; c=1.40;   % Flicker PM
        case 0,   b=1.10; c=1.20;   % White FM
        case -1,  b=0.85; c=0.50;   % Flicker FM
        case -2,  b=0.75; c=0.31;   % Random walk FM
        otherwise, b=NaN; c=NaN;
    end
end


function [b,c] = coeff_mhtot(alpha)
    switch alpha
        case 2,   b=3.904; c=9.640;   % White PM
        case 1,   b=2.656; c=11.093;   % Flicker PM
        case 0,   b=2.275; c=8.701;   % White FM
        case -1,  b=1.964; c=4.908;   % Flicker FM
        case -2,  b=1.572; c=4.534;   % Random walk FM
        otherwise, b=NaN; c=NaN;
    end
end

function [b0,b1] = coeff_htot(alpha)
    switch alpha
        case 0     % White FM
            b0 = 0.546;
            b1 = 1.41;
        case -1    % Flicker FM
            b0 = 0.667;
            b1 = 2.00;
        case -2    % RWFM
            b0 = 0.909;
            b1 = 1.00;
        otherwise
            b0 = NaN;
            b1 = NaN;
    end
end
