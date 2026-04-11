function B = bias_correction(alpha, var_type, tau, T)
% BIAS_CORRECTION  Bias factor B(alpha) for TOTVAR, MTOT, and HTOT corrections.
%
%   B = sigmatau.stats.bias_correction(alpha, var_type, tau, T)
%
%   Inputs:
%     alpha    – noise exponent (scalar or vector)
%     var_type – 'totvar' | 'mtot' | 'htot'
%     tau      – averaging time (scalar or vector, same size as alpha)
%     T        – record duration (s), (N-1)*tau0 (scalar)
%
%   Output:
%     B – bias factor for each tau (same size as alpha). Divide deviation by B.
%
%   Reference: SP1065 §§5.11–5.13; FCS 2001 Table 1.

alpha = double(alpha(:));
tau   = double(tau(:));
B     = ones(size(alpha));

switch lower(var_type)
    case 'totvar'
        for k = 1:numel(alpha)
            if alpha(k) == -1
                a = 1 / (3*log(2));   % ≈ 0.481, Flicker FM
                B(k) = 1 - a*(tau(k)/T);
            elseif alpha(k) == -2
                a = 0.75;             % RWFM
                B(k) = 1 - a*(tau(k)/T);
            else
                B(k) = 1;
            end
        end

    case 'mtot'
        % SP1065 Table 11
        table = containers.Map({'2','1','0','-1','-2'}, [1.06,1.17,1.27,1.30,1.31]);
        for k = 1:numel(alpha)
            key = num2str(alpha(k));
            if isKey(table, key)
                B(k) = table(key);
            else
                B(k) = 1;
            end
        end

    case 'htot'
        % FCS 2001 Table 1: a(alpha), B = 1/(1+a)
        table = containers.Map({'0','-1','-2','-3','-4'}, ...
                               [-0.005,-0.149,-0.229,-0.283,-0.321]);
        for k = 1:numel(alpha)
            key = num2str(alpha(k));
            if isKey(table, key)
                a = table(key);
                B(k) = 1 / (1 + a);
            else
                B(k) = 1;
            end
        end

    otherwise
        error('SigmaTau:bias_correction', ...
              'var_type must be ''totvar'', ''mtot'', or ''htot'', got ''%s''.', var_type);
end
end
