function B1 = b1_theory(N, mu)
% B1_THEORY  Theoretical B1 = classical-var / Allan-var vs. noise slope mu.
%
%   B1 = b1_theory(N, mu)
%
%   mu is defined by sigma_y^2(tau) ~ tau^mu.
%   Closed forms for integer mu; SP1065 Eq. 73 (Howe-Beard 1998) otherwise.
%     mu = +2 → FW FM (alpha=-3)
%     mu = +1 → RWFM  (alpha=-2)
%     mu =  0 → FLFM  (alpha=-1)
%     mu = -1 → WHFM  (alpha= 0)
%     mu = -2 → WHPM/FLPM (alpha=2,1)
switch mu
    case  2; B1 = N*(N+1)/6;                        % FW FM
    case  1; B1 = N/2;                              % RWFM
    case  0; B1 = N*log(N) / (2*(N-1)*log(2));      % FLFM
    case -1; B1 = 1.0;                              % WHFM (reference)
    case -2; B1 = (N^2 - 1) / (1.5 * N * (N-1));    % WHPM/FLPM
    otherwise
        B1 = (N * (1 - N^mu)) / (2 * (N-1) * (1 - 2^mu));   % SP1065 Eq. 73
end
end
