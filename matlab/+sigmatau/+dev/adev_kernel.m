function [v, neff] = adev_kernel(x, m, tau0)
% ADEV_KERNEL  Allan variance kernel for sigmatau.dev.engine.
%
%   SP1065 Eq. 14: AVAR(tau) = mean((x(n+2m) - 2x(n+m) + x(n))^2) / (2*m^2*tau0^2)
N = numel(x);
L = N - 2*m;
if L <= 0
    v = NaN; neff = 0;
    return;
end
d2   = x(1+2*m:N) - 2*x(1+m:N-m) + x(1:L);
v    = sum(d2.^2) / (L * 2 * m^2 * tau0^2);
neff = L;
end
