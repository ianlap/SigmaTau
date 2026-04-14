function [v, neff] = mdev_kernel(x, m, tau0)
% MDEV_KERNEL  Modified Allan variance kernel for sigmatau.dev.engine.
%
%   Uses cumsum prefix sums for O(N) computation per m.
%   SP1065 Eq. 15: MVAR(tau) = sum((s3 - 2s2 + s1)^2) / (Ne * 2 * m^2 * tau0^2)
N  = numel(x);
Ne = N - 3*m + 1;
if Ne <= 0
    v = NaN; neff = 0;
    return;
end
S  = cumsum([0; x]);
s1 = S(1+m:Ne+m)     - S(1:Ne);
s2 = S(1+2*m:Ne+2*m) - S(1+m:Ne+m);
s3 = S(1+3*m:Ne+3*m) - S(1+2*m:Ne+2*m);
d  = (s3 - 2*s2 + s1) / m;
v  = sum(d.^2) / (Ne * 2 * m^2 * tau0^2);
neff = Ne;
end
