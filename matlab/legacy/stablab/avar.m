function avar = avar(x, tau)
%AVAR  Classic Allan variance for phase data as in SP1065 Eq. (6)
%
%   avar = avar(x, tau)
%   Computes Allan variance from phase data spaced at interval tau.
%
%   Inputs:
%       x   - phase data vector (equally spaced samples)
%       tau - spacing interval (seconds)
%
%   Output:
%       avar - Allan variance at averaging time tau

    x = x(:);
    N = length(x);
    if N < 3
        error('Need at least 3 points for AVAR');
    end

    % Compute 2nd differences
    v = x(3:end) - 2*x(2:end-1) + x(1:end-2);
    avar = sum(v.^2) / (2 * (N - 2) * tau^2);
end
