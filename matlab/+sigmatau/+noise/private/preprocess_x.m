function x_out = preprocess_x(x)
% PREPROCESS_X  Remove >5σ outliers then linear detrend.
%
%   x_out = preprocess_x(x)
%
%   Mirrors Julia _preprocess: removes outliers outside the 5-sigma range
%   relative to the mean, then performs a linear detrend.
x = x(:);
x_mean = mean(x);
x_std  = std(x);
if x_std < eps
    x_out = sigmatau.util.detrend(x, 1);
    return;
end
z     = abs((x - x_mean) / x_std);
x_out = sigmatau.util.detrend(x(z < 5.0), 1);
end
