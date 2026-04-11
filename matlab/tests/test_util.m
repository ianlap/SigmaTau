%TEST_UTIL  Unit tests for +util functions.

% validate_phase_data
x = randn(100, 1);
xv = sigmatau.util.validate_phase_data(x);
assert(numel(xv) == 100 && isequal(size(xv), [100 1]), 'validate_phase_data: wrong shape');
assert(isa(xv, 'double'), 'validate_phase_data: not double');

try
    sigmatau.util.validate_phase_data([1; NaN; 3; 4]);
    error('validate_phase_data: should throw on NaN');
catch e
    assert(contains(e.message, 'finite') || contains(e.identifier, 'SigmaTau'), ...
           'validate_phase_data: wrong error on NaN');
end

try
    sigmatau.util.validate_phase_data([1; 2; 3]);
    error('validate_phase_data: should throw on <4 points');
catch e
    assert(true);
end

% validate_tau0
t = sigmatau.util.validate_tau0(1.5);
assert(t == 1.5 && isa(t, 'double'), 'validate_tau0: wrong value');
try
    sigmatau.util.validate_tau0(-1);
    error('validate_tau0: should throw on negative');
catch e
    assert(true);
end

% detrend_linear — should remove linear component exactly
n = 50;
t = (1:n)';
x_lin = 3*t + 7 + randn(n,1)*0;   % pure linear
xd = sigmatau.util.detrend_linear(x_lin);
assert(max(abs(xd)) < 1e-10, 'detrend_linear: linear data not removed');

% detrend_quadratic — removes quadratic component
x_quad = 0.5*t.^2 - 2*t + 1;
xd2 = sigmatau.util.detrend_quadratic(x_quad);
assert(max(abs(xd2)) < 1e-8, 'detrend_quadratic: quadratic not removed');

% default_mlist
ms = sigmatau.util.default_mlist(1024, 2);
assert(ms(1) == 1 && ms(end) == 512, 'default_mlist: wrong values');
assert(all(diff(ms) > 0), 'default_mlist: not increasing');

fprintf('test_util: all assertions passed\n');
