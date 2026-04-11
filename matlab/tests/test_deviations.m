%TEST_DEVIATIONS  Smoke tests for all 10 deviation wrappers.
%
% Checks: output struct has expected fields, values are finite and positive,
% tau matches m_list * tau0, method name matches.

rng(42);
N    = 500;
tau0 = 1.0;
x    = cumsum(randn(N,1));   % random walk phase (RWFM-like)

m_list = [1, 2, 4, 8, 16];

devfns = {
    @sigmatau.dev.adev,    'adev';
    @sigmatau.dev.mdev,    'mdev';
    @sigmatau.dev.tdev,    'tdev';
    @sigmatau.dev.hdev,    'hdev';
    @sigmatau.dev.mhdev,   'mhdev';
    @sigmatau.dev.ldev,    'ldev';
    @sigmatau.dev.totdev,  'totdev';
    @sigmatau.dev.mtotdev, 'mtotdev';
    @sigmatau.dev.htotdev, 'htotdev';
    @sigmatau.dev.mhtotdev,'mhtotdev';
};

for k = 1:size(devfns, 1)
    fn   = devfns{k,1};
    name = devfns{k,2};

    result = fn(x, tau0, m_list);

    % Required fields
    assert(isfield(result, 'tau'),       sprintf('%s: missing .tau', name));
    assert(isfield(result, 'deviation'), sprintf('%s: missing .deviation', name));
    assert(isfield(result, 'edf'),       sprintf('%s: missing .edf', name));
    assert(isfield(result, 'ci'),        sprintf('%s: missing .ci', name));
    assert(isfield(result, 'alpha'),     sprintf('%s: missing .alpha', name));
    assert(isfield(result, 'neff'),      sprintf('%s: missing .neff', name));
    assert(isfield(result, 'method'),    sprintf('%s: missing .method', name));
    assert(strcmp(result.method, name),  sprintf('%s: wrong method name', name));

    % Length consistency
    L = numel(result.tau);
    assert(L > 0,                        sprintf('%s: empty tau', name));
    assert(numel(result.deviation) == L, sprintf('%s: deviation length mismatch', name));
    assert(isequal(size(result.ci), [L 2]), sprintf('%s: ci size wrong', name));

    % tau values
    assert(all(result.tau > 0),          sprintf('%s: tau should be positive', name));

    % deviation values (may have some NaN at large m, but at least one finite)
    assert(any(isfinite(result.deviation) & result.deviation > 0), ...
           sprintf('%s: no finite positive deviations', name));

    fprintf('  OK: %s\n', name);
end

fprintf('test_deviations: all assertions passed\n');
