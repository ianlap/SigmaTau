---
name: deviation-engine
description: >
  Use when building, modifying, or debugging any deviation function (adev, mdev,
  hdev, mhdev, totdev, mtotdev, htotdev, mhtotdev, tdev, ldev) or the shared
  engine that powers them. Trigger when working in +sigmatau/+dev/ or
  julia/src/deviations.jl or julia/src/engine.jl.
---

# Deviation Engine Architecture

## The Problem
All 10 deviation functions repeat identical boilerplate: column-ize input, generate
default m_list, preallocate NaN arrays, call noise_id, loop over m_list, compute
kernel, call calculate_edf, call compute_ci. ~500 lines of duplication.

## The Solution
One shared `engine.m` that accepts a kernel function handle + parameter struct.
Each deviation is a thin wrapper (~15 lines) that defines its kernel and params.

## Engine Signature (MATLAB)
```matlab
function result = engine(x, tau0, m_list, kernel, params)
% Inputs:
%   kernel – @(x, m, tau0) → [variance, neff]
%   params – struct with:
%     .name        – string identifier ('adev','mdev',...)
%     .min_factor  – min N/m ratio for m_list generation (2,3,4)
%     .d           – difference order (2=Allan, 3=Hadamard)
%     .F_fn        – @(m) → F for EDF (m=unmodified, 1=modified)
%     .dmin, .dmax – noise_id differencing bounds
%     .is_total    – bool: use totaldev_edf instead of calculate_edf
%     .total_type  – string for totaldev_edf ('totvar','mtot','htot','mhtot')
%     .needs_bias  – bool: apply bias correction
%     .bias_type   – string for bias_correction
% Output:
%   result – struct: tau, dev, edf, ci, alpha, neff, method
```

## Wrapper Example
```matlab
function result = adev(x, tau0, m_list)
    if nargin < 3, m_list = []; end
    kernel = @adev_kernel;
    params = struct('name','adev', 'min_factor',2, 'd',2, ...
                    'F_fn',@(m) m, 'dmin',0, 'dmax',1, ...
                    'is_total',false, 'needs_bias',false);
    result = sigmatau.dev.engine(x, tau0, m_list, kernel, params);
end

function [v, neff] = adev_kernel(x, m, tau0)
    N = numel(x); L = N - 2*m;
    d2 = x(1+2*m:N) - 2*x(1+m:N-m) + x(1:L);
    v = mean(d2.^2) / (2 * m^2 * tau0^2);  % returns VARIANCE
    neff = L;
end
```

## Gotchas
- Engine returns variance from kernel, takes sqrt itself. Kernels return variance, not deviation.
- Total deviations (totdev, mtotdev, htotdev, mhtotdev) have more complex kernels that include
  detrending and reflection. These kernels are longer but still follow the same interface.
- tdev and ldev don't use the engine directly — they call mdev/mhdev and scale the result.
- mdev uses cumsum-based prefix sums for O(N) computation. Don't rewrite as O(N*m) loops.

## Deviation Parameters Quick Reference
| Dev      | d | min_factor | F_fn   | is_total | needs_bias |
|----------|---|------------|--------|----------|------------|
| adev     | 2 | 2          | @(m) m | false    | false      |
| mdev     | 2 | 3          | @(m) 1 | false    | false      |
| hdev     | 3 | 4          | @(m) m | false    | false      |
| mhdev    | 3 | 4          | @(m) 1 | false    | false      |
| totdev   | 2 | 2          | @(m) m | true     | true       |
| mtotdev  | 2 | 3          | @(m) 1 | true     | false      |
| htotdev  | 3 | 3          | @(m) m | true     | true       |
| mhtotdev | 3 | 4          | @(m) 1 | true     | false      |
| tdev     | — | —          | —      | (wraps mdev)          |
| ldev     | — | —          | —      | (wraps mhdev)         |
