# Cross-Validation Task

Run a full MATLAB vs Julia parity check for the component: $ARGUMENTS

1. **Source Julia**: Read the implementation in `julia/src/`.
2. **Source MATLAB**: Read the implementation in `matlab/+sigmatau/`.
3. **Compare Kernels**: Verify that the arithmetic is identical (normalizers, difference orders, Ne/L formulas).
4. **Compare EDF**: Verify that the EDF calculation (SP1065/Greenhall) is identical.
5. **Run Test**: Execute the relevant MATLAB test (e.g., `tests/test_crossval_julia.m`) if it exists.
6. **Assert Parity**: Report max relative error. Target: < 1e-12.

If any field (tau, dev, edf, ci) has a relative error > 1e-12, stop and explain the discrepancy.
