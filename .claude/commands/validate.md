Cross-validate SigmaTau implementations between MATLAB and Julia.

1. Generate test data: white PM, white FM, and RWFM noise, N=10000, tau0=1.0
2. Run the function named $ARGUMENTS through both MATLAB and Julia implementations
3. Compare outputs: tau, deviation values, edf, ci, alpha
4. Report maximum relative error for each output field
5. PASS if all relative errors < 1e-10. FAIL otherwise with details.

Use `matlab -batch` for MATLAB and `julia --project=../julia` for Julia.
