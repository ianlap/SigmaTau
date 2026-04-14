import numpy as np
import allantools as at
import pandas as pd
import sys
import os

def test_allantools_vs_stable32():
    # 1. Load data
    data_path = "reference/stable32gen.DAT"
    # Skip header lines (marked with # or descriptive text)
    # The header ends after "# Header End"
    phase_data = []
    with open(data_path, 'r') as f:
        header_passed = False
        for line in f:
            if "# Header End" in line:
                header_passed = True
                continue
            if header_passed:
                try:
                    phase_data.append(float(line.strip()))
                except ValueError:
                    continue
    
    x = np.array(phase_data)
    tau0 = 1.0
    
    # 2. Load Stable32 summary
    stable32_summary_path = "reference/stable32out/stable32_summary.csv"
    s32_df = pd.read_csv(stable32_summary_path)
    
    # Filter for Overlapping Allan (as SigmaTau's adev is overlapping by default)
    s32_oadev = s32_df[s32_df['Type'] == 'Overlapping Allan'].copy()
    
    # 3. Compute with allantools
    taus = s32_oadev['Tau'].values
    (taus_at, adev_at, adev_err, n) = at.oadev(x, rate=1.0/tau0, data_type="phase", taus=taus)
    
    # 4. Compare
    print(f"{'Tau':>10} | {'Stable32':>15} | {'allantools':>15} | {'Rel Error':>10}")
    print("-" * 55)
    
    max_rel_err = 0
    for i, tau in enumerate(taus):
        s32_val = s32_oadev.iloc[i]['Sigma']
        at_val = adev_at[i]
        rel_err = abs(s32_val - at_val) / s32_val
        max_rel_err = max(max_rel_err, rel_err)
        print(f"{tau:10.1e} | {s32_val:15.10e} | {at_val:15.10e} | {rel_err:10.2e}")
    
    print("-" * 55)
    print(f"Max relative error: {max_rel_err:.2e}")
    
    # We expect a small discrepancy (maybe 1e-4 to 1e-6) due to float precision in CSV or minor alg differences
    # but they should be in the same ballpark.
    if max_rel_err < 1e-3:
        print("Comparison PASSED (within 0.1% ballpark).")
    else:
        print("Comparison FAILED (outside 0.1% ballpark).")
        sys.exit(1)

if __name__ == "__main__":
    test_allantools_vs_stable32()
