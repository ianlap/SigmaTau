import numpy as np
import allantools as at
import sys

def test_adev_wfm():
    # Load cross-validation phase data
    x = np.loadtxt("matlab/tests/crossval_phase_wfm.txt")
    
    # Octave m-list: 1, 2, 4, 8, 16, 32, 64
    m_list = np.array([1, 2, 4, 8, 16, 32, 64])
    tau0 = 1.0
    taus = m_list * tau0
    
    # Compute overlapping Allan deviation using allantools
    (taus_at, adev_at, adev_err, n) = at.oadev(x, rate=1.0/tau0, data_type="phase", taus=taus)
    
    # SigmaTau expected results (from matlab/tests/crossval_results.txt)
    # adev	wfm	1	1.00000000000000000e+00	1.01087503038588644e+00	8.00030297872340384e+02	0
    # adev	wfm	2	2.00000000000000000e+00	7.16863736086823078e-01	3.64172370877411311e+02	0
    # adev	wfm	4	4.00000000000000000e+00	5.13015624597761288e-01	1.74828290524763446e+02	0
    # adev	wfm	8	8.00000000000000000e+00	3.50683244221281776e-01	8.54542119868584678e+01	0
    # adev	wfm	16	1.60000000000000000e+01	2.41230516275196677e-01	4.18503511996173287e+01	0
    # adev	wfm	32	3.20000000000000000e+01	2.03194435968464343e-01	2.02943315329783118e+01	0
    # adev	wfm	64	6.40000000000000000e+01	1.37393223137220244e-01	7.76815619284253422e+00	1

    expected_adev = np.array([
        1.01087503038588644e+00,
        7.16863736086823078e-01,
        5.13015624597761288e-01,
        3.50683244221281776e-01,
        2.41230516275196677e-01,
        2.03194435968464343e-01,
        1.37393223137220244e-01
    ])

    # Check against allantools
    print(f"Checking ADEV (WFM) against allantools...")
    rel_error = np.abs(adev_at - expected_adev) / expected_adev
    max_rel_error = np.max(rel_error)
    print(f"Max relative error: {max_rel_error:.2e}")
    
    if max_rel_error < 1e-12:
        print("Test PASSED.")
    else:
        print("Test FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    test_adev_wfm()
