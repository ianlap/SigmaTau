# SigmaTau — Equation Reference Index

This document maps each algorithm to the authoritative source equation, notes any discrepancies
between sources, and records audit status.

**Primary references** (PDFs in `docs/papers/`):

| Sigil | Full citation |
|-------|---------------|
| **MB23** | Banerjee & Matsakis, *An Introduction to Modern Timekeeping and Time Transfer*, Springer 2023 (ISBN 978-3-031-30779-9). → `2023_banerjee_matsakis_timekeeping_book.pdf` |
| **SP1065** | Riley, *Handbook of Frequency Stability Analysis*, NIST SP1065, 2008. → `sp1065.pdf` |
| **G97** | Greenhall, "The Third-Difference Approach to Modified Allan Variance," IEEE T-IM 46(3), June 1997. → `1997_greenhall_third_difference_mvar_ieeetim.pdf` |
| **GHP99** | Greenhall, Howe & Percival, "Total Variance, an Estimator of Long-Term Frequency Stability," IEEE UFFC 46(5), Sept 1999. → `1999_greenhall_howe_percival_total_variance_ieee.pdf` |
| **HV99** | Howe & Vernotte, "Generalization of the Total Variance Approach to the Modified Allan Variance," PTTI 1999 (31st). → `1999_howe_vernotte_total_mvar_ptti.pdf` — canonical **MTOT** reference. |
| **H00** | Howe, Beard, Greenhall, Vernotte, Riley, "A Total Estimator of the Hadamard Function Used for GPS Operations," PTTI 2000 (32nd). → `2000_howe_total_estimator_hadamard_ptti.pdf` |
| **FCS01** | Howe, Beard, Greenhall, Vernotte, Riley, "Total Hadamard Variance: Application to Clock Steering by Kalman Filtering," Proc. IEEE FCS 2001. → `2001_howe_total_hadamard_variance_fcs.pdf` — canonical **HTOT** reference; provides bias `a(α)` table used by `totaldev_edf` and `bias` (code's "FCS 2001"). |
| **GR03** | Greenhall & Riley, "Uncertainty of Stability Variances Based on Finite Differences," PTTI 2003. → `2003_greenhall_riley_uncertainty_stability_variances_ptti.pdf` |
| **RG04** | Riley & Greenhall, "Power Law Noise Identification Using the Lag 1 Autocorrelation," 18th EFTF 2004. → `2004_riley_greenhall_lag1_acf_noiseid.pdf` |
| **H05** | Howe et al., "Enhancements to GPS Operations and Clock Evaluations Using a 'Total' Hadamard Deviation," IEEE UFFC 52(8), Aug 2005. → `2005_howe_total_hadamard_ieee.pdf` |

## Sections

- [Allan Family](allan.md) (ADEV, MDEV, TDEV)
- [Hadamard Family](hadamard.md) (HDEV, MHDEV, LDEV)
- [Total Family](total.md) (TOTDEV, MTOTDEV, HTOTDEV, MHTOTDEV)
- [Kalman Filter](kalman.md) (State, Transition, Noise, Update, Steering)
- [Known Discrepancies](discrepancies.md) (Open issues and audit notes)
