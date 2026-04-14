# Known Discrepancies and Open Issues

| # | Location | Issue | Status |
|---|----------|-------|--------|
| 1 | MDEV / MHDEV | Code kernels carry an explicit inner `1/m` not present in SP1065 Eq. 16 or MB23 §4.4.3 | ✓ Algebraic artifact of the prefix-sum / third-difference (G97) formulation — outer normalization is `1/(2m²τ₀²·N_e)` instead of `1/(2m⁴τ₀²·N_e)`; the two forms are identical. No source typo. |
| 2 | `htotdev` EDF loop | CLAUDE.md flags potential off-by-one: loop over `numel(tau)` vs `numel(valid)` after trimming | ⚠ Not audited in this pass |
| 3 | `mhtotdev` Neff | CLAUDE.md flags: is segment count `N−4m+1` or `N−3m`? | ✓ Both MATLAB and Julia use `N−4m+1`; consistent with HV99 / FCS01 total methodology |
| 4 | MATLAB KF | `matlab/+sigmatau/+kf/` implementation available. | ✓ Ported from Julia |
| 5 | MHTOT reference attribution | Prior docs cited "Howe & Schlossberger, FCS 2001" — no such paper exists | ✓ Replaced: FCS01 = Howe/Beard/Greenhall/Vernotte/Riley 2001 (HTOT paper). MHTOT itself has no dedicated canonical reference; inferred from HV99 (MTOT) + FCS01 (HTOT). Code comments still say "FCS 2001" for MHTOT coefficients — understood to mean the HTOT table applied as approximation. |
| 6 | LDEV | Prior docs called it "Loran-C deviation" and cited SP1065 §5.6; §5.6 is "Bias Functions" — citation was bogus and formula `(τ²/6)·MHVAR` did not match the code's `√(10/3)` prefactor | ✓ Renamed **Lapinski Deviation**; formula updated to `LVAR = (3τ²/10)·MHVAR` to match MATLAB + Julia |
