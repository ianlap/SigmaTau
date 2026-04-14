# SigmaTau — Project TODO

This document outlines future development tasks, prioritized by impact and estimated by difficulty.

## Priority: High (Core Accuracy & Stability)

| Task | Description | Difficulty |
| :--- | :--- | :--- |
| **Mtot multi-noise validation** | Verify mtot deviation and bias correction across all 5 standard noise types (WPM to RWFM) against Stable32. | 🟢 Easy |
| **MATLAB KF Unit Tests** | Add dedicated unit tests for the ported Kalman Filter functions in `+kf/` (matching the rigor of `julia/test/test_filter.jl`). | 🟢 Easy |
| **MHTOTDEV EDF Model** | Refine the MHTOTDEV EDF coefficients. Current implementation uses approximations; a more rigorous Monte Carlo or analytical fit is needed. | 🔴 Hard |
| **Noise ID Scaling** | Optimize the lag-1 ACF identification for extremely large datasets (N > 10^7) by implementing block-processing or decimation-based estimation. | 🟡 Medium |
| **IRWFM Integration** | Deepen the integration of Integrated Random Walk FM (IRWFM) across all deviation kernels and the Kalman Filter Q-matrix validation. | 🟡 Medium |

## Priority: Medium (Usability & Tooling)

| Task | Description | Difficulty |
| :--- | :--- | :--- |
| **Julia Plotting Extensions** | Expand the Julia package extension for `Plots.jl` to include standard stability plots (Sigma-Tau, Phase/Freq residuals) with proper log-log scaling. | 🟡 Medium |
| **MATLAB Plotting Package** | Create a `+sigmatau/+plot` package to provide high-quality, publication-ready visualizations of deviation results. | 🟡 Medium |
| **Clock Steering Examples** | Add more comprehensive examples in `examples/` demonstrating PID steering against real-world clock data (e.g., GPS vs. H-Maser). | 🟢 Easy |
| **Documentation Guide** | Create a high-level architectural guide in `docs/` explaining the shared engine pattern and Kalman Filter kinematics. | 🟢 Easy |

## Priority: Medium (Documentation Program — Handbook Rollout)

| Task | Description | Difficulty |
| :--- | :--- | :--- |
| **CLI Reference Parity** | Ensure handbook command docs match `julia/cli/SigmaTauCLI/src/commands/output.jl` `HELP_LINES` and parser behavior in `parser.jl`. | 🟢 Easy |
| **Script Argument Coverage** | Add syntax/argument documentation for every user-facing script under `scripts/julia`, `scripts/python`, and `scripts/matlab`. | 🟢 Easy |
| **Data Contract Docs** | Document accepted input formats (single-column phase, two-column MJD+phase), `tau0` inference rules, and output folder conventions. | 🟢 Easy |
| **Workflow Recipes** | Publish practical “runbook” flows (quick all-dev run, full KF loop, validation report generation) with expected artifacts. | 🟢 Easy |
| **Docs Definition of Done** | Add a checklist requiring each public tool page to include purpose, syntax, options table, examples, outputs, and troubleshooting. | 🟢 Easy |

## Priority: Low (Technical Debt & Refinement)

| Task | Description | Difficulty |
| :--- | :--- | :--- |
| **Continuous Integration** | Expand GitHub Actions to include automated cross-validation between MATLAB and Julia on every PR. | 🟡 Medium |
| **Namespace Cleanup** | Final review of all private helpers to ensure consistent naming and location across both language implementations. | 🟢 Easy |
| **Type Safety (Julia)** | Audit Julia source for any remaining type instabilities in the inner loops of the deviation kernels. | 🟡 Medium |

---
*Difficulty Legend: 🟢 Easy (Hours), 🟡 Medium (Days), 🔴 Hard (Weeks)*
