# Changelog

All notable changes to StabLab.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2025-08-09

### Added
- **Time Interval Error (TIE)**: RMS phase deviation measurement for telecom timing analysis
- **Maximum Time Interval Error (MTIE)**: Peak phase deviation for ITU-T timing requirements
- **Parabolic Deviation (PDEV)**: Drift-robust frequency stability measure with parabolic weighting
- **THEO1 Deviation**: Two-sample variance with improved confidence intervals and extended range
- **ITU-T Timing Masks**: G.811/G.812 mask generation for telecom compliance testing
- **Comprehensive test suite**: Validation of theoretical relationships and numerical stability
- **Time error module**: Dedicated `src/time_error.jl` for telecom timing functions

### Technical Improvements
- Efficient MTIE algorithms for both small and large observation windows
- Numerical stability guards against edge cases in parabolic fitting
- Support for even-only averaging factors in THEO1 (ITU requirement)
- Professional documentation with mathematical definitions and references

### Testing
- Theoretical slope validation for different noise types
- Drift rejection comparison between PDEV and Allan deviations  
- Edge case handling for single-point and large-window scenarios
- Cross-validation with AllanTools reference implementations

## [0.4.1] - 2025-08-09

### Added
- **Professional plotting functionality**: `stabplot()` creates publication-quality stability plots
- **Data loading utilities**: `load_phase_data()` with automatic scaling and format detection
- **Comprehensive reporting**: `stability_report()` generates multi-method analysis with plots
- **Results table printing**: Formatted statistical output with confidence intervals and EDF
- **Confidence interval visualization**: Shaded regions on plots for uncertainty bounds
- **Example demonstration**: Complete plotting example with rubidium clock data
- **Repository management**: Added `.gitignore` for cleaner version control

### Dependencies
- Added Plots.jl for professional visualization
- Added DelimitedFiles for flexible data loading
- Added Printf for formatted table output

### Technical Improvements
- Fixed Julia string interpolation in formatted output
- Support for both single and two-column data file formats
- Automatic plot labeling based on deviation type
- Log-log and linear scale options for flexibility

## [0.4.0] - 2025-08-09

### Added
- **Noise identification system**: Lag-1 ACF and B1 ratio methods for power-law noise characterization
- **Equivalent degrees of freedom (EDF)**: Full Greenhall algorithm implementation for all deviation types
- **Professional confidence intervals**: Chi-squared statistical bounds with Gaussian fallback
- **Two-tier CI system**: Simple σ/√n mode for fair benchmarking + full EDF-based analysis
- **Statistical integration**: All deviation functions now support on-demand confidence intervals via `compute_ci()`
- **Fair performance benchmarking**: Competitive with AllanTools when using equivalent statistical methods

### Improved  
- **Performance optimization**: Resolved 6x performance gap by implementing optional advanced statistics
- **API consistency**: Unified interface for both simple statistical errors and advanced confidence intervals
- **Documentation**: Comprehensive statistical methods documentation and usage examples

### Technical
- **Dependencies**: Added Distributions.jl for chi-squared calculations
- **Method flexibility**: `compute_ci(result, method="simple"|"full")` for different statistical approaches
- **Cross-validation**: Verified statistical correctness against MATLAB and Python implementations

## [0.3.0] - 2025-08-09

### Added
- **Complete NIST SP1065 deviation suite**: All 10 deviation types now implemented
- **Hadamard deviation (`hdev`)**: Overlapping third differences for robust drift rejection
- **Modified total deviation (`mtotdev`)**: Half-average detrending method
- **Hadamard total deviation (`htotdev`)**: SP1065 detrending method matching allantools/Stable32
- **Modified Hadamard total deviation (`mhtotdev`)**: Linear detrending with symmetric reflection
- **Comprehensive benchmark suite**: Three-way performance comparison (Julia vs Python vs MATLAB)
- **Professional project structure**: Organized tests/ and examples/ directories
- **Usage examples**: Complete demonstration of all features and API patterns

### Performance
- **Julia StabLab.jl is ~1.9x faster** than Python AllanTools for equivalent computations
- **Throughput**: ~0.61 Msamples/sec on reference hardware (500k sample benchmark)
- **Memory efficient**: Optimized algorithms with minimal memory allocation
- **Scalable**: Tested with datasets up to 10^7 samples

### Technical Improvements
- **Algorithm validation**: All functions tested against theoretical noise slopes
- **Cross-platform validation**: Results match MATLAB AllanLab and Python AllanTools
- **Mathematical relationships verified**: TDEV/MDEV and LDEV/MHDEV relationships confirmed
- **Type safety**: Comprehensive input validation and error handling
- **API consistency**: Unified interface across all 10 deviation types

### Project Organization
- **Clean directory structure**: Root contains only essential files
- **Comprehensive testing**: Individual tests for each deviation type plus full suite validation
- **Benchmark framework**: Professional performance comparison scripts
- **Development documentation**: CLAUDE.md with debugging workflows and known issues
- **Usage examples**: Practical demonstrations for new users

## [0.2.0] - 2025-08-09

### Added
- **Time deviation (`tdev`)**: Compute time deviation using TDEV = τ × MDEV / √3 relationship
- **Lapinski deviation (`ldev`)**: Compute Lapinski deviation using LDEV = τ × MHDEV / √(10/3) relationship
- **Total deviation (`totdev`)**: Uses all overlapping samples with detrending and symmetric reflection
- Comprehensive mathematical relationship validation in tests
- Enhanced test suite with theoretical slope validation for all deviation types
- Proper units handling (time deviations return seconds, frequency deviations are dimensionless)

### Changed
- Enhanced `DeviationResult` struct to support all deviation types consistently
- Improved test coverage with dedicated tests for TDEV/LDEV mathematical relationships
- Updated exports to include `tdev`, `ldev`, and `totdev` functions

### Technical Details
- TDEV and LDEV implementations efficiently reuse existing MDEV and MHDEV calculations
- All mathematical relationships verified: TDEV/MDEV ratio ≈ 0.577 at τ=1s, LDEV/MHDEV ratio ≈ 0.548 at τ=1s
- Confidence intervals properly scaled according to mathematical transformations

## [0.1.0] - 2025-08-09

### Added
- **Core package structure** with modern Julia architecture
- **Allan deviation (`adev`)**: Fundamental frequency stability measure with exact MATLAB algorithm translation
- **Modified Allan deviation (`mdev`)**: Removes dead time effects using triple-difference algorithm  
- **Modified Hadamard deviation (`mhdev`)**: Robust against frequency drift using third differences
- **Flexible API design** supporting both struct returns and multiple output patterns
- **Type-safe data structures** with `DeviationResult{T}` for comprehensive results
- **Comprehensive input validation** for phase data and sampling intervals
- **Keyword argument support** for custom averaging factors (`mlist`) and confidence levels
- **Theoretical validation** against expected noise slopes:
  - White phase noise: ADEV slope ≈ -1.0, MDEV slope ≈ -1.5 ✓
  - Random walk phase: Both ADEV and MDEV slope ≈ -0.5 ✓

### Implementation Highlights
- **Exact indexing translation** from MATLAB with careful attention to algorithm correctness
- **Efficient cumulative sum approach** for Modified Hadamard deviation computation
- **Octave-spaced default averaging factors** with appropriate minimum point requirements
- **Placeholder implementations** for noise identification and confidence interval computation
- **Multiple dispatch support** for different return patterns using `Val` types

### Testing
- Comprehensive slope validation against theoretical values for different noise types
- Algorithm correctness verification using 10,000 sample datasets
- Mathematical relationship testing between deviation types
- Edge case and error condition testing

### Dependencies
- Julia ≥ 1.6
- Statistics.jl (standard library)

### Initial Features
- Clean, consistent API across all deviation functions
- Type-safe implementations with comprehensive validation
- Performance-optimized algorithms following Julia best practices
- Extensive documentation with mathematical background and usage examples
