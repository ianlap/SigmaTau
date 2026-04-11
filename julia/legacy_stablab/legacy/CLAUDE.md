# CLAUDE.md

Development guidance for AI assistants working on StabLab.jl.

## Running tests & validation

```bash
julia --project=. tests/runtests.jl              # @testset-based suite
julia --project=. validation/validate.jl          # slopes, relationships, AllanTools comparison
julia --project=. validation/validate.jl --slopes # subset
```

## External reference tools

- **MATLAB AllanLab**: `validation/generate_matlab_reference.m` (edit `allanlab_path` inside)
- **Python AllanTools**: `python validation/generate_reference.py`

## Source layout

| File | Contents |
|------|----------|
| `src/deviations.jl` | All 10 NIST deviation functions + unified `Val(N)` tuple API |
| `src/time_error.jl` | TIE, MTIE, PDEV, THEO1 |
| `src/confidence.jl` | EDF + chi-squared confidence intervals |
| `src/noise.jl` | Noise identification + Timmer-Koenig synthesis |
| `src/core.jl` | Validation helpers, `unpack_result` |
| `src/plotting.jl` | `stabplot`, `stability_report`, `load_phase_data` |

## Common Julia gotchas

- **Global scoping**: use `global` when mutating outer variables inside `for` loops in scripts.
- **Statistics cache warning**: harmless; package still loads.
- **String interpolation**: extract complex expressions to variables before interpolating.

## Style

- Avoid emojis in code output.
- Keep docstrings short (one-paragraph + "see also").
- Algorithms translated from MATLAB AllanLab; preserve indexing comments where non-obvious.
