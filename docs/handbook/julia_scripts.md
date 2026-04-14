# Julia Scripts Reference

## `compute_all_devs.jl`

**Purpose:** Compute all 10 NIST deviations from one dataset.

**Syntax:**
```bash
julia --project=julia scripts/julia/compute_all_devs.jl <file> <tau0>
```

| Argument | Required | Description |
| :--- | :---: | :--- |
| `<file>` | Yes | Input phase/frequency text file. |
| `<tau0>` | Yes | Sample interval in seconds. |

---

## `mhdev_preview.jl`

**Purpose:** Quick preview of noise characteristics used during KF preparation.

**Syntax:**
```bash
julia --project=julia scripts/julia/mhdev_preview.jl <dataset>
```

| Argument | Required | Description |
| :--- | :---: | :--- |
| `<dataset>` | Yes | Dataset label or source expected by the script pipeline. |

---

## `kf_pipeline.jl`

**Purpose:** End-to-end Kalman pipeline (noise estimate, optimization, filtering).

**Syntax:**
```bash
julia --project=julia scripts/julia/kf_pipeline.jl <dataset>
```

| Argument | Required | Description |
| :--- | :---: | :--- |
| `<dataset>` | Yes | Dataset name consumed by pipeline artifacts and report scripts. |

---

## `basic_usage.jl`

**Purpose:** Minimal API demonstration for SigmaTau in Julia.

**Syntax:**
```bash
julia --project=julia scripts/julia/basic_usage.jl
```

## Outputs

Scripts usually create dataset-scoped result folders (for tables/figures) under `results/<dataset>/` or adjacent script-defined directories.

## Troubleshooting

- Ensure dependencies are installed: `julia --project=julia -e 'using Pkg; Pkg.instantiate()'`.
- If a script cannot resolve a dataset label, verify file/folder naming expected by the script.
