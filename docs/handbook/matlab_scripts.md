# MATLAB Scripts Reference

MATLAB scripts mirror major Julia workflows.

## Setup

```matlab
cd matlab
addpath(genpath(pwd));
```

## `compute_all_devs.m`

**Purpose:** Compute all 10 deviation families for one dataset.

**Syntax:**
```matlab
compute_all_devs('reference/validation/stable32gen.DAT', 1.0)
```

## `kf_pipeline.m`

**Purpose:** Run full Kalman pipeline and save analysis outputs.

**Syntax:**
```matlab
results = kf_pipeline('reference/validation/stable32gen.DAT', 1.0)
```

## `basic_usage.m`

**Purpose:** Minimal demonstration of package API calls.

**Syntax:**
```matlab
basic_usage
```

## Outputs

- Deviation tables and derived metrics.
- Pipeline outputs used by downstream plotting/analysis scripts.

## Troubleshooting

- If functions are unresolved, re-run `addpath(genpath(pwd))` from `matlab/`.
- Confirm file path and `tau0` consistency with your dataset sampling interval.
