# Workflow Recipes

## 1) Quick all-deviation batch (Julia)

```bash
julia --project=julia scripts/julia/compute_all_devs.jl reference/validation/stable32gen.DAT 1.0
python3 scripts/python/plot_devs.py stable32gen
```

**Expected artifacts:** deviation CSV(s) and plot images for the dataset.

---

## 2) Full Kalman characterization pipeline

```bash
# Preview noise behavior
julia --project=julia scripts/julia/mhdev_preview.jl 6krb25apr
python3 scripts/python/plot_mhdev_preview.py 6krb25apr

# Interactive power-law fitting
python3 scripts/python/mhdev_fit_interactive.py 6krb25apr

# Run KF optimization + filtering
julia --project=julia scripts/julia/kf_pipeline.jl 6krb25apr

# Plot diagnostics
python3 scripts/python/plot_kf.py 6krb25apr
```

**Expected artifacts:** noise-fit configuration, KF outputs, and diagnostic plots.

---

## 3) Cross-validation report generation

```bash
python3 scripts/python/generate_comprehensive_report.py
```

**Expected artifacts:** comparison tables/markdown for Stable32/allantools/SigmaTau agreement.

---

## 4) Interactive CLI workflow

```text
bin/sigmatau
load reference/validation/stable32gen.DAT as s32 --tau0 1.0
dev all --on s32 --m "1,2,4,...,1024"
view s32:adev --save results/s32_adev.png
export s32:adev --out results/s32_adev.csv
exit
```

**Expected artifacts:** saved plot and CSV export from session-cached results.
