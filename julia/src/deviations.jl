# deviations.jl — Deviation wrappers (Thin wrappers over the shared engine)
# Architecture: deviation-engine skill / CLAUDE.md §Architecture

# Shared constants and internal helpers
include("deviations/common.jl")

# Deviation families
include("deviations/allan.jl")     # ADEV, MDEV, TDEV
include("deviations/hadamard.jl")  # HDEV, MHDEV, LDEV
include("deviations/total.jl")     # TOTDEV, MTOTDEV, HTOTDEV, MHTOTDEV
