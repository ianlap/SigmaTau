# common.jl — Shared constants for CLI command modules.

import SigmaTau: adev, mdev, hdev, mhdev, tdev, ldev,
                 totdev, mtotdev, htotdev, mhtotdev

"""
Ordered registry of all deviation functions the CLI exposes. Ordering
matters for `dev all` output.
"""
const DEV_FUNCTIONS = (
    ("adev",     adev),
    ("mdev",     mdev),
    ("hdev",     hdev),
    ("mhdev",    mhdev),
    ("tdev",     tdev),
    ("ldev",     ldev),
    ("totdev",   totdev),
    ("mtotdev",  mtotdev),
    ("htotdev",  htotdev),
    ("mhtotdev", mhtotdev),
)

const DEV_DICT = Dict{String, Function}(name => fn for (name, fn) in DEV_FUNCTIONS)
const DEV_NAMES = [name for (name, _) in DEV_FUNCTIONS]
const DEV_NAMES_JOINED = join(DEV_NAMES, ", ")
