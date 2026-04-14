# types.jl — Session and dataset types for the SigmaTau CLI.

using SigmaTau: DeviationResult

"""
    Dataset

A loaded clock dataset with all metadata required to run SigmaTau deviations.
"""
struct Dataset
    name        :: Symbol
    data        :: Vector{Float64}
    tau0        :: Float64
    data_type   :: Symbol          # :phase or :freq
    source_file :: String
    column      :: Int
    nrows       :: Int
end

"""
    ResultKey = Tuple{Symbol, String}

Keys in `Session.results` — `(:clkA, "adev")` means the ADEV result for dataset `clkA`.
"""
const ResultKey = Tuple{Symbol, String}

"""
    Session

Mutable state carried through an interactive CLI session. Fresh `Session()` is
also used for one-shot invocations — nothing in the struct is session-specific.
"""
mutable struct Session
    datasets    :: Dict{Symbol, Dataset}
    results     :: Dict{ResultKey, DeviationResult}
    current     :: Union{Symbol, Nothing}
    history     :: Vector{String}
    last_result :: Union{DeviationResult, Nothing}
end

Session() = Session(Dict{Symbol, Dataset}(),
                    Dict{ResultKey, DeviationResult}(),
                    nothing, String[], nothing)

"""
    resolve_dataset(session, flags) → Symbol

Return the dataset name the caller wants to operate on: `--on NAME` if given,
else `session.current`. Throws with a helpful message when neither is set.
"""
function resolve_dataset(session::Session, flags::Dict)
    if haskey(flags, "on")
        name = Symbol(flags["on"])
        haskey(session.datasets, name) ||
            throw(ArgumentError("unknown dataset :$name (loaded: $(keys(session.datasets)))"))
        return name
    end
    session.current === nothing &&
        throw(ArgumentError("no active dataset — use `load` or `use NAME`"))
    return session.current
end
