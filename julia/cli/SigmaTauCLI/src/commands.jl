# commands.jl — Command handlers dispatched from the interactive loop and
# the one-shot ARGS entry point. Handler signature is uniform:
#   cmd_*(session::Session, positional::Vector{String}, flags::Dict) → Any

include("commands/common.jl")
include("commands/data.jl")
include("commands/compute.jl")
include("commands/output.jl")

# ── dispatch table ────────────────────────────────────────────────────────────

const COMMANDS = Dict{String, Function}(
    "load"     => cmd_load,
    "list"     => cmd_list,
    "use"      => cmd_use,
    "info"     => cmd_info,
    "dev"      => cmd_dev,
    "noise-id" => cmd_noise_id,
    "view"     => cmd_view,
    "export"   => cmd_export,
    "help"     => cmd_help,
    "?"        => cmd_help,
    "history"  => cmd_history,
    "clear"    => cmd_clear,
    "exit"     => cmd_exit,
    "quit"     => cmd_exit,
)

"""
    dispatch(session, line) → Any

Parse and execute one command line. Errors are caught and printed — the
caller (interactive loop) stays alive. `ExitRequested` escapes to the loop.
"""
function dispatch(session::Session, line::AbstractString)
    verb, pos, flags = parse_line(line)
    isempty(verb) && return nothing
    haskey(COMMANDS, verb) ||
        (println(stderr, "Error: unknown command '$verb' (try `help`)"); return nothing)
    try
        return COMMANDS[verb](session, pos, flags)
    catch err
        err isa ExitRequested   && rethrow(err)
        err isa InterruptException && (println("\n[cancelled]"); return nothing)
        err isa ArgumentError   && (println(stderr, "Error: ", err.msg); return nothing)
        # Unknown error — print stack but keep loop alive.
        println(stderr, "Error: ", sprint(showerror, err))
        Base.show_backtrace(stderr, catch_backtrace())
        println(stderr)
        return nothing
    end
end
