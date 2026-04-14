# commands/output.jl — Output: view, export, help, misc.

using Printf: @printf
using DelimitedFiles: writedlm

# ── view ──────────────────────────────────────────────────────────────────────

function cmd_view(session::Session, positional::Vector{String}, flags::Dict)
    targets = DeviationResult[]
    titles  = String[]

    if isempty(positional)
        session.last_result === nothing &&
            throw(ArgumentError("nothing to view — run `dev` first or pass NAME:DEV"))
        push!(targets, session.last_result)
        push!(titles, session.last_result.method)
    else
        for tok in positional
            ds_name, dev = parse_dataset_spec(tok)
            key = (ds_name, dev)
            haskey(session.results, key) ||
                throw(ArgumentError("no stored result for $ds_name:$dev"))
            push!(targets, session.results[key])
            push!(titles, "$ds_name:$dev")
        end
    end

    save_path = haskey(flags, "save") ? string(flags["save"]) : nothing
    do_open   = get(flags, "open", false) === true
    title     = length(titles) == 1 ? "Frequency Stability — $(titles[1])" :
                                      "Frequency Stability ($(length(titles)) curves)"
    return full_view(targets, save_path, do_open; title=title)
end

# ── export ────────────────────────────────────────────────────────────────────

function cmd_export(session::Session, positional::Vector{String}, flags::Dict)
    r, label = if isempty(positional)
        session.last_result === nothing &&
            throw(ArgumentError("nothing to export — run `dev` first or pass NAME:DEV"))
        session.last_result, session.last_result.method
    else
        ds_name, dev = parse_dataset_spec(positional[1])
        key = (ds_name, dev)
        haskey(session.results, key) ||
            throw(ArgumentError("no stored result for $ds_name:$dev"))
        session.results[key], "$(ds_name)_$(dev)"
    end

    path = String(get(flags, "out", "$(label).csv"))
    rows = hcat(r.tau, r.deviation, r.ci[:,1], r.ci[:,2],
                r.edf, r.alpha, r.neff)
    open(path, "w") do io
        println(io, "tau,sigma,ci_lo,ci_hi,edf,alpha,neff")
        writedlm(io, rows, ',')
    end
    println("Exported $(size(rows,1)) rows → $path")
    return path
end

# ── help ──────────────────────────────────────────────────────────────────────

const HELP_LINES = [
    ("load",     "load <file> [as <name>] [--col N] [--tau0 S] [--type phase|freq]"),
    ("list",     "show loaded datasets and stored results"),
    ("use",      "use <name>  — switch active dataset"),
    ("info",     "info [<name>] [--dev DEV]  — dataset or result summary"),
    ("dev",      "dev <name|all> […] [--on NAME] [--m \"1,2,4,…,4096\"]"),
    ("noise-id", "noise-id [<name>] [--dev DEV]  — per-τ α table from a stored result"),
    ("view",     "view [NAME:DEV …] [--save PATH.png] [--open]"),
    ("export",   "export [NAME:DEV] [--out PATH.csv]"),
    ("help",     "help [<command>]"),
    ("history",  "history  — list commands entered this session"),
    ("clear",    "clear  — clear the terminal"),
    ("exit",     "exit  — leave the session (alias: quit)"),
]

function cmd_help(_session::Session, positional::Vector{String}, _flags::Dict)
    if isempty(positional)
        println("SigmaTau CLI — σ(τ) stability analysis")
        println()
        for (cmd, summary) in HELP_LINES
            @printf("  %-9s  %s\n", cmd, summary)
        end
        println()
        println("Deviations available: $DEV_NAMES_JOINED (and `all`).")
        println("Dataset specs use NAME:DEV syntax (e.g. clkA:adev) in view/export.")
        return
    end
    q = lowercase(positional[1])
    for (cmd, summary) in HELP_LINES
        if cmd == q
            println("$cmd — $summary")
            return
        end
    end
    println("no such command: $q")
end

# ── misc ──────────────────────────────────────────────────────────────────────

function cmd_history(session::Session, _::Vector{String}, _flags::Dict)
    for (i, line) in enumerate(session.history)
        @printf("  %3d  %s\n", i, line)
    end
end

function cmd_clear(_session::Session, _::Vector{String}, _flags::Dict)
    print("\033[2J\033[H")
end

function cmd_exit(_session::Session, _::Vector{String}, _flags::Dict)
    throw(ExitRequested())
end

struct ExitRequested <: Exception end
