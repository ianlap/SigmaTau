# commands.jl — Command handlers dispatched from the interactive loop and
# the one-shot ARGS entry point. Handler signature is uniform:
#   cmd_*(session::Session, positional::Vector{String}, flags::Dict) → Any

using DelimitedFiles: writedlm
using Printf: @printf, @sprintf

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

# ── load ──────────────────────────────────────────────────────────────────────

function cmd_load(session::Session, positional::Vector{String}, flags::Dict)
    ds = load_dataset(positional, flags)
    session.datasets[ds.name] = ds
    session.current = ds.name
    return ds
end

# ── list ──────────────────────────────────────────────────────────────────────

function cmd_list(session::Session, _::Vector{String}, _flags::Dict)
    if isempty(session.datasets)
        println("(no datasets loaded)")
    else
        println("Datasets:")
        for (name, ds) in session.datasets
            marker = name == session.current ? "*" : " "
            @printf("  %s %-10s  N=%-8d  tau0=%-8g  type=%-5s  %s\n",
                    marker, name, ds.nrows, ds.tau0,
                    ds.data_type, basename(ds.source_file))
        end
    end

    if !isempty(session.results)
        println("Results:")
        for (key, r) in session.results
            dsname, dev = key
            @printf("  %s:%-8s  τ∈[%g, %g], L=%d\n",
                    dsname, dev,
                    first(r.tau), last(r.tau), length(r.tau))
        end
    end
end

# ── use ───────────────────────────────────────────────────────────────────────

function cmd_use(session::Session, positional::Vector{String}, _flags::Dict)
    isempty(positional) && throw(ArgumentError("usage: use <name>"))
    name = Symbol(positional[1])
    haskey(session.datasets, name) ||
        throw(ArgumentError("unknown dataset :$name"))
    session.current = name
    println("Active dataset: :$name")
end

# ── info ──────────────────────────────────────────────────────────────────────

function cmd_info(session::Session, positional::Vector{String}, flags::Dict)
    if isempty(positional) && !haskey(flags, "dev")
        name = resolve_dataset(session, flags)
        ds = session.datasets[name]
        println("Dataset :$name")
        @printf("  source    : %s\n", ds.source_file)
        @printf("  column    : %d\n", ds.column)
        @printf("  samples   : %d\n", ds.nrows)
        @printf("  tau0      : %g s\n", ds.tau0)
        @printf("  data_type : %s\n", ds.data_type)
        @printf("  data[1:3] : %g, %g, %g\n",
                ds.data[1], ds.data[min(2,end)], ds.data[min(3,end)])
    else
        name = isempty(positional) ? resolve_dataset(session, flags) :
                                      Symbol(positional[1])
        dev = String(get(flags, "dev", ""))
        isempty(dev) &&
            throw(ArgumentError("usage: info <name> --dev <devname>"))
        key = (name, dev)
        haskey(session.results, key) ||
            throw(ArgumentError("no result for $name:$dev"))
        r = session.results[key]
        println("Result $name:$dev")
        @printf("  method     : %s\n", r.method)
        @printf("  N          : %d\n", r.N)
        @printf("  tau0       : %g\n", r.tau0)
        @printf("  points     : %d (τ = %g … %g)\n",
                length(r.tau), first(r.tau), last(r.tau))
        @printf("  confidence : %.3f\n", r.confidence)
        print_result_table(r)
    end
end

function print_result_table(r::DeviationResult)
    println("    m    τ(s)         σ(τ)         ci_lo         ci_hi        α    neff")
    for i in eachindex(r.tau)
        m = round(Int, r.tau[i] / r.tau0)
        @printf("   %4d  %-11g  %-11g  %-11g  %-11g  %3d  %6d\n",
                m, r.tau[i], r.deviation[i],
                r.ci[i,1], r.ci[i,2], r.alpha[i], r.neff[i])
    end
end

# ── dev ───────────────────────────────────────────────────────────────────────

function cmd_dev(session::Session, positional::Vector{String}, flags::Dict)
    isempty(positional) &&
        throw(ArgumentError("usage: dev <devname|all> [devname2 …] [--on NAME] [--m \"1,2,4,…\"]"))
    name = resolve_dataset(session, flags)
    ds = session.datasets[name]

    devs = String[]
    for tok in positional
        if tok == "all"
            append!(devs, DEV_NAMES)
        elseif haskey(DEV_DICT, tok)
            push!(devs, tok)
        else
            throw(ArgumentError("unknown deviation '$tok' (known: $DEV_NAMES_JOINED)"))
        end
    end
    unique!(devs)

    m_list = haskey(flags, "m") ? parse_mlist(string(flags["m"])) : nothing

    last_r = nothing
    for dev in devs
        fn = DEV_DICT[dev]
        dev == "mhtotdev" && println("  computing mhtotdev (may take a while)…")
        r = fn(ds.data, ds.tau0; m_list=m_list, data_type=ds.data_type)
        session.results[(name, dev)] = r
        last_r = r
        if isempty(r.tau)
            @printf("  %-8s → 0 points computed\n", dev)
        else
            @printf("  %-8s → %d points, σ(τ=%g)=%g\n",
                    dev, length(r.tau), first(r.tau), first(r.deviation))
        end
    end

    session.last_result = last_r
    last_r === nothing || inline_preview(last_r)
    return last_r
end

# ── noise-id ──────────────────────────────────────────────────────────────────

function cmd_noise_id(session::Session, positional::Vector{String}, flags::Dict)
    name = isempty(positional) ? resolve_dataset(session, flags) :
                                  Symbol(positional[1])
    dev  = String(get(flags, "dev", "adev"))
    key = (name, dev)
    haskey(session.results, key) ||
        throw(ArgumentError("no stored $dev result for :$name — run `dev $dev` first"))
    r = session.results[key]

    println("Noise identification ($name:$dev)  (α: 2=WPM 1=FPM 0=WFM -1=FFM -2=RWFM)")
    println("    m    τ(s)         σ(τ)         α")
    for i in eachindex(r.tau)
        m = round(Int, r.tau[i] / r.tau0)
        @printf("   %4d  %-11g  %-11g  %3d\n",
                m, r.tau[i], r.deviation[i], r.alpha[i])
    end
    return r.alpha
end

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
