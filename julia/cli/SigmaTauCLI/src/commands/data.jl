# commands/data.jl — Dataset management: load, list, use, info.

using Printf: @printf

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
            if isempty(r.tau)
                @printf("  %s:%-8s  0 points computed\n", dsname, dev)
            else
                @printf("  %s:%-8s  τ∈[%g, %g], L=%d\n",
                        dsname, dev,
                        first(r.tau), last(r.tau), length(r.tau))
            end
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
        if isempty(r.tau)
            println("  points     : 0 computed")
        else
            @printf("  points     : %d (τ = %g … %g)\n",
                    length(r.tau), first(r.tau), last(r.tau))
            @printf("  confidence : %.3f\n", r.confidence)
            print_result_table(r)
        end
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
