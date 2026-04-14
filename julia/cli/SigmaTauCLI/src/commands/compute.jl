# commands/compute.jl — Computation: dev, noise-id.

using Printf: @printf

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
