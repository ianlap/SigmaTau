"""
    SigmaTauCLI

Hybrid one-shot + interactive CLI for SigmaTau frequency-stability analysis.

Entry point: `SigmaTauCLI.main()`. Invoked by `bin/sigmatau`.

- With no `ARGS`   → `interactive_loop(Session())` — `σt>` prompt with named
                     datasets, history, inline UnicodePlots previews.
- With `ARGS`      → one-shot dispatch: parse the arguments as a single
                     command line and execute it. Errors exit non-zero.
- Piped stdin      → script mode: read lines from stdin, run each as a
                     command. Exits 0 on clean end.
"""
module SigmaTauCLI

export main

include("types.jl")
include("parser.jl")
include("loader.jl")
include("plotting.jl")
include("commands.jl")

const PROMPT = "σt> "
const BANNER = """
SigmaTau CLI v0.1.0 — σ(τ) stability analysis
Type `help` for commands, `exit` to quit.
"""

"""
    main() → Nothing

Top-level entry point. Branches on `ARGS` emptiness and stdin TTY status.
"""
function main()
    session = Session()
    if !isempty(ARGS)
        return one_shot(session, ARGS)
    elseif !isinteractive() && !isa(stdin, Base.TTY)
        return script_stdin(session)
    else
        return interactive_loop(session)
    end
end

"""
    one_shot(session, args) → Nothing

Treat the entire ARGS list as a single command line (already tokenized by the
shell). Errors exit with non-zero status for script-ability.
"""
function one_shot(session::Session, args::Vector{String})
    # Re-quote tokens that contain whitespace so parse_line sees them as one
    # token each after its own tokenizer runs.
    reconstructed = join((occursin(r"\s", a) ? "\"$a\"" : a for a in args), ' ')
    try
        dispatch(session, reconstructed)
    catch err
        err isa ExitRequested && return nothing
        println(stderr, "Error: ", sprint(showerror, err))
        exit(1)
    end
    return nothing
end

"""
    script_stdin(session) → Nothing

Read lines from stdin and run each as a command. For shell pipelines such as
`printf 'load x.csv\\ndev adev\\n' | sigmatau`.
"""
function script_stdin(session::Session)
    for line in eachline(stdin)
        push!(session.history, line)
        try
            dispatch(session, line)
        catch err
            err isa ExitRequested && return nothing
            println(stderr, "Error: ", sprint(showerror, err))
            exit(1)
        end
    end
    return nothing
end

"""
    interactive_loop(session) → Nothing

Read-eval-print loop. Ctrl-D (EOFError) exits cleanly; Ctrl-C cancels the
current command without killing the loop.
"""
function interactive_loop(session::Session)
    print(BANNER)
    while true
        print(PROMPT)
        flush(stdout)
        local line
        try
            line = readline(stdin)
        catch err
            err isa InterruptException && (println(); continue)
            rethrow(err)
        end
        isempty(line) && eof(stdin) && break
        isempty(strip(line)) && continue
        push!(session.history, line)
        try
            dispatch(session, line)
        catch err
            err isa ExitRequested && break
            rethrow(err)
        end
    end
    println("bye")
    return nothing
end

end # module
