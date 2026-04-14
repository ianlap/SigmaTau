# parser.jl — Shell-style command line parser.
#
# parse_line("dev adev --on clkA --m 1,2,4,...,64")
#   → ("dev", ["adev"], Dict("on"=>"clkA", "m"=>"1,2,4,...,64"))

# Flags that take no value. Everything else consumes the next token.
const BOOLEAN_FLAGS = Set([
    "open",      # view --open
    "view",      # one-shot pipeline: load + dev + view
    "help",
    "verbose",
])

"""
    tokenize(line) → Vector{String}

Whitespace-split with double-quoted string support. Quotes are stripped;
everything inside a matching pair stays in one token.
"""
function tokenize(line::AbstractString)
    tokens = String[]
    buf = IOBuffer()
    in_quote = false
    for c in line
        if c == '"'
            in_quote = !in_quote
        elseif !in_quote && isspace(c)
            s = String(take!(buf))
            isempty(s) || push!(tokens, s)
        else
            print(buf, c)
        end
    end
    in_quote && throw(ArgumentError("unterminated quoted string: $line"))
    s = String(take!(buf))
    isempty(s) || push!(tokens, s)
    return tokens
end

"""
    parse_line(line) → (verb, positional, flags)

Parse a single CLI command line. Empty/whitespace input returns
`("", String[], Dict())` — callers should treat this as a no-op.
"""
function parse_line(line::AbstractString)
    tokens = tokenize(line)
    isempty(tokens) && return ("", String[], Dict{String,Any}())

    verb = tokens[1]
    positional = String[]
    flags = Dict{String,Any}()

    i = 2
    while i <= length(tokens)
        tok = tokens[i]
        if startswith(tok, "--")
            key, sep, val = partition(tok[3:end], '=')
            if sep == "="            # --key=value
                flags[key] = val
            elseif key in BOOLEAN_FLAGS
                flags[key] = true
            elseif i < length(tokens) # --key value
                flags[key] = tokens[i+1]
                i += 1
            else                      # trailing --flag with no value
                flags[key] = true
            end
        else
            push!(positional, tok)
        end
        i += 1
    end
    return (verb, positional, flags)
end

# String.partition-like helper; Julia's split doesn't give the separator.
function partition(s::AbstractString, sep::Char)
    idx = findfirst(==(sep), s)
    idx === nothing ? (s, "", "") : (s[1:prevind(s, idx)], string(sep), s[nextind(s, idx):end])
end

"""
    parse_mlist(s) → Vector{Int}

Parse an `--m` flag value.

- `"1,2,4,8,16"` → `[1,2,4,8,16]`
- `"1,2,4,...,4096"` → geometric expansion, ratio inferred from the two
  values before `...`, expanded up to (and including) the endpoint, deduped,
  sorted.
"""
function parse_mlist(s::AbstractString)
    parts = strip.(split(s, ','))
    ellipsis_idx = findfirst(==("..."), parts)
    if ellipsis_idx === nothing
        return sort!(unique!(parse.(Int, parts)))
    end

    ellipsis_idx >= 3 ||
        throw(ArgumentError("m-list `...` needs at least two preceding values: $s"))
    ellipsis_idx < length(parts) ||
        throw(ArgumentError("m-list `...` needs a trailing endpoint: $s"))

    explicit = parse.(Int, parts[1:ellipsis_idx-1])
    endpoint = parse(Int, parts[ellipsis_idx+1])
    a, b     = explicit[end-1], explicit[end]
    ratio    = b / a
    ratio > 1 || throw(ArgumentError("m-list ratio must be > 1: got $a → $b"))

    vals = Float64.(explicit)
    v = Float64(b)
    while true
        v *= ratio
        v > endpoint + 0.5 && break
        push!(vals, v)
    end
    return sort!(unique!(round.(Int, vals)))
end

"""
    parse_dataset_spec(s) → (name, dev)

Parse a `view`/`export` target token of the form `clkA:adev`. Returns
`(:clkA, "adev")`. Missing colon throws.
"""
function parse_dataset_spec(s::AbstractString)
    idx = findfirst(==(':'), s)
    idx === nothing &&
        throw(ArgumentError("expected NAME:DEV (e.g. clkA:adev), got $s"))
    return (Symbol(s[1:prevind(s, idx)]), String(s[nextind(s, idx):end]))
end
