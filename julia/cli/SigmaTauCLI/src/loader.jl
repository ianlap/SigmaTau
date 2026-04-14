# loader.jl — Load delimited text / CSV phase or frequency files.

using DelimitedFiles: readdlm
using Printf: @printf, @sprintf
using SigmaTau: validate_phase_data, validate_tau0

"""
    load_dataset(positional, flags; stdin_=stdin, stdout_=stdout) → Dataset

Entry point for the `load` command.

Usage:
    load <file> [as <name>] [--col <n>] [--tau0 <s>] [--type phase|freq]

When `--col`, `--tau0`, and `--type` are all present, runs silently.
Otherwise shows a preview of the file and prompts for the missing pieces.

`stdin_` / `stdout_` are pluggable so tests can run non-interactively.
"""
function load_dataset(positional::Vector{String}, flags::Dict;
                      stdin_=stdin, stdout_=stdout)
    path, name = parse_load_positional(positional)

    isfile(path) || throw(ArgumentError("file not found: $path"))

    matrix = read_matrix(path)
    nrows, ncols = size(matrix)

    col       = get_col(flags, matrix, stdin_, stdout_)
    data_type = get_type(flags, stdin_, stdout_)
    tau0      = get_tau0(flags, stdin_, stdout_)

    col in 1:ncols ||
        throw(ArgumentError("column $col out of range (file has $ncols columns)"))

    raw = matrix[:, col]
    data = validate_phase_data(raw)
    tau0_v = validate_tau0(tau0)

    ds = Dataset(name, data, tau0_v, data_type, path, col, length(data))
    println(stdout_, "Loaded ':$(name)': $(length(data)) $(data_type) samples, " *
                     "tau0=$(tau0_v)s, col=$(col), file=$(basename(path))")
    return ds
end

# ── Helpers ───────────────────────────────────────────────────────────────────

"""
    parse_load_positional(positional) → (path, name)

Accept `["file.csv"]` or `["file.csv", "as", "myname"]`. Auto-name from
basename-without-extension when `as` is absent.
"""
function parse_load_positional(positional::Vector{String})
    isempty(positional) &&
        throw(ArgumentError("usage: load <file> [as <name>] [--flags]"))
    path = positional[1]
    name = if length(positional) >= 3 && lowercase(positional[2]) == "as"
        Symbol(positional[3])
    elseif length(positional) == 1
        Symbol(splitext(basename(path))[1])
    else
        throw(ArgumentError("unexpected tokens after filename: $(positional[2:end])"))
    end
    return (path, name)
end

"""
    read_matrix(path) → Matrix{Float64}

Read a delimited text file. Auto-detects comma vs whitespace from the first
non-comment line.
"""
function read_matrix(path::AbstractString)
    first_data = ""
    open(path) do io
        for ln in eachline(io)
            s = strip(ln)
            isempty(s) && continue
            startswith(s, '#') && continue
            first_data = s
            break
        end
    end
    delim = occursin(',', first_data) ? ',' : nothing
    raw = delim === nothing ? readdlm(path; comments=true) :
                              readdlm(path, delim; comments=true)
    # readdlm may return Any if mixed — coerce numeric.
    return Float64.(raw)
end

"""
    preview_matrix(matrix, stdout_; n=5)

Print the first `n` rows so the user can pick a column interactively.
"""
function preview_matrix(matrix::AbstractMatrix, stdout_; n::Int=5)
    nrows, ncols = size(matrix)
    shown = min(n, nrows)
    println(stdout_, "File preview ($nrows rows × $ncols columns):")
    header = "  row   " * join(("col$(j)" for j in 1:ncols), "  ")
    println(stdout_, header)
    for i in 1:shown
        row = join((@sprintf("%.6g", matrix[i, j]) for j in 1:ncols), "  ")
        @printf(stdout_, "  %3d   %s\n", i, row)
    end
    nrows > shown && println(stdout_, "  … ($(nrows - shown) more rows)")
end

function get_col(flags, matrix, stdin_, stdout_)
    haskey(flags, "col") && return parse(Int, string(flags["col"]))
    ncols = size(matrix, 2)
    ncols == 1 && return 1
    preview_matrix(matrix, stdout_)
    print(stdout_, "Column index [1]: "); flush(stdout_)
    line = strip(readline(stdin_))
    return isempty(line) ? 1 : parse(Int, line)
end

function get_type(flags, stdin_, stdout_)
    if haskey(flags, "type")
        t = lowercase(string(flags["type"]))
        t in ("phase", "freq") ||
            throw(ArgumentError("--type must be phase or freq, got $t"))
        return Symbol(t)
    end
    print(stdout_, "Data type (phase/freq) [phase]: "); flush(stdout_)
    line = lowercase(strip(readline(stdin_)))
    isempty(line) && return :phase
    line in ("phase", "freq") ||
        throw(ArgumentError("expected phase or freq, got $line"))
    return Symbol(line)
end

function get_tau0(flags, stdin_, stdout_)
    haskey(flags, "tau0") && return parse(Float64, string(flags["tau0"]))
    print(stdout_, "tau0 (sampling interval in seconds): "); flush(stdout_)
    line = strip(readline(stdin_))
    isempty(line) && throw(ArgumentError("tau0 is required"))
    return parse(Float64, line)
end
