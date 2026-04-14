using Test
using Random
using DelimitedFiles: writedlm
using SigmaTauCLI
using SigmaTau: DeviationResult

# Bring internals into scope for tests.
using SigmaTauCLI: Session, Dataset, parse_line, parse_mlist, parse_dataset_spec,
                   dispatch, COMMANDS, load_dataset, inline_preview, full_view

# ── Parser ────────────────────────────────────────────────────────────────────

@testset "parser: parse_line" begin
    v, p, f = parse_line("")
    @test v == "" && isempty(p) && isempty(f)

    v, p, f = parse_line("   \t  ")
    @test v == "" && isempty(p) && isempty(f)

    v, p, f = parse_line("dev adev")
    @test v == "dev" && p == ["adev"] && isempty(f)

    v, p, f = parse_line("dev adev mdev --on clkA --m 1,2,4")
    @test v == "dev"
    @test p == ["adev", "mdev"]
    @test f["on"] == "clkA"
    @test f["m"] == "1,2,4"

    # --key=value form
    v, p, f = parse_line("load file.csv --tau0=0.5 --type=phase")
    @test f["tau0"] == "0.5"
    @test f["type"] == "phase"

    # Boolean flags
    v, p, f = parse_line("view --open")
    @test f["open"] === true

    # Quoted strings
    v, p, f = parse_line(raw"""load "my file.csv" as clkA""")
    @test p == ["my file.csv", "as", "clkA"]

    @test_throws ArgumentError parse_line("load \"unterminated")
end

@testset "parser: parse_mlist" begin
    @test parse_mlist("1,2,4,8") == [1, 2, 4, 8]
    @test parse_mlist("4,2,1") == [1, 2, 4]        # sorted
    @test parse_mlist("1,2,2,4") == [1, 2, 4]      # deduped

    # Geometric expansion (octave).
    expanded = parse_mlist("1,2,4,...,64")
    @test expanded == [1, 2, 4, 8, 16, 32, 64]

    # Non-power-of-2 ratio rounds and dedupes.
    tripled = parse_mlist("1,3,...,27")
    @test tripled == [1, 3, 9, 27]

    @test_throws ArgumentError parse_mlist("...,8")
    @test_throws ArgumentError parse_mlist("1,2,...")
    @test_throws ArgumentError parse_mlist("1,1,...,8")  # ratio == 1
end

@testset "parser: parse_dataset_spec" begin
    @test parse_dataset_spec("clkA:adev") == (:clkA, "adev")
    @test_throws ArgumentError parse_dataset_spec("clkA-adev")
end

# ── Loader ────────────────────────────────────────────────────────────────────

@testset "loader: silent (all flags)" begin
    # Synthetic 2-column CSV: col1 noise, col2 cumulative (phase-like).
    rng = Xoshiro(42)
    y = randn(rng, 200)
    x = cumsum(y)
    path = tempname() * ".csv"
    open(path, "w") do io
        for i in 1:length(x)
            println(io, "$(y[i]),$(x[i])")
        end
    end

    ds = load_dataset([path, "as", "clkA"],
                      Dict("col" => "2", "tau0" => "1.0", "type" => "phase"))
    @test ds.name == :clkA
    @test ds.nrows == 200
    @test ds.tau0 == 1.0
    @test ds.data_type == :phase
    @test ds.column == 2
    @test length(ds.data) == 200
end

@testset "loader: auto-name from basename, whitespace-delimited" begin
    path = tempname() * ".txt"
    rng = Xoshiro(0)
    x = cumsum(randn(rng, 128))
    open(path, "w") do io
        for v in x
            println(io, v)
        end
    end
    base = Symbol(splitext(basename(path))[1])
    ds = load_dataset([path], Dict("col" => "1", "tau0" => "0.1", "type" => "phase"))
    @test ds.name == base
    @test ds.nrows == 128
    @test ds.tau0 == 0.1
end

# ── Commands (end-to-end via dispatch) ────────────────────────────────────────

# Build a session pre-loaded with synthetic white-FM phase data.
function make_session_with_data(n::Int=128, tau0::Float64=1.0)
    session = Session()
    rng = Xoshiro(123)
    y = randn(rng, n)
    x = cumsum(y)
    path = tempname() * ".txt"
    open(path, "w") do io
        for v in x
            println(io, v)
        end
    end
    dispatch(session, "load $path as clk --col 1 --tau0 $tau0 --type phase")
    return session
end

@testset "dispatch: unknown verb doesn't throw" begin
    session = Session()
    @test dispatch(session, "wat") === nothing
end

@testset "dispatch: empty / whitespace line" begin
    session = Session()
    @test dispatch(session, "") === nothing
    @test dispatch(session, "   ") === nothing
end

@testset "command: list + use + info" begin
    session = make_session_with_data()
    @test session.current == :clk
    dispatch(session, "list")
    dispatch(session, "info")           # dataset summary
    dispatch(session, "use nope")       # non-fatal; prints to stderr
    @test session.current == :clk       # unchanged on failure
    dispatch(session, "use clk")
    @test session.current == :clk
end

@testset "command: dev single and all" begin
    session = make_session_with_data()
    r = dispatch(session, "dev adev")
    @test r isa DeviationResult
    @test r.method == "adev"
    @test haskey(session.results, (:clk, "adev"))

    # "all" stores 10 entries. mhtotdev can be slow on 128 samples but still runs.
    dispatch(session, "dev all")
    stored = [k[2] for k in keys(session.results) if k[1] == :clk]
    for d in ("adev", "mdev", "hdev", "mhdev", "tdev", "ldev",
              "totdev", "mtotdev", "htotdev", "mhtotdev")
        @test d in stored
    end
end

@testset "command: dev with explicit m-list" begin
    session = make_session_with_data()
    r = dispatch(session, "dev adev --m 1,2,4,8")
    @test r isa DeviationResult
    @test round.(Int, r.tau ./ r.tau0) == [1, 2, 4, 8]
end

@testset "command: noise-id on stored result" begin
    session = make_session_with_data(256)
    dispatch(session, "dev adev")
    a = dispatch(session, "noise-id")
    @test a isa Vector{Int}
    @test all(-3 .<= a .<= 3)
end

@testset "command: export writes a readable CSV" begin
    session = make_session_with_data()
    dispatch(session, "dev adev")
    out = tempname() * ".csv"
    dispatch(session, "export clk:adev --out $out")
    @test isfile(out)
    content = read(out, String)
    @test occursin("tau,sigma,ci_lo,ci_hi,edf,alpha,neff", content)
end

# ── Plotting smoke test ───────────────────────────────────────────────────────

@testset "plotting: full_view writes a nonempty PNG" begin
    session = make_session_with_data()
    dispatch(session, "dev adev")
    r = session.results[(:clk, "adev")]
    path = tempname() * ".png"
    full_view([r], path, false)
    @test isfile(path)
    @test filesize(path) > 0
end

@testset "plotting: inline_preview runs without throwing" begin
    session = make_session_with_data()
    dispatch(session, "dev adev")
    r = session.results[(:clk, "adev")]
    buf = IOBuffer()
    inline_preview(r; io=buf)
    @test length(take!(buf)) > 0
end
