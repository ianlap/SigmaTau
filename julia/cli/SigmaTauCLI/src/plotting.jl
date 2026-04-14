# plotting.jl — Inline terminal preview (UnicodePlots) and full-res view (Plots).

using UnicodePlots: lineplot, lineplot!
using Plots
using SigmaTau: DeviationResult

"""
    inline_preview(result; io=stdout)

Render a compact log-log σ(τ) preview to the terminal right after a `dev`
computation. Uses UnicodePlots on log10-transformed data (UnicodePlots has no
native log-scale).
"""
function inline_preview(result::DeviationResult; io::IO=stdout)
    tau_l = log10.(result.tau)
    dev_l = log10.(result.deviation)
    finite = isfinite.(dev_l)
    any(finite) || (println(io, "(no finite deviation values to plot)"); return)
    p = lineplot(tau_l[finite], dev_l[finite];
                 title  = uppercase(result.method),
                 xlabel = "log10 τ/s",
                 ylabel = "log10 σ(τ)",
                 width  = 60,
                 height = 12)
    show(io, p)
    println(io)
end

"""
    full_view(results, save_path, do_open; title="Frequency Stability")

Render a Plots.jl σ(τ) figure overlaying all `results`. Saves to `save_path`
(or a tempfile PNG if `save_path === nothing`) and optionally hands the file
to the OS viewer.
"""
function full_view(results::AbstractVector{<:DeviationResult},
                   save_path::Union{Nothing, AbstractString},
                   do_open::Bool;
                   title::AbstractString = "Frequency Stability")
    isempty(results) && throw(ArgumentError("nothing to view"))

    p = plot(; xscale=:log10, yscale=:log10,
               xlabel="τ (s)", ylabel="σ(τ)",
               title=title, legend=:topright,
               framestyle=:box, minorgrid=true)

    for r in results
        finite = isfinite.(r.deviation) .& (r.deviation .> 0)
        any(finite) || continue
        τ   = r.tau[finite]
        σ   = r.deviation[finite]
        ci  = r.ci[finite, :]
        # Guard against negative/zero CI bounds on a log axis.
        lo  = max.(σ .- ci[:, 1], 1e-300)
        hi  = max.(ci[:, 2] .- σ, 1e-300)
        plot!(p, τ, σ; label=uppercase(r.method),
                      yerror=(lo, hi),
                      marker=:circle, markersize=3, linewidth=1.5)
    end

    path = save_path === nothing ? tempname() * ".png" : String(save_path)
    savefig(p, path)
    println("Saved: $path")

    if do_open
        cmd = if Sys.islinux()
            `xdg-open $path`
        elseif Sys.isapple()
            `open $path`
        elseif Sys.iswindows()
            `cmd /c start "" $path`
        else
            @warn "don't know how to open files on this OS; leaving at $path"
            return path
        end
        try
            run(pipeline(cmd; stdout=devnull, stderr=devnull); wait=false)
        catch err
            @warn "failed to launch viewer" exception=err
        end
    end
    return path
end
