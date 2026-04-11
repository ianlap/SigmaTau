# Plotting functionality for StabLab.jl

using Plots, Printf, DelimitedFiles

"""
    stabplot(result::DeviationResult; kwargs...)

Create a professional Allan deviation plot with confidence intervals.

# Arguments
- `result`: DeviationResult from adev(), mdev(), hdev(), etc.

# Keyword Arguments
- `title`: Plot title (default: auto-generated)
- `xlabel`: X-axis label (default: "Averaging Time τ (s)")
- `ylabel`: Y-axis label (default: auto-generated based on deviation type)
- `show_ci`: Show confidence intervals (default: true)
- `logscale`: Use log-log scale (default: true)
- `save_path`: Path to save plot (optional)
- `show_table`: Show results table below plot (default: true)
- `figsize`: Figure size tuple (default: (800, 600))

# Returns
Plot object

# Example
```julia
result = adev(phase_data, 1.0)
result_ci = compute_ci(result)
stabplot(result_ci, title="Rubidium Clock Stability")
```
"""
function stabplot(result::DeviationResult; 
                  title::String="",
                  xlabel::String="Averaging Time τ (s)",
                  ylabel::String="",
                  show_ci::Bool=true,
                  logscale::Bool=true,
                  save_path::Union{String,Nothing}=nothing,
                  show_table::Bool=true,
                  figsize::Tuple{Int,Int}=(800, 600))
    
    # Auto-generate title and ylabel if not provided
    if title == ""
        method_name = uppercase(result.method)
        title = "$method_name Analysis"
    end
    
    if ylabel == ""
        method_map = Dict(
            "adev" => "Allan Deviation σ_y(τ)",
            "mdev" => "Modified Allan Deviation σ_y(τ)",
            "hdev" => "Hadamard Deviation σ_y(τ)", 
            "mhdev" => "Modified Hadamard Deviation σ_y(τ)",
            "tdev" => "Time Deviation σ_x(τ) (s)",
            "ldev" => "Lapinski Deviation σ_x(τ) (s)",
            "totdev" => "Total Deviation σ_y(τ)",
            "mtotdev" => "Modified Total Deviation σ_y(τ)",
            "htotdev" => "Hadamard Total Deviation σ_y(τ)",
            "mhtotdev" => "Modified Hadamard Total Deviation σ_y(τ)"
        )
        ylabel = get(method_map, result.method, "Deviation")
    end
    
    # Create plot
    if logscale
        p = plot(result.tau, result.deviation,
                 xscale=:log10, yscale=:log10,
                 xlabel=xlabel, ylabel=ylabel, title=title,
                 linewidth=2, marker=:circle, markersize=4,
                 label=uppercase(result.method),
                 size=figsize,
                 grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
    else
        p = plot(result.tau, result.deviation,
                 xlabel=xlabel, ylabel=ylabel, title=title,
                 linewidth=2, marker=:circle, markersize=4,
                 label=uppercase(result.method),
                 size=figsize,
                 grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
    end
    
    # Add confidence intervals if available and requested
    if show_ci && !all(isnan.(result.ci[:, 1]))
        ci_lower = result.ci[:, 1]
        ci_upper = result.ci[:, 2]
        
        # Filter out NaN values for CI plotting
        valid_idx = .!isnan.(ci_lower) .& .!isnan.(ci_upper)
        
        if any(valid_idx)
            plot!(result.tau[valid_idx], ci_lower[valid_idx],
                  fillrange=ci_upper[valid_idx],
                  fillalpha=0.2, color=:blue, label="", 
                  linewidth=0)
            
            plot!(result.tau[valid_idx], ci_lower[valid_idx],
                  color=:blue, linestyle=:dash, linewidth=1,
                  label="$(round(result.confidence*100, digits=1))% CI")
            
            plot!(result.tau[valid_idx], ci_upper[valid_idx],
                  color=:blue, linestyle=:dash, linewidth=1, label="")
        end
    end
    
    # Save if requested
    if save_path !== nothing
        savefig(p, save_path)
        println("Plot saved to: $save_path")
    end
    
    # Print results table if requested
    if show_table
        print_results_table(result)
    end
    
    return p
end

"""
    print_results_table(result::DeviationResult)

Print a formatted table of results.
"""
function print_results_table(result::DeviationResult)
    println("\n" * "="^80)
    println("$(uppercase(result.method)) ANALYSIS RESULTS")
    println("="^80)
    println("Data points: $(result.N), Sampling interval: $(result.tau0) s")
    
    # Determine if we have confidence intervals
    has_ci = !all(isnan.(result.ci[:, 1]))
    has_edf = !all(isnan.(result.edf))
    
    # Header
    if has_ci && has_edf
        println(@sprintf("%-12s %-15s %-12s %-15s %-15s %-8s", 
                "τ (s)", "Deviation", "EDF", "CI Lower", "CI Upper", "N_eff"))
        println("-"^80)
    else
        println(@sprintf("%-12s %-15s %-8s", "τ (s)", "Deviation", "N_eff"))
        println("-"^40)
    end
    
    # Data rows
    for i in 1:length(result.tau)
        tau_str = @sprintf("%.3e", result.tau[i])
        dev_str = @sprintf("%.6e", result.deviation[i])
        neff_str = @sprintf("%d", result.neff[i])
        
        if has_ci && has_edf && !isnan(result.edf[i])
            edf_str = @sprintf("%.1f", result.edf[i])
            ci_low_str = @sprintf("%.6e", result.ci[i, 1])
            ci_high_str = @sprintf("%.6e", result.ci[i, 2])
            println(@sprintf("%-12s %-15s %-12s %-15s %-15s %-8s",
                    tau_str, dev_str, edf_str, ci_low_str, ci_high_str, neff_str))
        else
            println(@sprintf("%-12s %-15s %-8s", tau_str, dev_str, neff_str))
        end
    end
    println("-"^80)
end

"""
    load_phase_data(filename::String; tau0::Float64=1.0, scale::Float64=1e-9)

Load phase data from a text file.

# Arguments
- `filename`: Path to data file
- `tau0`: Sampling interval in seconds (default: 1.0)
- `scale`: Scale factor to convert data units (default: 1e-9 for ns to s)

# Returns
Vector of phase data in seconds
"""
function load_phase_data(filename::String; tau0::Float64=1.0, scale::Float64=1e-9)
    try
        # Try to read as two-column format (timestamp, phase)
        data = readdlm(filename, Float64)
        if size(data, 2) == 2
            # Use second column (phase data)
            phase_data = data[:, 2] .* scale
            println("Loaded $(length(phase_data)) phase samples from $filename")
            println("Data range: $(round(minimum(phase_data)*1e9, digits=1)) to $(round(maximum(phase_data)*1e9, digits=1)) ns")
            return phase_data
        elseif size(data, 2) == 1
            # Single column of phase data
            phase_data = data[:, 1] .* scale  
            println("Loaded $(length(phase_data)) phase samples from $filename")
            println("Data range: $(round(minimum(phase_data)*1e9, digits=1)) to $(round(maximum(phase_data)*1e9, digits=1)) ns")
            return phase_data
        else
            error("Unsupported data format: expected 1 or 2 columns, got $(size(data, 2))")
        end
    catch e
        error("Failed to load data from $filename: $e")
    end
end

"""
    stability_report(phase_data, tau0; methods=["adev"], save_path=nothing)

Generate a comprehensive stability analysis report with plots and tables.

# Arguments
- `phase_data`: Vector of phase data (seconds)
- `tau0`: Sampling interval (seconds)
- `methods`: List of methods to analyze (default: ["adev"])
- `save_path`: Base path for saving files (optional)

# Example
```julia
phase_data = load_phase_data("data.txt")
stability_report(phase_data, 1.0, methods=["adev", "mdev", "hdev"])
```
"""
function stability_report(phase_data, tau0; methods=["adev"], save_path=nothing)
    println("="^80)
    println("FREQUENCY STABILITY ANALYSIS REPORT") 
    println("="^80)
    println("Data length: $(length(phase_data)) samples")
    println("Sampling interval: $(tau0) s")
    println("Analysis methods: $(join(methods, ", "))")
    println("="^80)
    
    results = Dict()
    plots_array = []
    
    for method in methods
        println("\nAnalyzing $method...")
        
        # Compute deviation
        if method == "adev"
            result = adev(phase_data, tau0)
        elseif method == "mdev"
            result = mdev(phase_data, tau0)
        elseif method == "hdev"
            result = hdev(phase_data, tau0)
        elseif method == "mhdev"
            result = mhdev(phase_data, tau0)
        elseif method == "tdev"
            result = tdev(phase_data, tau0)
        elseif method == "ldev"
            result = ldev(phase_data, tau0)
        else
            println("Warning: Method $method not implemented, skipping")
            continue
        end
        
        # Add confidence intervals
        result_ci = compute_ci(result)
        results[method] = result_ci
        
        # Create plot
        plot_title = "$(uppercase(method)) Analysis"
        p = stabplot(result_ci, title=plot_title, show_table=false)
        push!(plots_array, p)
        
        # Save individual plot if requested
        if save_path !== nothing
            plot_path = "$(save_path)_$(method).png"
            savefig(p, plot_path)
        end
    end
    
    # Print all tables
    for (method, result) in results
        print_results_table(result)
    end
    
    return results, plots_array
end