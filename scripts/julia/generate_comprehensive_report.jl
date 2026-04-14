# scripts/julia/generate_comprehensive_report.jl
# Structured report generator called from Python.

using SigmaTau
using DelimitedFiles

function main()
    # Simple positional argument parsing:
    # 1: input path, 2: tau0, 3: m_list (comma-separated)
    if length(ARGS) < 3
        println(stderr, "Usage: julia generate_comprehensive_report.jl <input> <tau0> <m_list>")
        exit(1)
    end

    input_path = ARGS[1]
    tau0 = parse(Float64, ARGS[2])
    m_list = parse.(Int, split(ARGS[3], ','))
    
    # Load data
    raw = readdlm(input_path)
    x = vec(raw[:, 1])
    
    dev_funcs = [:adev, :mdev, :hdev, :mhdev, :tdev, :totdev, :mtotdev, :htotdev, :mhtotdev, :ldev]
    
    # Header
    println("func,m,tau,sigma,ci_lo,ci_hi,sigma_unbiased,ci_lo_unbiased,ci_hi_unbiased")
    
    for f in dev_funcs
        res = getfield(SigmaTau, f)(x, tau0; m_list=m_list)
        
        # Calculate unbiased results for totdev, mtotdev, and htotdev
        # Stable32 doesn't apply bias correction for these.
        
        res_unbiased = res
        fname = string(f)
        if fname in ["totdev", "mtotdev", "htotdev"]
             T_rec = (res.N - 1) * res.tau0
             # Map function name to bias type
             bias_type = fname == "totdev" ? "totvar" : 
                         fname == "mtotdev" ? "mtot" : "htot"
             
             B = SigmaTau.bias_correction(res.alpha, bias_type, res.tau, T_rec)
             
             raw_sigma = res.deviation
             if fname in ["totdev", "mtotdev", "htotdev"]
                raw_sigma = res.deviation .* B
             end
             
             # Recompute CI for raw sigma
             res_tmp = SigmaTau.DeviationResult(
                 res.tau, raw_sigma, res.edf, res.ci,
                 res.alpha, res.neff, res.tau0, res.N, res.method, res.confidence
             )
             res_unbiased = SigmaTau.compute_ci(res_tmp)
        end

        for i in 1:length(res.tau)
            # res.ci is a Matrix size (L, 2)
            lo = res.ci[i, 1]
            hi = res.ci[i, 2]
            ci_lo = isnan(lo) ? "NaN" : string(lo)
            ci_hi = isnan(hi) ? "NaN" : string(hi)
            
            # Unbiased values
            sig_unb = res_unbiased.deviation[i]
            lo_unb = res_unbiased.ci[i, 1]
            hi_unb = res_unbiased.ci[i, 2]
            
            ci_lo_unb = isnan(lo_unb) ? "NaN" : string(lo_unb)
            ci_hi_unb = isnan(hi_unb) ? "NaN" : string(hi_unb)
            
            # Calculate m from tau and tau0
            m_val = round(Int, res.tau[i] / res.tau0)
            println(string(f), ",", m_val, ",", res.tau[i], ",", res.deviation[i], ",", ci_lo, ",", ci_hi, ",", sig_unb, ",", ci_lo_unb, ",", ci_hi_unb)
        end
    end
end

main()
