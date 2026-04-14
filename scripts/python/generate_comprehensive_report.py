import numpy as np
import allantools as at
import pandas as pd
import subprocess
import os
import tempfile
from io import StringIO

def read_stable32_data(filepath):
    phase_data = []
    with open(filepath, 'r') as f:
        header_passed = False
        for line in f:
            if "# Header End" in line:
                header_passed = True
                continue
            if header_passed:
                try:
                    phase_data.append(float(line.strip()))
                except ValueError:
                    continue
    return np.array(phase_data)

def get_sigmatau_results(phase_data, tau0, m_list):
    # Prepare a temporary data file for Julia to read
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        np.savetxt(tmp.name, phase_data)
        tmp_path = tmp.name
    
    try:
        # Julia script to compute all deviations and output to CSV
        m_list_str = "[" + ", ".join(map(str, m_list)) + "]"
        julia_code = f"""
        using SigmaTau
        using DelimitedFiles

        # Readdlm might return a matrix; take first column and ensure it is a vector
        raw = readdlm("{tmp_path.replace('\\', '/')}")
        x = vec(raw[:, 1])
        tau0 = {tau0}
        m_list = {m_list_str}
        
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
                 
                 # sigma_biased = res.deviation
                 # We want the 'res' to hold the biased version and 'res_unbiased' to hold the unbiased one.
                 # Since totdev, mtotdev, and htotdev are all biased by default now in SigmaTau.engine,
                 # res already holds the biased version.
                 
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
        """
        
        # Call Julia
        process = subprocess.run(['julia', '--project=julia', '-e', julia_code], 
                                 capture_output=True, text=True)
        
        if process.returncode != 0:
            print(f"Julia Error (Exit Code {process.returncode}):")
            print(process.stderr)
            raise RuntimeError("Julia SigmaTau execution failed")

        stdout = process.stdout
        if not stdout.strip():
            print("Julia returned empty output")
            return {}

        # Parse CSV output
        try:
            df = pd.read_csv(StringIO(stdout))
        except Exception as e:
            print(f"Failed to parse Julia CSV: {e}")
            return {}
        
        results = {}
        for func in df['func'].unique():
            func_df = df[df['func'] == func]
            results[func] = {
                "tau": func_df['tau'].tolist(),
                "sigma": func_df['sigma'].tolist(),
                "ci_lo": func_df['ci_lo'].tolist(),
                "ci_hi": func_df['ci_hi'].tolist(),
                "sigma_unbiased": func_df['sigma_unbiased'].tolist(),
                "ci_lo_unbiased": func_df['ci_lo_unbiased'].tolist(),
                "ci_hi_unbiased": func_df['ci_hi_unbiased'].tolist()
            }
        return results
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def main():
    data_path = "reference/validation/stable32gen.DAT"
    s32_full_csv = "reference/validation/stable32out/stable32_data_full.csv"
    
    x = read_stable32_data(data_path)
    tau0 = 1.0
    
    s32_df = pd.read_csv(s32_full_csv)
    
    # Get unique m values from Stable32 results
    m_list = sorted(s32_df['AF'].unique())
    
    print("Computing SigmaTau results...")
    st_results = get_sigmatau_results(x, tau0, m_list)
    print(f"Computed SigmaTau results for: {list(st_results.keys())}")
    
    # Mapping S32 -> allantools -> SigmaTau
    mapping = [
        ("Overlapping Allan", "oadev", "adev"),
        ("Modified Allan", "mdev", "mdev"),
        ("Time", "tdev", "tdev"),
        ("Overlapping Hadamard", "ohdev", "hdev"),
        ("Total", "totdev", "totdev"),
        ("Modified Total", "mtotdev", "mtotdev"),
        ("Hadamard Total", "htotdev", "htotdev"),
    ]
    
    report = []
    
    for s32_type, at_func, st_func in mapping:
        s32_subset = s32_df[s32_df['Type'] == s32_type]
        if s32_subset.empty:
            continue
            
        print(f"Processing {s32_type}...")
        
        # Compute with allantools
        target_taus = sorted(s32_subset['Tau'].unique())
        try:
            func = getattr(at, at_func)
            at_taus, at_sigma, at_err, at_n = func(x, rate=1.0/tau0, data_type="phase", taus=target_taus)
            at_map = dict(zip(at_taus, at_sigma))
        except Exception as e:
            at_map = {}
            print(f"allantools error for {at_func}: {e}")
            
        # Get SigmaTau results
        st_data = st_results.get(st_func, {})
        st_tau_list = st_data.get("tau", [])
        
        st_map = dict(zip(st_tau_list, st_data.get("sigma", [])))
        st_ci_lo_map = dict(zip(st_tau_list, st_data.get("ci_lo", [])))
        st_ci_hi_map = dict(zip(st_tau_list, st_data.get("ci_hi", [])))
        
        st_unb_map = dict(zip(st_tau_list, st_data.get("sigma_unbiased", [])))
        st_ci_lo_unb_map = dict(zip(st_tau_list, st_data.get("ci_lo_unbiased", [])))
        st_ci_hi_unb_map = dict(zip(st_tau_list, st_data.get("ci_hi_unbiased", [])))
        
        for i, row in s32_subset.iterrows():
            tau = float(row['Tau'])
            
            report.append({
                "Type": s32_type,
                "m": int(row['AF']),
                "Tau": tau,
                "Stable32": float(row['Sigma']),
                "allantools": at_map.get(tau),
                "SigmaTau": st_map.get(tau),
                "SigmaTau_Unbiased": st_unb_map.get(tau),
                "S32_CI_Lo": float(row['MinSigma']) if row['MinSigma'] else None,
                "S32_CI_Hi": float(row['MaxSigma']) if row['MaxSigma'] else None,
                "ST_CI_Lo": st_ci_lo_map.get(tau),
                "ST_CI_Hi": st_ci_hi_map.get(tau),
                "ST_CI_Lo_Unbiased": st_ci_lo_unb_map.get(tau),
                "ST_CI_Hi_Unbiased": st_ci_hi_unb_map.get(tau),
            })
            
    report_df = pd.DataFrame(report)
    report_df.to_csv("reference/validation/stable32out/comprehensive_comparison.csv", index=False)
    
    # Generate Markdown Report
    with open("reference/validation/stable32out/comprehensive_comparison.md", "w") as f:
        f.write("# Comprehensive Comparison: Stable32 vs allantools vs SigmaTau\n\n")
        
        for s32_type in report_df['Type'].unique():
            f.write(f"## {s32_type}\n\n")
            subset = report_df[report_df['Type'] == s32_type]
            
            is_total = s32_type in ["Total", "Modified Total", "Hadamard Total"]
            
            # Create a nice table
            if is_total:
                f.write("| Tau | Stable32 | SigmaTau (Biased) | SigmaTau (Unbiased) | S32 CI | ST CI (Unbiased) |\n")
                f.write("|:---|:---|:---|:---|:---|:---|\n")
            else:
                f.write("| Tau | Stable32 | allantools | SigmaTau | S32 CI | ST CI |\n")
                f.write("|:---|:---|:---|:---|:---|:---|\n")
                
            for _, row in subset.iterrows():
                s32_ci = f"[{row['S32_CI_Lo']:.2e}, {row['S32_CI_Hi']:.2e}]" if not np.isnan(row['S32_CI_Lo']) else "N/A"
                
                if is_total:
                    st_ci_unb = f"[{row['ST_CI_Lo_Unbiased']:.2e}, {row['ST_CI_Hi_Unbiased']:.2e}]" if row['ST_CI_Lo_Unbiased'] is not None and not np.isnan(row['ST_CI_Lo_Unbiased']) else "N/A"
                    st_biased = f"{row['SigmaTau']:.5e}" if row['SigmaTau'] is not None and not np.isnan(row['SigmaTau']) else "N/A"
                    st_unbiased = f"{row['SigmaTau_Unbiased']:.5e}" if row['SigmaTau_Unbiased'] is not None and not np.isnan(row['SigmaTau_Unbiased']) else "N/A"
                    f.write(f"| {row['Tau']:.1e} | {row['Stable32']:.5e} | {st_biased} | {st_unbiased} | {s32_ci} | {st_ci_unb} |\n")
                else:
                    st_ci = f"[{row['ST_CI_Lo']:.2e}, {row['ST_CI_Hi']:.2e}]" if row['ST_CI_Lo'] is not None and not np.isnan(row['ST_CI_Lo']) else "N/A"
                    st_val = f"{row['SigmaTau']:.5e}" if row['SigmaTau'] is not None and not np.isnan(row['SigmaTau']) else "N/A"
                    at_val = f"{row['allantools']:.5e}" if row['allantools'] is not None and not np.isnan(row['allantools']) else "N/A"
                    f.write(f"| {row['Tau']:.1e} | {row['Stable32']:.5e} | {at_val} | {st_val} | {s32_ci} | {st_ci} |\n")
            f.write("\n")

if __name__ == "__main__":
    main()
