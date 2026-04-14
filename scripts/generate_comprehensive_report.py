import numpy as np
import allantools as at
import pandas as pd
import subprocess
import json
import os

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
    np.savetxt("tmp_phase.txt", phase_data)
    
    # Julia script to compute all deviations and output to CSV
    m_list_str = "[" + ", ".join(map(str, m_list)) + "]"
    julia_code = f"""
    using SigmaTau
    using DelimitedFiles

    # Readdlm might return a matrix; take first column and ensure it is a vector
    raw = readdlm("tmp_phase.txt")
    x = vec(raw[:, 1])
    tau0 = {tau0}
    m_list = {m_list_str}
    
    dev_funcs = [:adev, :mdev, :hdev, :mhdev, :tdev, :totdev, :mtotdev, :htotdev, :mhtotdev, :ldev]
    
    # Header
    println("func,m,tau,sigma,ci_lo,ci_hi")
    
    for f in dev_funcs
        res = getfield(SigmaTau, f)(x, tau0; m_list=m_list)
        for i in 1:length(res.tau)
            # res.ci is a Matrix size (L, 2)
            lo = res.ci[i, 1]
            hi = res.ci[i, 2]
            ci_lo = isnan(lo) ? "NaN" : string(lo)
            ci_hi = isnan(hi) ? "NaN" : string(hi)
            # Calculate m from tau and tau0
            m_val = round(Int, res.tau[i] / res.tau0)
            println(string(f), ",", m_val, ",", res.tau[i], ",", res.deviation[i], ",", ci_lo, ",", ci_hi)
        end
    end
    """
    
    # Call Julia
    process = subprocess.Popen(['julia', '--project=julia', '-e', julia_code], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if stderr:
        print(f"Julia Error: {stderr}")
    
    print(f"DEBUG: Julia stdout length: {len(stdout)}")
    if len(stdout) < 200:
        print(f"DEBUG: Julia stdout content: {stdout}")
        
    os.remove("tmp_phase.txt")
    
    if not stdout.strip():
        print("Julia returned empty output")
        return {}

    # Parse CSV output
    from io import StringIO
    try:
        # Check if we have more than just the header
        lines = stdout.strip().split('\n')
        if len(lines) <= 1:
            print("Julia returned only header or nothing")
            return {}
            
        df = pd.read_csv(StringIO(stdout))
    except Exception as e:
        print(f"Failed to parse Julia CSV: {e}")
        print("Raw output:")
        print(stdout)
        return {}
    
    results = {}
    for func in df['func'].unique():
        func_df = df[df['func'] == func]
        results[func] = {
            "tau": func_df['tau'].tolist(),
            "sigma": func_df['sigma'].tolist(),
            "ci_lo": func_df['ci_lo'].tolist(),
            "ci_hi": func_df['ci_hi'].tolist()
        }
    return results

def main():
    data_path = "reference/stable32gen.DAT"
    s32_full_csv = "reference/stable32out/stable32_data_full.csv"
    
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
        # Filter for the specific taus in this subset
        target_taus = sorted(s32_subset['Tau'].unique())
        try:
            func = getattr(at, at_func)
            # PASS THE TARGET TAUS EXPLICITLY TO MATCH STABLE32
            at_taus, at_sigma, at_err, at_n = func(x, rate=1.0/tau0, data_type="phase", taus=target_taus)
            # Create a map for easy lookup
            at_map = dict(zip(at_taus, at_sigma))
        except Exception as e:
            at_map = {}
            print(f"allantools error for {at_func}: {e}")
            
        # Get SigmaTau results
        st_data = st_results.get(st_func, {})
        st_sigma_list = st_data.get("sigma", [])
        st_tau_list = st_data.get("tau", [])
        st_ci_lo_list = st_data.get("ci_lo", [])
        st_ci_hi_list = st_data.get("ci_hi", [])
        
        st_map = dict(zip(st_tau_list, st_sigma_list))
        st_ci_lo_map = dict(zip(st_tau_list, st_ci_lo_list))
        st_ci_hi_map = dict(zip(st_tau_list, st_ci_hi_list))
        
        for i, row in s32_subset.iterrows():
            m = int(row['AF'])
            tau = float(row['Tau'])
            s32_sigma = float(row['Sigma'])
            s32_ci_lo = float(row['MinSigma']) if row['MinSigma'] else None
            s32_ci_hi = float(row['MaxSigma']) if row['MaxSigma'] else None
            
            at_val = at_map.get(tau)
            st_val = st_map.get(tau)
            
            report.append({
                "Type": s32_type,
                "m": m,
                "Tau": tau,
                "Stable32": s32_sigma,
                "allantools": at_val,
                "SigmaTau": st_val,
                "S32_CI_Lo": s32_ci_lo,
                "S32_CI_Hi": s32_ci_hi,
                "ST_CI_Lo": st_ci_lo_map.get(tau),
                "ST_CI_Hi": st_ci_hi_map.get(tau),
            })
            
    # Also add types only in SigmaTau/allantools
    # e.g., mhdev, ldev, mhtotdev
    
    report_df = pd.DataFrame(report)
    report_df.to_csv("reference/stable32out/comprehensive_comparison.csv", index=False)
    
    # Generate Markdown Report
    with open("reference/stable32out/comprehensive_comparison.md", "w") as f:
        f.write("# Comprehensive Comparison: Stable32 vs allantools vs SigmaTau\n\n")
        
        for s32_type in report_df['Type'].unique():
            f.write(f"## {s32_type}\n\n")
            subset = report_df[report_df['Type'] == s32_type]
            
            # Create a nice table
            f.write("| Tau | Stable32 | allantools | SigmaTau | S32 CI | ST CI |\n")
            f.write("|:---|:---|:---|:---|:---|:---|\n")
            for _, row in subset.iterrows():
                s32_ci = f"[{row['S32_CI_Lo']:.2e}, {row['S32_CI_Hi']:.2e}]" if not np.isnan(row['S32_CI_Lo']) else "N/A"
                st_ci = f"[{row['ST_CI_Lo']:.2e}, {row['ST_CI_Hi']:.2e}]" if row['ST_CI_Lo'] is not None and not np.isnan(row['ST_CI_Lo']) else "N/A"
                st_val = f"{row['SigmaTau']:.5e}" if row['SigmaTau'] is not None and not np.isnan(row['SigmaTau']) else "N/A"
                at_val = f"{row['allantools']:.5e}" if row['allantools'] is not None and not np.isnan(row['allantools']) else "N/A"
                f.write(f"| {row['Tau']:.1e} | {row['Stable32']:.5e} | {at_val} | {st_val} | {s32_ci} | {st_ci} |\n")
            f.write("\n")

if __name__ == "__main__":
    main()
