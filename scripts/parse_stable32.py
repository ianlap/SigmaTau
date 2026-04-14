import re
import csv

def parse_stable32_output(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Split by "STATISTICS FOR FILE:" to get different sections
    sections = content.split("STATISTICS FOR FILE:")
    
    all_data = []
    
    for section in sections:
        if not section.strip():
            continue
            
        # Extract Sigma Type
        type_match = re.search(r"Sigma Type:\s*(.+)", section)
        if not type_match:
            continue
        sigma_type = type_match.group(1).strip()
        
        # Find the table header to start parsing data
        # Headers vary slightly (e.g., Sigma, Mod Sigma, Hadamard Sig, Total Sigma)
        lines = section.splitlines()
        table_started = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for header
            if "AF" in line and "Tau" in line and "Sigma" in line:
                table_started = True
                continue
                
            if table_started:
                # Matches: AF, Tau, #, Alpha, Min Sigma, Sigma, Max Sigma
                # 1   1.0000e+00     8190     2   9.9898e-01    1.0097e+00     1.0209e+00
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        af = parts[0]
                        tau = parts[1]
                        n_points = parts[2]
                        alpha = parts[3]
                        min_sigma = parts[4]
                        sigma = parts[5]
                        max_sigma = parts[6] if len(parts) > 6 else ""
                        
                        all_data.append({
                            "Type": sigma_type,
                            "AF": af,
                            "Tau": tau,
                            "N": n_points,
                            "Alpha": alpha,
                            "MinSigma": min_sigma,
                            "Sigma": sigma,
                            "MaxSigma": max_sigma
                        })
                    except (ValueError, IndexError):
                        # Some lines might be incomplete or footer
                        continue
                else:
                    # Table might have ended
                    if any(c.isalpha() for c in line) and "e+" not in line:
                        table_started = False

    return all_data

if __name__ == "__main__":
    input_file = "reference/stable32out/stable32out.txt"
    output_file = "reference/stable32out/stable32_data_full.csv"
    
    data = parse_stable32_output(input_file)
    
    if data:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"Successfully wrote {len(data)} rows to {output_file}")
    else:
        print("No data found to parse.")
