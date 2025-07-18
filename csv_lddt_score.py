#!/usr/bin/env python3
"""
Compute per-residue lDDT scores between mutant and original coordinates from a CSV file.
Outputs a CSV with columns: PID, sequence, mutated_sequence, lddt_scores
Usage: python csv_lddt_score.py mutation_info2.csv
"""
import sys
import numpy as np
import pandas as pd
import base64
import csv
from lddt import LDDTCalculator

if len(sys.argv) != 2:
    print("Usage: python csv_lddt_score.py mutation_info2.csv")
    sys.exit(1)

input_csv = sys.argv[1]
output_csv = input_csv.replace('.csv', '_lddt.csv')

def decode_coords(b64str):
    arr = np.frombuffer(base64.b64decode(b64str), dtype=np.float32)
    return arr.reshape(-1, 3)

df = pd.read_csv(input_csv)
calculator = LDDTCalculator()

with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ["PID", "sequence", "mutated_sequence", "lddt_scores"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for idx, row in df.iterrows():
        try:
            coords = decode_coords(row["coords"])
            mutant_coords = decode_coords(row["mutant_coords"])
            if coords.shape != mutant_coords.shape:
                print(f"Skipping {row['PID']}: coordinate shapes do not match ({coords.shape} vs {mutant_coords.shape})")
                continue
            # Distance check at mutation site
            seq = row["sequence"]
            mut_seq = row["mutated_sequence"]
            if isinstance(seq, str) and isinstance(mut_seq, str) and len(seq) == len(mut_seq):
                diff_indices = [i for i, (a, b) in enumerate(zip(seq, mut_seq)) if a != b]
                if len(diff_indices) == 1:
                    i = diff_indices[0]
                    dist = np.linalg.norm(coords[i] - mutant_coords[i])
                    print(f"{row['PID']}: mutation at {i} ({seq[i]}->{mut_seq[i]}), CA distance = {dist:.3f} Å")
            lddt_scores = calculator.calculate_lddt(mutant_coords, coords)
            writer.writerow({
                "PID": row["PID"],
                "sequence": row["sequence"],
                "mutated_sequence": row["mutated_sequence"],
                "lddt_scores": list(lddt_scores)
            })
        except Exception as e:
            print(f"Error processing {row['PID']}: {e}")
print(f"Wrote lDDT results to {output_csv}") 