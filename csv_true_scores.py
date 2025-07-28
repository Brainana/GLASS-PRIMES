#!/usr/bin/env python3
"""
Compute true lDDT, TM scores between mutant and original coordinates from a CSV file.
Outputs a CSV with columns: PID, sequence, mutated_sequence, lddt_scores, tm_score, description
"""
import sys
import numpy as np
import pandas as pd
import base64
import csv
from lddt_weighted import LDDTCalculatorWeighted
from tmtools import tm_align

if len(sys.argv) != 2:
    print("Usage: python csv_true_scores.py <>.csv")
    sys.exit(1)

input_csv = sys.argv[1]
output_csv = input_csv.replace('.csv', '_lddt.csv')

def decode_coords(b64str):
    arr = np.frombuffer(base64.b64decode(b64str), dtype=np.float32)
    return arr.reshape(-1, 3)


def compute_tm_score(original_coords, mutant_coords, original_seq, mutant_seq):
    try:
        # Run TM-align
        result = tm_align(original_coords, mutant_coords, original_seq, mutant_seq)
        
        # Debug: Print TM-align results
        print(f"  TM-score (chain1): {result.tm_norm_chain1:.4f}")
        print(f"  TM-score (chain2): {result.tm_norm_chain2:.4f}")
        print(f"  RMSD: {result.rmsd:.4f}")
        print(f"  Number of aligned residues: {len(result.seqxA)}")
        
        # Return TM-score normalized by the original structure length
        return result.tm_norm_chain1
    except Exception as e:
        print(f"Error computing TM-score: {e}")
        return None


df = pd.read_csv(input_csv)
calculator = LDDTCalculatorWeighted(weight_exponent=3.0)

with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ["PID", "sequence", "mutated_sequence", "lddt_scores", "tm_score", "description"]
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
            print(f"\nProcessing {row['PID']}:")
            print(f"  Original sequence: {seq}")
            print(f"  Mutant sequence: {mut_seq}")
            
            if isinstance(seq, str) and isinstance(mut_seq, str) and len(seq) == len(mut_seq):
                diff_indices = [i for i, (a, b) in enumerate(zip(seq, mut_seq)) if a != b]
                print(f"  Differences found at positions: {diff_indices}")
                if len(diff_indices) == 1:
                    i = diff_indices[0]
                    dist = np.linalg.norm(coords[i] - mutant_coords[i])
                    print(f"  Mutation at {i} ({seq[i]}->{mut_seq[i]}), CA distance = {dist:.3f} Å")
                    
                    # Check if coordinates actually changed
                    if dist < 0.1:
                        print(f"  WARNING: Very small coordinate change ({dist:.3f} Å) - mutation may not have been applied!")
                else:
                    print(f"  WARNING: Found {len(diff_indices)} differences, expected 1 for point mutation")
            else:
                print(f"  WARNING: Sequence lengths don't match or invalid sequences")
            lddt_scores = calculator.calculate_lddt(coords, mutant_coords)
            
            # Compute TM-score between original and mutant structures
            tm_score = compute_tm_score(coords, mutant_coords, row["sequence"], row["mutated_sequence"])
            
            writer.writerow({
                "PID": row["PID"],
                "sequence": row["sequence"],
                "mutated_sequence": row["mutated_sequence"],
                "lddt_scores": list(lddt_scores),
                "tm_score": tm_score,
                "description": row["description"] if "description" in row else ""
            })
        except Exception as e:
            print(f"Error processing {row['PID']}: {e}")
print(f"Wrote lDDT results to {output_csv}") 