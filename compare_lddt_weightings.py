#!/usr/bin/env python3
"""
Compare lDDT scores with different distance weighting schemes.
Creates plots to visualize how different weighting exponents affect the scores.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import base64
from lddt_weighted import LDDTCalculatorWeighted

def decode_coords(b64str):
    """Decode base64 coordinates."""
    arr = np.frombuffer(base64.b64decode(b64str), dtype=np.float32)
    return arr.reshape(-1, 3)

def compare_weighting_schemes(csv_path):
    """
    Compare lDDT scores with different weighting exponents.
    
    Args:
        csv_path: Path to CSV file with coords and mutant_coords columns
    """
    
    # Read data
    df = pd.read_csv(csv_path)
    
    # Define weighting schemes to compare
    weighting_schemes = {
        'Equal (exp=0.0)': 0.0,
        'Linear (exp=1.0)': 1.0,
        'Squared (exp=2.0)': 2.0,
        'Cubic (exp=3.0)': 3.0
    }
    
    # Store results for each protein
    all_results = {}
    
    print(f"Processing {len(df)} protein pairs...")
    
    for idx, row in df.iterrows():
        try:
            coords = decode_coords(row["coords"])
            mutant_coords = decode_coords(row["mutant_coords"])
            
            if coords.shape != mutant_coords.shape:
                print(f"Skipping {row.get('PID', idx)}: coordinate shapes don't match")
                continue
            
            pid = row.get('PID', f'protein_{idx}')
            description = row.get('description', '')
            
            print(f"Processing {pid}: {description}")
            
            # Calculate lDDT scores for each weighting scheme
            protein_results = {}
            
            for scheme_name, weight_exp in weighting_schemes.items():
                calculator = LDDTCalculatorWeighted(weight_exponent=weight_exp)
                lddt_scores = calculator.calculate_lddt(mutant_coords, coords)
                protein_results[scheme_name] = lddt_scores
                print(f"  {scheme_name}: scores[0:5] = {lddt_scores[:5]}")
            
            all_results[pid] = {
                'description': description,
                'scores': protein_results,
                'sequence_length': len(coords)
            }
            
        except Exception as e:
            print(f"Error processing {row.get('PID', idx)}: {e}")
            continue
    
    # Create comparison plots
    create_weighting_comparison_plots(all_results)
    
    return all_results

def create_weighting_comparison_plots(all_results):
    """Create plots comparing different weighting schemes."""
    
    # Get schemes from the first protein
    schemes = list(all_results[list(all_results.keys())[0]]['scores'].keys())
    
    # 1. Overall comparison plot - just one plot showing all schemes
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('lDDT Score Comparison: Different Distance Weighting Schemes', fontsize=16)
    
    # Plot separate plot for each protein
    colors = ['blue', 'red', 'green', 'orange']
    
    for pid, data in all_results.items():
        # Create a new figure for each protein
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        seq_len = data['sequence_length']
        positions = np.arange(seq_len)
        
        for i, scheme in enumerate(schemes):
            scores = data['scores'][scheme]
            ax.plot(positions, scores, label=scheme, 
                   color=colors[i], alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Residue Position')
        ax.set_ylabel('lDDT Score')
        ax.set_title(f'{pid}: {data["description"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 3. Summary statistics
    summary_stats = {}
    for scheme in schemes:
        all_scores = []
        for data in all_results.values():
            all_scores.extend(data['scores'][scheme])
        
        summary_stats[scheme] = {
            'mean': np.mean(all_scores),
            'std': np.std(all_scores),
            'min': np.min(all_scores),
            'max': np.max(all_scores),
            'count': len(all_scores)
        }
    
    # Print summary statistics
    summary_df = pd.DataFrame(summary_stats).T
    
    print("\nSummary Statistics:")
    print(summary_df)
    
    return summary_stats

def main():
    """Main function to run the comparison."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python compare_lddt_weightings.py <csv_file>")
        print("CSV should contain: coords, mutant_coords, PID, description columns")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    print("Comparing lDDT weighting schemes...")
    results = compare_weighting_schemes(csv_path)
    
    print(f"\nProcessed {len(results)} protein pairs")
    print("Check the output directory for detailed plots and statistics")

if __name__ == "__main__":
    main() 