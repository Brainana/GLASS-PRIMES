#!/usr/bin/env python3
"""
Compare lDDT scores with different Gaussian sigma weighting schemes.
Creates plots to visualize how different Gaussian sigma values affect the scores.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import base64
from lddt_gaussian_weight import LDDTCalculatorGaussian

def decode_coords(b64str):
    """Decode base64 coordinates."""
    arr = np.frombuffer(base64.b64decode(b64str), dtype=np.float32)
    return arr.reshape(-1, 3)

def compare_gaussian_weighting_schemes(csv_path):
    """
    Compare lDDT scores with different Gaussian sigma weighting schemes.
    
    Args:
        csv_path: Path to CSV file with coords and mutant_coords columns
    """
    
    # Read data
    df = pd.read_csv(csv_path)
    
    # Define Gaussian sigma weighting schemes to compare
    gaussian_schemes = {
        'Very Local (σ=1.0)': 1.0,
        'Local (σ=2.0)': 2.0,
        'Balanced (σ=5.0)': 5.0,
        'Extended (σ=8.0)': 8.0,
        'Global (σ=12.0)': 12.0,
        'Very Global (σ=15.0)': 15.0
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
            
            # Find mutation site
            mutation_site = row['mutation_start']
            
            print(f"Processing {pid}: {description}")
            
            # Calculate lDDT scores for each Gaussian sigma scheme
            protein_results = {}
            
            for scheme_name, sigma in gaussian_schemes.items():
                calculator = LDDTCalculatorGaussian(sigma=sigma)
                lddt_scores = calculator.calculate_lddt(coords, mutant_coords)
                protein_results[scheme_name] = lddt_scores
                print(f"  {scheme_name}: mean score = {np.mean(lddt_scores):.4f}")
            
            all_results[pid] = {
                'description': description,
                'scores': protein_results,
                'sequence_length': len(coords),
                'mutation_site': mutation_site
            }
            
        except Exception as e:
            print(f"Error processing {row.get('PID', idx)}: {e}")
            continue

        break  # Process only first protein for now
    
    # Create comparison plots
    create_gaussian_weighting_comparison_plots(all_results)
    
    return all_results

def create_gaussian_weighting_comparison_plots(all_results):
    """Create plots comparing different Gaussian sigma weighting schemes."""
    
    # Get schemes from the first protein
    schemes = list(all_results[list(all_results.keys())[0]]['scores'].keys())
    
    # 1. Overall comparison plot - just one plot showing all schemes
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('lDDT Score Comparison: Different Gaussian Sigma Weighting Schemes', fontsize=16)
    
    # Plot all schemes for one protein
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Get the first (and only) protein
    pid, data = list(all_results.items())[0]
    seq_len = data['sequence_length']
    positions = np.arange(seq_len)
    
    for i, scheme in enumerate(schemes):
        scores = data['scores'][scheme]
        ax.plot(positions, scores, label=scheme, 
               color=colors[i], alpha=0.8, linewidth=2)
    
    # Add mutation site line if available
    if data.get('mutation_site') is not None:
        ax.axvline(x=data['mutation_site'], color='red', linestyle='--', alpha=0.7, 
                   label=f'Mutation site ({data["mutation_site"]})')
    
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('lDDT Score')
    ax.set_title(f'{pid}: {data["description"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.0)  # Focus on typical lDDT score range
    
    plt.tight_layout()
    plt.show()
    
    # 2. Create a heatmap showing the difference between schemes
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calculate differences from the most local scheme (σ=1.0)
    baseline_scheme = 'Very Local (σ=1.0)'
    baseline_scores = data['scores'][baseline_scheme]
    
    # Create difference matrix
    diff_matrix = []
    scheme_names = []
    for scheme in schemes:
        if scheme != baseline_scheme:
            scores = data['scores'][scheme]
            diff = scores - baseline_scores
            diff_matrix.append(diff)
            scheme_names.append(scheme)
    
    if diff_matrix:
        diff_matrix = np.array(diff_matrix)
        
        # Create heatmap
        im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', 
                       vmin=-0.1, vmax=0.1)
        
        ax.set_yticks(range(len(scheme_names)))
        ax.set_yticklabels(scheme_names)
        ax.set_xlabel('Residue Position')
        ax.set_title(f'Difference from {baseline_scheme} (σ=1.0)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score Difference')
        
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
        print("Usage: python compare_lddt_gaussian_weightings.py <csv_file>")
        print("CSV should contain: coords, mutant_coords, PID, description columns")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    print("Comparing lDDT Gaussian weighting schemes...")
    results = compare_gaussian_weighting_schemes(csv_path)
    
    print(f"\nProcessed {len(results)} protein pairs")
    print("Check the output directory for detailed plots and statistics")

if __name__ == "__main__":
    main() 