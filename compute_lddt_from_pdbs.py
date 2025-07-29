#!/usr/bin/env python3
"""
Compute lDDT scores between two PDB files.
Usage: python compute_lddt_from_pdbs.py <reference_pdb> <model_pdb>
"""
import sys
import numpy as np
import pandas as pd
from lddt_weighted import LDDTCalculatorWeighted
from extract_pdb_info import PDBInfoExtractor
import argparse
import os

def compute_lddt_between_pdbs(reference_pdb, model_pdb, weight_exponent=5.0):
    """
    Compute lDDT scores between two PDB files.
    
    Args:
        reference_pdb: Path to reference PDB file
        model_pdb: Path to model PDB file
        weight_exponent: Exponent for distance weighting (default: 5.0)
    
    Returns:
        dict: Dictionary containing lDDT scores and statistics
    """
    
    # Initialize lDDT calculator
    calculator = LDDTCalculatorWeighted(weight_exponent=weight_exponent)
    
    # Extract coordinates and sequences
    extractor = PDBInfoExtractor()
    
    print(f"Extracting coordinates from {reference_pdb}...")
    ref_coords, ref_seq = extractor.extract_ca_coords_and_sequence(reference_pdb)
    
    print(f"Extracting coordinates from {model_pdb}...")
    model_coords, model_seq = extractor.extract_ca_coords_and_sequence(model_pdb)
    
    # Check if sequences match
    if ref_seq != model_seq:
        print(f"WARNING: Sequences don't match!")
        print(f"Reference sequence: {ref_seq}")
        print(f"Model sequence: {model_seq}")
        print("Proceeding with coordinate-based comparison...")
    
    # Check coordinate shapes
    print(f"Reference coordinates shape: {ref_coords.shape}")
    print(f"Model coordinates shape: {model_coords.shape}")
    
    if ref_coords.shape != model_coords.shape:
        print(f"ERROR: Coordinate shapes don't match!")
        print(f"Reference: {ref_coords.shape}, Model: {model_coords.shape}")
        return None
    
    # Compute lDDT scores
    print("Computing lDDT scores...")
    lddt_scores = calculator.calculate_lddt(ref_coords, model_coords)
    
    # Calculate statistics
    mean_lddt = np.mean(lddt_scores)
    std_lddt = np.std(lddt_scores)
    min_lddt = np.min(lddt_scores)
    max_lddt = np.max(lddt_scores)
    median_lddt = np.median(lddt_scores)
    
    # Find best and worst residues
    best_idx = np.argmax(lddt_scores)
    worst_idx = np.argmin(lddt_scores)
    
    results = {
        'reference_pdb': reference_pdb,
        'model_pdb': model_pdb,
        'sequence_length': len(lddt_scores),
        'lddt_scores': lddt_scores.tolist(),
        'reference_sequence': ref_seq,
        'model_sequence': model_seq,
        'statistics': {
            'mean': mean_lddt,
            'std': std_lddt,
            'min': min_lddt,
            'max': max_lddt,
            'median': median_lddt
        },
        'best_residue': {
            'position': best_idx + 1,  # 1-based indexing
            'score': lddt_scores[best_idx],
            'amino_acid': ref_seq[best_idx] if ref_seq else 'N/A'
        },
        'worst_residue': {
            'position': worst_idx + 1,  # 1-based indexing
            'score': lddt_scores[worst_idx],
            'amino_acid': ref_seq[worst_idx] if ref_seq else 'N/A'
        }
    }
    
    return results

def print_results(results):
    """Print formatted results."""
    if results is None:
        print("ERROR: Could not compute lDDT scores")
        return
    
    print("\n" + "="*60)
    print("LDDT SCORE ANALYSIS")
    print("="*60)
    print(f"Reference PDB: {results['reference_pdb']}")
    print(f"Model PDB: {results['model_pdb']}")
    print(f"Sequence length: {results['sequence_length']}")
    print()
    
    stats = results['statistics']
    print("LDDT SCORE STATISTICS:")
    print(f"  Mean lDDT: {stats['mean']:.4f}")
    print(f"  Standard deviation: {stats['std']:.4f}")
    print(f"  Minimum: {stats['min']:.4f}")
    print(f"  Maximum: {stats['max']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print()
    
    best = results['best_residue']
    worst = results['worst_residue']
    
    print("BEST RESIDUE:")
    print(f"  Position: {best['position']}")
    print(f"  Score: {best['score']:.4f}")
    print(f"  Amino acid: {best['amino_acid']}")
    print()
    
    print("WORST RESIDUE:")
    print(f"  Position: {worst['position']}")
    print(f"  Score: {worst['score']:.4f}")
    print(f"  Amino acid: {worst['amino_acid']}")
    print("="*60)

def save_to_csv(results, pid):
    """Save results to CSV file."""
    # Create DataFrame with one row containing all scores
    df_data = [{
        'PID': pid,
        'sequence': results['reference_sequence'],
        'mutated_sequence': results['model_sequence'],
        'lddt_scores': results['lddt_scores']
    }]
    
    df = pd.DataFrame(df_data)
    
    # Generate output filename
    output_csv = f"{pid}_lddt_scores.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved lDDT scores to: {output_csv}")
    print(f"CSV contains 1 row with {len(results['lddt_scores'])} lDDT scores")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compute lDDT scores between two PDB files"
    )
    parser.add_argument(
        'reference_pdb', 
        type=str, 
        help='Path to reference PDB file'
    )
    parser.add_argument(
        'model_pdb', 
        type=str, 
        help='Path to model PDB file'
    )
    parser.add_argument(
        '--weight-exponent', 
        type=float, 
        default=0.0,
        help='Exponent for distance weighting (default: 3.0)'
    )
    parser.add_argument(
        '--pid', 
        type=str, 
        default='protein',
        help='Protein ID for the CSV output (default: protein)'
    )

    
    args = parser.parse_args()
    
    # Check if PDB files exist
    import os
    if not os.path.exists(args.reference_pdb):
        print(f"ERROR: Reference PDB file not found: {args.reference_pdb}")
        sys.exit(1)
    
    if not os.path.exists(args.model_pdb):
        print(f"ERROR: Model PDB file not found: {args.model_pdb}")
        sys.exit(1)
    
    # Compute lDDT scores
    results = compute_lddt_between_pdbs(
        args.reference_pdb, 
        args.model_pdb, 
        weight_exponent=args.weight_exponent
    )
    
    # Print results
    print_results(results)
    
    # Save to CSV
    if results:
        save_to_csv(results, args.pid)

if __name__ == "__main__":
    main() 