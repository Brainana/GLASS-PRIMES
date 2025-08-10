#!/usr/bin/env python3
"""
Analyze distance matrix for lDDT calculation.

This script takes a CSV file with protein structure data and generates a square matrix
visualization showing residue pairs and their distance differences for lDDT calculation.
Neighbor pairs (distance <= 15 Å) are colored based on distance difference thresholds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import argparse
import sys
from typing import List, Tuple, Dict, Any
import base64


def parse_score_list(score_str):
    """Parse a string representation of a list of scores."""
    try:
        if isinstance(score_str, str):
            try:
                return ast.literal_eval(score_str)
            except:
                try:
                    result = eval(score_str)
                    if isinstance(result, list):
                        return [float(x) for x in result]
                    return result
                except:
                    print(f"Warning: Could not parse score string: {score_str[:100]}...")
                    return []
        elif isinstance(score_str, list):
            return score_str
        else:
            return []
    except Exception as e:
        print(f"Error parsing score string: {e}")
        return []


def decode_coords(b64str):
    """Decode base64 coordinates."""
    arr = np.frombuffer(base64.b64decode(b64str), dtype=np.float32)
    return arr.reshape(-1, 3)


def parse_coordinates(coord_str):
    """Parse coordinate string from CSV."""
    try:
        if isinstance(coord_str, str):
            # Decode base64 coordinates using numpy
            coords_array = decode_coords(coord_str)
            return coords_array.tolist()
        elif isinstance(coord_str, list):
            return coord_str
        else:
            return []
    except Exception as e:
        print(f"Error parsing coordinates: {e}")
        return []


def calculate_distance_matrix(coords: List[List[float]]) -> np.ndarray:
    """Calculate distance matrix from coordinates."""
    if not coords:
        return np.array([])
    
    coords = np.array(coords)
    n_residues = len(coords)
    dist_matrix = np.zeros((n_residues, n_residues))
    
    for i in range(n_residues):
        for j in range(i+1, n_residues):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


def calculate_distance_differences(wild_coords: List[List[float]], 
                                 mutant_coords: List[List[float]]) -> np.ndarray:
    """Calculate distance differences between wild-type and mutant structures."""
    wild_dist = calculate_distance_matrix(wild_coords)
    mutant_dist = calculate_distance_matrix(mutant_coords)
    
    if wild_dist.size == 0 or mutant_dist.size == 0:
        return np.array([])
    
    # Ensure same size
    min_size = min(wild_dist.shape[0], mutant_dist.shape[0])
    wild_dist = wild_dist[:min_size, :min_size]
    mutant_dist = mutant_dist[:min_size, :min_size]
    
    return np.abs(wild_dist - mutant_dist)


def create_distance_difference_matrix(wild_coords: List[List[float]], 
                                    mutant_coords: List[List[float]], 
                                    distance_threshold: float = 15.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create matrix showing distance differences for neighbor pairs."""
    wild_dist = calculate_distance_matrix(wild_coords)
    mutant_dist = calculate_distance_matrix(mutant_coords)
    
    if wild_dist.size == 0 or mutant_dist.size == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Ensure same size
    min_size = min(wild_dist.shape[0], mutant_dist.shape[0])
    wild_dist = wild_dist[:min_size, :min_size]
    mutant_dist = mutant_dist[:min_size, :min_size]
    
    # Calculate distance differences
    dist_diff = np.abs(wild_dist - mutant_dist)
    
    # Create neighbor mask (pairs within distance threshold)
    neighbor_mask = wild_dist <= distance_threshold
    
    # Create colored matrix with continuous values for neighbor pairs
    colored_matrix = np.zeros_like(dist_diff)
    
    # For neighbor pairs, use actual distance difference values
    # For non-neighbor pairs, keep as 0 (will be transparent in plot)
    colored_matrix[neighbor_mask] = dist_diff[neighbor_mask]
    
    return colored_matrix, wild_dist, dist_diff


def plot_distance_matrix(colored_matrix: np.ndarray, 
                        wild_dist: np.ndarray,
                        dist_diff: np.ndarray,
                        protein_name: str,
                        output_prefix: str):
    """Plot the distance difference matrix."""
    if colored_matrix.size == 0:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Colored distance difference matrix
    im1 = axes[0].imshow(colored_matrix, cmap='RdYlGn_r', aspect='equal', vmin=0, vmax=10)
    axes[0].set_title(f'{protein_name}\nDistance Difference Matrix\n(Neighbor Pairs ≤ 15Å)')
    axes[0].set_xlabel('Residue Index')
    axes[0].set_ylabel('Residue Index')
    
    # Add colorbar with continuous scale
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Distance Difference (Å)')
    
    # Plot 2: Wild-type distance matrix
    im2 = axes[1].imshow(wild_dist, cmap='viridis', aspect='equal')
    axes[1].set_title('Wild-type Distance Matrix')
    axes[1].set_xlabel('Residue Index')
    axes[1].set_ylabel('Residue Index')
    plt.colorbar(im2, ax=axes[1], label='Distance (Å)')
    
    # Plot 3: Distance difference matrix (all pairs)
    im3 = axes[2].imshow(dist_diff, cmap='Reds', aspect='equal')
    axes[2].set_title('Distance Difference Matrix\n(All Pairs)')
    axes[2].set_xlabel('Residue Index')
    axes[2].set_ylabel('Residue Index')
    plt.colorbar(im3, ax=axes[2], label='Distance Difference (Å)')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_distance_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distance matrix plot to: {output_prefix}_distance_matrix.png")


def analyze_neighbor_pairs(colored_matrix: np.ndarray, 
                          wild_dist: np.ndarray,
                          dist_diff: np.ndarray) -> Dict[str, Any]:
    """Analyze neighbor pairs and their distance differences."""
    if colored_matrix.size == 0:
        return {}
    
    # Count pairs in each category
    total_pairs = colored_matrix.size
    neighbor_pairs = np.sum(colored_matrix > 0)
    
    thresholds = [1.0, 2.0, 4.0, 8.0, float('inf')]
    counts = []
    
    for i, threshold in enumerate(thresholds):
        if i == 0:
            count = np.sum(colored_matrix == i + 1)
        else:
            count = np.sum(colored_matrix == i + 1)
        counts.append(count)
    
    # Calculate statistics
    neighbor_distances = wild_dist[colored_matrix > 0]
    neighbor_differences = dist_diff[colored_matrix > 0]
    
    stats = {
        'total_residues': colored_matrix.shape[0],
        'total_pairs': total_pairs,
        'neighbor_pairs': neighbor_pairs,
        'neighbor_percentage': (neighbor_pairs / total_pairs) * 100 if total_pairs > 0 else 0,
        'threshold_counts': dict(zip([f'≤{t}Å' if t != float('inf') else '>8Å' for t in thresholds], counts)),
        'avg_neighbor_distance': np.mean(neighbor_distances) if len(neighbor_distances) > 0 else 0,
        'avg_distance_difference': np.mean(neighbor_differences) if len(neighbor_differences) > 0 else 0,
        'max_distance_difference': np.max(neighbor_differences) if len(neighbor_differences) > 0 else 0
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze distance matrix for lDDT calculation")
    parser.add_argument('input_csv', type=str, help='Input CSV file with protein data')
    parser.add_argument('--distance-threshold', type=float, default=15.0,
                        help='Distance threshold for neighbor pairs (default: 15.0 Å)')
    
    args = parser.parse_args()
    
    input_csv = args.input_csv
    output_prefix = input_csv.replace('.csv', '')
    
    print(f"Reading CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Found {len(df)} rows in CSV")
    
    if len(df) == 0:
        print("Error: CSV file is empty")
        return
    
    row = df.iloc[0]  # Always process first row
    protein_name = row.get('PID', 'Protein_0')
    
    print(f"Analyzing protein: {protein_name}")
    
    # Parse coordinates
    wild_coords = parse_coordinates(row.get('coords', []))
    mutant_coords = parse_coordinates(row.get('mutant_coords', []))
    
    if not wild_coords or not mutant_coords:
        print("Error: Missing coordinate data")
        return
    
    print(f"Wild-type coordinates: {len(wild_coords)} residues")
    print(f"Mutant coordinates: {len(mutant_coords)} residues")
    
    # Create distance difference matrix
    colored_matrix, wild_dist, dist_diff = create_distance_difference_matrix(
        wild_coords, mutant_coords, args.distance_threshold
    )
    
    if colored_matrix.size == 0:
        print("Error: Could not create distance matrix")
        return
    
    # Analyze neighbor pairs
    stats = analyze_neighbor_pairs(colored_matrix, wild_dist, dist_diff)
    
    # Print statistics
    print(f"\n=== DISTANCE MATRIX STATISTICS ===")
    print(f"Protein: {protein_name}")
    print(f"Total residues: {stats['total_residues']}")
    print(f"Total residue pairs: {stats['total_pairs']}")
    print(f"Neighbor pairs (≤{args.distance_threshold}Å): {stats['neighbor_pairs']} ({stats['neighbor_percentage']:.1f}%)")
    print(f"Average neighbor distance: {stats['avg_neighbor_distance']:.2f} Å")
    print(f"Average distance difference: {stats['avg_distance_difference']:.2f} Å")
    print(f"Maximum distance difference: {stats['max_distance_difference']:.2f} Å")
    
    print(f"\nDistance difference distribution:")
    for threshold, count in stats['threshold_counts'].items():
        percentage = (count / stats['neighbor_pairs']) * 100 if stats['neighbor_pairs'] > 0 else 0
        print(f"  {threshold}: {count} pairs ({percentage:.1f}%)")
    
    # Create plot
    plot_distance_matrix(colored_matrix, wild_dist, dist_diff, protein_name, 
                        f"{output_prefix}_{protein_name}")
    
    print(f"\nAnalysis complete!")


if __name__ == "__main__":
    main() 