import numpy as np
from lddt import LDDTCalculator
import random

def create_test_protein_coords(length=300, base_coords=None):
    """
    Create protein coordinates for testing.
    
    Args:
        length: Length of the protein
        base_coords: Base coordinates to modify (if None, creates random coords)
    
    Returns:
        coordinates: Protein coordinates [length, 3]
    """
    if base_coords is None:
        # Create random coordinates in a reasonable protein-like structure
        coords = np.random.rand(length, 3) * 20  # Random coordinates in 20Å cube
        # Add some structure by making consecutive residues closer
        for i in range(1, length):
            coords[i] = coords[i-1] + np.random.normal(0, 3.8, 3)  # ~3.8Å typical Cα-Cα distance
        return coords
    else:
        return base_coords.copy()

def create_point_mutation(coords, position, magnitude=1.0):
    """
    Create a point mutation by perturbing coordinates at a specific position.
    
    Args:
        coords: Original coordinates
        position: Position to mutate (0-indexed)
        magnitude: Magnitude of the mutation in Å
    
    Returns:
        mutated_coords: Coordinates with mutation
    """
    mutated_coords = coords.copy()
    # Add random perturbation
    mutation = np.random.normal(0, magnitude, 3)
    mutated_coords[position] += mutation
    return mutated_coords

def create_multiple_mutations(coords, positions, magnitudes=None):
    """
    Create multiple point mutations.
    
    Args:
        coords: Original coordinates
        positions: List of positions to mutate
        magnitudes: List of mutation magnitudes (if None, uses 1.0 for all)
    
    Returns:
        mutated_coords: Coordinates with mutations
    """
    if magnitudes is None:
        magnitudes = [1.0] * len(positions)
    
    mutated_coords = coords.copy()
    for pos, mag in zip(positions, magnitudes):
        mutation = np.random.normal(0, mag, 3)
        mutated_coords[pos] += mutation
    return mutated_coords

def test_lddt_first_10_residues():
    """
    Test lDDT scores comparing only the first 10 residues with different test cases.
    """
    print("=== lDDT Test Cases (First 10 Residues Only) ===\n")
    
    calculator = LDDTCalculator(max_distance=10.0)
    
    # Create base protein structure
    base_coords = create_test_protein_coords(300)
    
    # Test Case 1: Exact Match
    print("--- Test Case 1: Exact Structure Match ---")
    exact_match_coords = base_coords.copy()
    exact_lddt = calculator.calculate_lddt(exact_match_coords[:10], base_coords[:10])
    print(f"lDDT Scores: {exact_lddt}")
    
    # Test Case 2: Single Point Mutation
    print("\n--- Test Case 2: Single Point Mutation (Position 5) ---")
    single_mutation_coords = create_point_mutation(base_coords, position=5, magnitude=1.0)
    point_lddt = calculator.calculate_lddt(single_mutation_coords[:10], base_coords[:10])
    print(f"lDDT Scores: {point_lddt}")
    
    # Test Case 3: Multiple Point Mutations
    print("\n--- Test Case 3: Multiple Point Mutations ---")
    multiple_coords = base_coords.copy()
    for pos in [2, 7, 9]:
        multiple_coords = create_point_mutation(multiple_coords, pos, magnitude=1.0)
    multiple_point_lddt = calculator.calculate_lddt(multiple_coords[:10], base_coords[:10])
    print(f"lDDT Scores: {multiple_point_lddt}")
    
    # Test Case 4: Large Structural Change
    print("\n--- Test Case 4: Large Structural Change ---")
    large_change_coords = create_point_mutation(base_coords, position=4, magnitude=5.0)
    large_lddt = calculator.calculate_lddt(large_change_coords[:10], base_coords[:10])
    print(f"lDDT Scores: {large_lddt}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Exact Match: {exact_lddt}")
    print(f"Single Mutation: {point_lddt}")
    print(f"Multiple Mutations: {multiple_point_lddt}")
    print(f"Large Change: {large_lddt}")

def test_lddt_with_different_sequence_lengths():
    """
    Test lDDT scores with different sequence lengths (all using first 10 residues).
    """
    print("\n=== Testing Different Sequence Lengths (First 10 Residues) ===\n")
    
    calculator = LDDTCalculator(max_distance=10.0)
    
    # Test with different total sequence lengths
    lengths = [50, 100, 200, 300]
    
    for length in lengths:
        print(f"\n--- Sequence Length: {length} ---")
        
        # Create base coordinates
        base_coords = create_test_protein_coords(length)
        
        # Create mutation
        mutated_coords = create_point_mutation(base_coords, position=5, magnitude=1.0)
        
        # Calculate lDDT for first 10 residues only
        per_residue = calculator.calculate_lddt(mutated_coords[:10], base_coords[:10])
        score = np.mean(per_residue)
        
        print(f"lDDT Score: {score:.4f}")
        print(f"Per-residue scores (first 10): {per_residue[:10]}")

def test_lddt_threshold_sensitivity():
    """
    Test how sensitive lDDT is to different distance thresholds.
    """
    print("\n=== Testing Distance Threshold Sensitivity ===\n")
    
    # Create test coordinates
    base_coords = create_test_protein_coords(300)
    mutated_coords = create_point_mutation(base_coords, position=5, magnitude=2.0)
    
    # Test different distance thresholds
    thresholds = [5.0, 10.0, 15.0, 20.0]
    
    for threshold in thresholds:
        calculator = LDDTCalculator(max_distance=threshold)
        per_residue = calculator.calculate_lddt(mutated_coords[:10], base_coords[:10])
        score = np.mean(per_residue)
        print(f"Max distance {threshold}Å: lDDT = {score:.4f}")

if __name__ == "__main__":
    # Run all tests
    test_lddt_first_10_residues()
    test_lddt_with_different_sequence_lengths()
    test_lddt_threshold_sensitivity()
    
    print("\n=== All Tests Completed ===") 