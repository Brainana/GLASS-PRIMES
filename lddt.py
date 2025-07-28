import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.distance import cdist

class LDDTCalculator:
    """
    Local Distance Difference Test (lDDT) calculator using Cα coordinates.
    
    Implementation based on the algorithm described in:
    Mariani, V., et al. "lDDT: a local superposition-free score for comparing 
    protein structures and models using distance difference tests." 
    Bioinformatics 29.21 (2013): 2722-2728.
    
    This implementation uses Cα coordinates instead of all atom coordinates
    as described in the original paper.
    """
    
    def __init__(self, distance_thresholds: List[float] = None, 
                 max_distance: float = 15.0):
        """
        Initialize lDDT calculator.
        
        Args:
            distance_thresholds: List of distance thresholds for scoring
                                Default: [0.5, 1.0, 2.0, 4.0] as in paper
            max_distance: Maximum distance to consider for neighbor atoms
        """
        if distance_thresholds is None:
            self.distance_thresholds = [0.5, 1.0, 2.0, 4.0]
        else:
            self.distance_thresholds = distance_thresholds
        
        self.max_distance = max_distance
        self.num_thresholds = len(self.distance_thresholds)
    
    def calculate_lddt(self, reference_coords: np.ndarray, model_coords: np.ndarray) -> np.ndarray:
        """
        Calculate lDDT score for a protein model against reference structure.
        Assumes reference_coords and model_coords are in one-to-one correspondence.
        
        Args:
            reference_coords: Cα coordinates of reference [N, 3]
            model_coords: Cα coordinates of model [N, 3]
        
        Returns:
            per_residue_scores: lDDT scores for each residue [N]
        """
        if reference_coords.shape != model_coords.shape:
            raise ValueError("Reference and model coordinates must have the same shape for one-to-one correspondence.")
        N = reference_coords.shape[0]
        # Calculate distance matrices for all residues
        reference_distances = self._calculate_distance_matrix(reference_coords)
        model_distances = self._calculate_distance_matrix(model_coords)
        # Calculate lDDT scores for each residue
        per_residue_scores = np.zeros(N)
        for i in range(N):
            per_residue_scores[i] = self._calculate_residue_lddt(
                i, reference_distances, model_distances
            )
        return per_residue_scores

    def _calculate_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise distance matrix for Cα atoms using scipy's cdist.
        Args:
            coords: Cα coordinates [N, 3]
        Returns:
            Distance matrix [N, N]
        """
        distances = cdist(coords, coords, metric='euclidean')
        return distances
    
    def _calculate_residue_lddt(self, residue_idx: int, 
                               reference_distances: np.ndarray,
                               model_distances: np.ndarray) -> float:
        """
        Calculate lDDT score for a single residue.
        
        Args:
            residue_idx: Index of the residue to score
            reference_distances: Distance matrix for reference [N, N]
            model_distances: Distance matrix for model [N, N]
        
        Returns:
            lDDT score for the residue (0-1)
        """
        ref_distances_residue = reference_distances[residue_idx]
        neighbors = np.where((ref_distances_residue > 0) & 
                           (ref_distances_residue <= self.max_distance))[0]
        
        if len(neighbors) == 0:
            return 0  # No neighbors to compare

        # Calculate distance differences
        distance_diffs = []
        for neighbor in neighbors:
            model_dist = model_distances[residue_idx, neighbor]
            ref_dist = reference_distances[residue_idx, neighbor]
            
            if ref_dist > 0:  # Valid distance in reference
                diff = abs(model_dist - ref_dist)
                distance_diffs.append(diff)
        
        if len(distance_diffs) == 0:
            return 0 # No valid comparisons
        
        # Calculate scores for each threshold
        threshold_scores = []
        for threshold in self.distance_thresholds:
            # Count distances within threshold
            within_threshold = sum(1 for diff in distance_diffs if diff <= threshold)
            score = within_threshold / len(distance_diffs)
            threshold_scores.append(score)
        
        # Average score across all thresholds
        residue_score = np.mean(threshold_scores)
        
        return residue_score
    
    def calculate_lddt_batch(self, model_coords_list: List[np.ndarray], reference_coords_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Calculate lDDT scores for multiple protein pairs.
        Assumes each pair of model_coords and reference_coords are in one-to-one correspondence.
        
        Args:
            model_coords_list: List of model coordinates
            reference_coords_list: List of reference coordinates
        
        Returns:
            per_residue_scores_list: List of per-residue scores
        """
        per_residue_scores_list = []
        for model_coords, ref_coords in zip(model_coords_list, reference_coords_list):
            per_residue_scores = self.calculate_lddt(model_coords, ref_coords)
            per_residue_scores_list.append(per_residue_scores)
        return per_residue_scores_list

def example_usage():
    """
    Example usage of the lDDT calculator assuming one-to-one correspondence.
    """
    np.random.seed(42)
    N = 100
    reference_coords = np.random.rand(N, 3) * 20  # Random coordinates
    model_coords = reference_coords + np.random.normal(0, 0.5, (N, 3))  # Add noise
    calculator = LDDTCalculator()
    per_residue_scores = calculator.calculate_lddt(model_coords, reference_coords)
    print(f"Number of residues: {N}")
    print(f"Mean per-residue score: {np.mean(per_residue_scores):.4f}")
    print(f"Score range: {np.min(per_residue_scores):.4f} - {np.max(per_residue_scores):.4f}")


if __name__ == "__main__":
    example_usage() 