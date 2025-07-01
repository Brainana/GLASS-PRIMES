import numpy as np
from typing import List, Tuple, Optional
import warnings

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
    
    def calculate_lddt(self, model_coords: np.ndarray, 
                      reference_coords: np.ndarray,
                      alignment: List[Tuple[int, int]]) -> np.ndarray:
        """
        Calculate lDDT score for a protein model against reference structure.
        
        Args:
            model_coords: Cα coordinates of model [N_model, 3]
            reference_coords: Cα coordinates of reference [N_ref, 3]
            alignment: List of tuples (model_idx, ref_idx) representing aligned residues
        
        Returns:
            per_residue_scores: lDDT scores for each aligned residue [len(alignment)]
        """
        # Validate inputs
        if not alignment:
            raise ValueError("Alignment list cannot be empty")
        
        # Extract aligned coordinates
        aligned_model_coords = np.array([model_coords[i] for i, _ in alignment])
        aligned_ref_coords = np.array([reference_coords[j] for _, j in alignment])
        
        N_aligned = len(alignment)
        
        # Calculate distance matrices for aligned residues
        model_distances = self._calculate_distance_matrix(aligned_model_coords)
        reference_distances = self._calculate_distance_matrix(aligned_ref_coords)
        
        # Calculate lDDT scores for each aligned residue
        per_residue_scores = np.zeros(N_aligned)
        
        for i in range(N_aligned):
            per_residue_scores[i] = self._calculate_residue_lddt(
                i, model_distances, reference_distances
            )
        
        return per_residue_scores
    
    def _calculate_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise distance matrix for Cα atoms.
        
        Args:
            coords: Cα coordinates [N, 3]
        
        Returns:
            Distance matrix [N, N]
        """
        N = coords.shape[0]
        distances = np.zeros((N, N))
        
        # Calculate pairwise distances
        for i in range(N):
            for j in range(N):
                if i != j:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    distances[i, j] = dist
        
        return distances
    
    def _calculate_residue_lddt(self, residue_idx: int, 
                               model_distances: np.ndarray,
                               reference_distances: np.ndarray) -> float:
        """
        Calculate lDDT score for a single residue.
        
        Args:
            residue_idx: Index of the residue to score
            model_distances: Distance matrix for model [N, N]
            reference_distances: Distance matrix for reference [N, N]
        
        Returns:
            lDDT score for the residue (0-1)
        """
        # Find neighbors within max_distance in reference structure
        ref_distances = reference_distances[residue_idx]
        neighbors = np.where((ref_distances > 0) & 
                           (ref_distances <= self.max_distance))[0]
        
        if len(neighbors) == 0:
            return 0  # No neighbors to compare
        
        # Calculate distance differences
        distance_diffs = []
        for neighbor in neighbors:
            model_dist = model_distances[residue_idx, neighbor]
            ref_dist = reference_distances[residue_idx, neighbor]
            
            if model_dist > 0:  # Valid distance in model
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
    
    def calculate_lddt_batch(self, model_coords_list: List[np.ndarray],
                           reference_coords_list: List[np.ndarray],
                           alignments: List[List[Tuple[int, int]]]) -> List[np.ndarray]:
        """
        Calculate lDDT scores for multiple protein pairs.
        
        Args:
            model_coords_list: List of model coordinates
            reference_coords_list: List of reference coordinates
            alignments: List of alignments, where each alignment is a list of (model_idx, ref_idx) tuples
        
        Returns:
            per_residue_scores_list: List of per-residue scores
        """
        per_residue_scores_list = []
        
        for model_coords, ref_coords, alignment in zip(model_coords_list, reference_coords_list, alignments):
            per_residue_scores = self.calculate_lddt(
                model_coords, ref_coords, alignment
            )
            per_residue_scores_list.append(per_residue_scores)
        
        return per_residue_scores_list


def load_pdb_coordinates(pdb_file: str, chain_id: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Cα coordinates from PDB file.
    
    Args:
        pdb_file: Path to PDB file
        chain_id: Chain ID to extract (if None, uses first chain)
    
    Returns:
        coordinates: Cα coordinates [N, 3]
        residue_ids: Residue IDs [N]
    """
    try:
        import Bio.PDB
    except ImportError:
        raise ImportError("Biopython is required. Install with: pip install biopython")
    
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    coordinates = []
    residue_ids = []
    
    for model in structure:
        for chain in model:
            if chain_id is None or chain.id == chain_id:
                for residue in chain:
                    if residue.has_id('CA'):
                        ca_atom = residue['CA']
                        coordinates.append(ca_atom.get_coord())
                        residue_ids.append(residue.get_id()[1])  # Residue number
                break  # Only process first matching chain
    
    if not coordinates:
        raise ValueError(f"No Cα atoms found in PDB file {pdb_file}")
    
    return np.array(coordinates), np.array(residue_ids)

def example_usage():
    """
    Example usage of the lDDT calculator with alignment.
    """
    # Create sample coordinates (replace with actual PDB loading)
    np.random.seed(42)
    
    # Generate sample protein coordinates
    N_model = 100
    N_ref = 95  # Different number of residues
    reference_coords = np.random.rand(N_ref, 3) * 20  # Random coordinates
    model_coords = np.random.rand(N_model, 3) * 20  # Random coordinates
    
    # Create alignment (assuming first 90 residues are aligned)
    alignment = [(i, i) for i in range(90)]
    
    # Initialize calculator
    calculator = LDDTCalculator()
    
    # Calculate lDDT
    per_residue_scores = calculator.calculate_lddt(
        model_coords, reference_coords, alignment
    )
    
    print(f"Number of aligned residues: {len(alignment)}")
    print(f"Mean per-residue score: {np.mean(per_residue_scores):.4f}")
    print(f"Score range: {np.min(per_residue_scores):.4f} - {np.max(per_residue_scores):.4f}")


if __name__ == "__main__":
    example_usage() 