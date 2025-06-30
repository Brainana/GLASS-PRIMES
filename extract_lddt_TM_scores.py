#!/usr/bin/env python3
"""
Extract LDDT scores for protein pairs using provided alignments.
Downloads PDB files from GCS and computes structural comparisons.
UPDATED: Accepts seqxA, seqyA, and seqM as alignment inputs.
UPDATED: Accepts GCS file paths directly instead of protein IDs.
COLAB VERSION: Includes setup for Google Colab environment.
CORRECTED: Implements proper LDDT algorithm with 15Å cutoff and distance difference thresholds.
"""

# Google Cloud authentication for Colab
try:
    from google.colab import auth
    auth.authenticate_user()
    print("✓ Google Cloud authentication completed")
except ImportError:
    print("Not running in Colab - skipping authentication")
    print("Make sure you have proper GCS credentials set up")

import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
from google.cloud import storage
from Bio.PDB import PDBParser
from tqdm import tqdm
import csv
import multiprocessing as mp
from functools import partial

MAX_RESIDUES = 600

# Try to import PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
    print("✓ PyTorch available for GPU acceleration")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not available - using CPU only")


class LDDTExtractor:
    """
    Extract LDDT scores for protein pairs using provided alignments.
    UPDATED: Accepts seqxA, seqyA, and seqM as direct inputs.
    UPDATED: Accepts GCS file paths directly instead of protein IDs.
    CORRECTED: Implements proper LDDT algorithm with 15Å cutoff and distance difference thresholds.
    """
    
    def __init__(self, bucket_name: str, use_gpu: bool = True):
        """
        Initialize the LDDT extractor.
        
        Args:
            bucket_name: GCS bucket name
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.bucket_name = bucket_name
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        
        # Initialize GCS client
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Initialize GPU device if available
        if self.use_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"✓ Using GPU acceleration: {self.device}")
        else:
            self.device = None
            print("Using CPU only")
        
        # Cache for PDB structures to avoid re-reading
        self.structure_cache = {}
        self.max_cache_size = 100  # Limit cache size to prevent memory issues
        
        print(f"Initialized LDDT extractor (UPDATED) for bucket: {bucket_name}")
    
    def read_pdb_from_gcs(self, gcs_path: str) -> str:
        """
        Read PDB file directly from GCS without downloading.
        
        Args:
            gcs_path: GCS path to PDB file (e.g., 'pdb_folder/Q1GF61.pdb')
            
        Returns:
            File-like object for reading the PDB file
        """
        # Get blob and return as file-like object
        blob = self.bucket.blob(gcs_path)
        return blob.open('r')
    
    def get_all_atom_coords_from_structure(self, structure_file) -> List[np.ndarray]:
        """Extract all atom coordinates from structure, grouped by residue."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', structure_file)
        
        residue_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        # Get all atom coordinates for this residue
                        atom_coords = []
                        for atom in residue:
                            atom_coords.append(atom.get_coord())
                        if atom_coords:  # Only add if residue has atoms
                            residue_coords.append(np.array(atom_coords))
                break
        return residue_coords
    
    def get_lddt_scores_with_alignment(self, pdb1_gcs_path: str, pdb2_gcs_path: str, 
                                     seqxA: str, seqyA: str, seqM: str) -> np.ndarray:
        """
        Get LDDT scores using provided alignment sequences.
        Implements proper LDDT algorithm: calculate LDDT for aligned residues only.
        
        Args:
            pdb1_gcs_path: GCS path to template PDB file (reference)
            pdb2_gcs_path: GCS path to target PDB file (being evaluated)
            seqxA: Aligned sequence A (template)
            seqyA: Aligned sequence B (target)
            seqM: Alignment annotation
            
        Returns:
            Array with LDDT scores for aligned residues
        """
        # Get structures from cache or load them
        template_residue_coords = self._get_cached_structure(pdb1_gcs_path)
        target_residue_coords = self._get_cached_structure(pdb2_gcs_path)
        
        # Calculate LDDT scores for aligned residues
        lddt_scores = self._calculate_lddt_with_alignment(
            template_residue_coords, target_residue_coords, seqxA, seqyA, seqM
        )
        
        return lddt_scores
    
    def _calculate_lddt_with_alignment(self, template_residue_coords, target_residue_coords, 
                                     seqxA: str, seqyA: str, seqM: str):
        """
        Calculate LDDT scores for all residues in protein 2 using the alignment information.
        
        Args:
            template_residue_coords: All template residue coordinates
            target_residue_coords: All target residue coordinates
            seqxA: Aligned sequence A (template)
            seqyA: Aligned sequence B (target)
            seqM: Alignment annotation
            
        Returns:
            Array of LDDT scores for all residues in protein 2 (0 for unaligned, real score for aligned)
        """
        # LDDT distance thresholds
        thresholds = [0.5, 1.0, 2.0, 4.0]
        
        # Initialize array for all protein 2 residues with 0 scores
        lddt_scores = np.zeros(MAX_RESIDUES)
        
        # Track residue indices in original sequences
        template_idx = target_idx = 0
        
        for a1, a2, ann in zip(seqxA, seqyA, seqM):
            if a1 != '-' and a2 != '-':
                if ann in [':', '.']:  # Only include aligned residues
                    if template_idx < len(template_residue_coords) and target_idx < len(target_residue_coords):
                        # Calculate LDDT for this aligned residue pair
                        lddt_score = self._calculate_residue_lddt(
                            template_residue_coords, target_residue_coords,
                            template_idx, target_idx,
                            thresholds
                        )
                        # Assign score to the correct position in protein 2 sequence
                        lddt_scores[target_idx] = lddt_score
                
                template_idx += 1
                target_idx += 1
            elif a1 == '-' and a2 != '-':
                # Gap in template, residue in target (unaligned)
                # lddt_scores[target_idx] remains 0.0
                target_idx += 1
            elif a1 != '-' and a2 == '-':
                # Residue in template, gap in target
                template_idx += 1
        
        return lddt_scores
    
    def _calculate_residue_lddt(self, template_residue_coords, target_residue_coords,
                               template_res_idx, target_res_idx, thresholds):
        """
        Calculate LDDT score for a single aligned residue pair.
        
        Proper LDDT algorithm:
        1. Find all atom pairs within 15Å of the current residue in the template structure
        2. For each such pair, calculate |distance_in_target - distance_in_template|
        3. Count how many differences fall within each threshold (0.5, 1, 2, 4Å)
        4. Final score = 1/4 * (count_0.5 + count_1 + count_2 + count_4) / total_pairs
        
        Args:
            template_residue_coords: All template residue coordinates
            target_residue_coords: All target residue coordinates
            template_res_idx: Index of current template residue
            target_res_idx: Index of current target residue
            thresholds: Distance thresholds for LDDT
            
        Returns:
            LDDT score for this residue (between 0 and 1)
        """
        # Get atoms of the current template residue
        if template_res_idx >= len(template_residue_coords):
            return 0.0
        
        current_template_atoms = template_residue_coords[template_res_idx]
        if len(current_template_atoms) == 0:
            return 0.0
        
        # Collect all template atom coordinates with their residue indices
        template_atom_coords = []
        template_atom_to_residue = []  # Maps atom index to residue index
        
        for res_idx, res_atoms in enumerate(template_residue_coords):
            for atom_idx, atom_coord in enumerate(res_atoms):
                template_atom_coords.append(atom_coord)
                template_atom_to_residue.append(res_idx)
        
        if len(template_atom_coords) == 0:
            return 0.0
        
        template_atom_coords = np.array(template_atom_coords)
        
        # Find atom pairs within 15Å of the current residue
        local_atom_pairs = []
        current_atom_start_idx = sum(len(template_residue_coords[i]) for i in range(template_res_idx))
        current_atom_end_idx = current_atom_start_idx + len(current_template_atoms)
        
        # For each atom in the current residue, find all other atoms within 15Å
        for i in range(current_atom_start_idx, current_atom_end_idx):
            for j in range(len(template_atom_coords)):
                if i != j:  # Don't include self-pairs
                    dist = np.linalg.norm(template_atom_coords[i] - template_atom_coords[j])
                    if dist <= 15.0:  # 15Å threshold
                        local_atom_pairs.append((i, j, dist))
        
        if not local_atom_pairs:
            return 0.0
        
        # Collect all target atom coordinates
        target_atom_coords = []
        for res_idx, res_atoms in enumerate(target_residue_coords):
            for atom_idx, atom_coord in enumerate(res_atoms):
                target_atom_coords.append(atom_coord)
        
        if len(target_atom_coords) == 0:
            return 0.0
        
        target_atom_coords = np.array(target_atom_coords)
        
        # Calculate distance differences for each local atom pair
        distance_differences = []
        for i, j, template_dist in local_atom_pairs:
            if i < len(target_atom_coords) and j < len(target_atom_coords):
                # Get corresponding target coordinates
                target_coord1 = target_atom_coords[i]
                target_coord2 = target_atom_coords[j]
                target_dist = np.linalg.norm(target_coord1 - target_coord2)
                
                # Calculate absolute difference
                diff = abs(target_dist - template_dist)
                distance_differences.append(diff)
        
        if not distance_differences:
            return 0.0
        
        # Count differences within each threshold
        counts = []
        for threshold in thresholds:
            count = sum(1 for diff in distance_differences if diff <= threshold)
            counts.append(count)
        
        # Calculate final LDDT score
        total_pairs = len(distance_differences)
        if total_pairs == 0:
            return 0.0
        
        # Final score = 1/4 * (sum of normalized counts)
        lddt_score = sum(count / total_pairs for count in counts) / len(thresholds)
        
        return lddt_score
    
    def process_protein_pair(self, pdb1_gcs_path: str, pdb2_gcs_path: str, 
                           seqxA: str, seqyA: str, seqM: str, tm_score: float = None) -> Optional[Dict]:
        """
        Process a single protein pair to extract LDDT scores.
        
        Args:
            pdb1_gcs_path: GCS path to first PDB file (template)
            pdb2_gcs_path: GCS path to second PDB file (target)
            seqxA: Aligned sequence A
            seqyA: Aligned sequence B
            seqM: Alignment annotation
            tm_score: Optional TM score
            
        Returns:
            Dictionary with LDDT scores, or None if error
        """
        try:
            # Get LDDT scores for all residues in protein 2 using provided alignment
            lddt_scores = self.get_lddt_scores_with_alignment(pdb1_gcs_path, pdb2_gcs_path, seqxA, seqyA, seqM)
            
            # Convert LDDT scores to string format for CSV
            lddt_scores_str = json.dumps(lddt_scores.tolist())
            
            # Extract protein IDs from GCS paths for output
            protein_id1 = os.path.splitext(os.path.basename(pdb1_gcs_path))[0]
            protein_id2 = os.path.splitext(os.path.basename(pdb2_gcs_path))[0]
            
            # Count aligned residues (non-zero scores)
            aligned_residues = lddt_scores[lddt_scores > 0]
            num_aligned = len(aligned_residues)
            
            # Calculate average LDDT only for aligned residues
            avg_lddt = float(np.mean(aligned_residues)) if num_aligned > 0 else 0.0
            
            return {
                'protein_id1': protein_id1,
                'protein_id2': protein_id2,
                'pdb1_gcs_path': pdb1_gcs_path,
                'pdb2_gcs_path': pdb2_gcs_path,
                'lddt_scores_protein2': lddt_scores_str,
                'seqxA': seqxA,
                'seqyA': seqyA,
                'seqM': seqM,
                'tm_score': tm_score,
                'num_aligned_residues': num_aligned,
                'avg_lddt_protein2': avg_lddt
            }
            
        except Exception as e:
            print(f"Error processing {pdb1_gcs_path} vs {pdb2_gcs_path}: {e}")
            return None
    
    def cleanup_temp_files(self):
        """
        Clean up temporary PDB files.
        """
        # This method is now empty as we're not downloading files
        pass
    
    def __del__(self):
        """
        Cleanup on deletion.
        """
        self.cleanup_temp_files()
    
    def _calculate_distances_gpu(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise distances between two sets of coordinates using GPU.
        
        Args:
            coords1: First set of coordinates [N, 3]
            coords2: Second set of coordinates [M, 3]
            
        Returns:
            Distance matrix [N, M]
        """
        if not self.use_gpu:
            # Fallback to CPU calculation
            return self._calculate_distances_cpu(coords1, coords2)
        
        # Convert to PyTorch tensors
        coords1_tensor = torch.FloatTensor(coords1).to(self.device)
        coords2_tensor = torch.FloatTensor(coords2).to(self.device)
        
        # Calculate distances using broadcasting
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        dist_sq = torch.sum(coords1_tensor**2, dim=1, keepdim=True) + \
                  torch.sum(coords2_tensor**2, dim=1) - \
                  2 * torch.mm(coords1_tensor, coords2_tensor.t())
        
        # Take square root and return as numpy
        distances = torch.sqrt(torch.clamp(dist_sq, min=0)).cpu().numpy()
        return distances
    
    def _calculate_distances_cpu(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise distances between two sets of coordinates using CPU.
        
        Args:
            coords1: First set of coordinates [N, 3]
            coords2: Second set of coordinates [M, 3]
            
        Returns:
            Distance matrix [N, M]
        """
        # Use scipy for efficient distance calculation
        from scipy.spatial.distance import cdist
        return cdist(coords1, coords2)
    
    def _find_pairs_within_threshold_gpu(self, coords: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
        """
        Find all atom pairs within a distance threshold using GPU acceleration.
        
        Args:
            coords: Atom coordinates [N, 3]
            threshold: Distance threshold
            
        Returns:
            List of (i, j, distance) tuples for pairs within threshold
        """
        if not self.use_gpu:
            return self._find_pairs_within_threshold_cpu(coords, threshold)
        
        # Convert to PyTorch tensor
        coords_tensor = torch.FloatTensor(coords).to(self.device)
        
        # Calculate all pairwise distances
        dist_matrix = self._calculate_distances_gpu(coords, coords)
        
        # Find pairs within threshold (upper triangle only)
        pairs = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = dist_matrix[i, j]
                if dist <= threshold:
                    pairs.append((i, j, dist))
        
        return pairs
    
    def _find_pairs_within_threshold_cpu(self, coords: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
        """
        Find all atom pairs within a distance threshold using CPU.
        
        Args:
            coords: Atom coordinates [N, 3]
            threshold: Distance threshold
            
        Returns:
            List of (i, j, distance) tuples for pairs within threshold
        """
        pairs = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= threshold:
                    pairs.append((i, j, dist))
        return pairs
    
    def _get_cached_structure(self, gcs_path: str):
        """
        Get PDB structure from cache or load it.
        
        Args:
            gcs_path: GCS path to PDB file
            
        Returns:
            List of residue coordinates
        """
        if gcs_path in self.structure_cache:
            return self.structure_cache[gcs_path]
        
        # Load structure
        pdb_file = self.read_pdb_from_gcs(gcs_path)
        structure_coords = self.get_all_atom_coords_from_structure(pdb_file)
        pdb_file.close()
        
        # Cache the result
        if len(self.structure_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.structure_cache))
            del self.structure_cache[oldest_key]
        
        self.structure_cache[gcs_path] = structure_coords
        return structure_coords


def load_protein_pairs_with_alignments_batch(csv_file, pdb_folder: str, batch_size: int = 1000) -> List[Tuple[str, str, str, str, str, float]]:
    """
    Load protein pairs with alignments from CSV file in batches.
    
    Args:
        csv_file: File-like object for CSV file with protein pairs and alignments
        pdb_folder: GCS folder containing PDB files
        batch_size: Number of rows to process at once
        
    Yields:
        List of (pdb1_gcs_path, pdb2_gcs_path, seqxA, seqyA, seqM, tm_score) tuples
    """
    
    # Read CSV in chunks
    chunk_iter = pd.read_csv(csv_file, chunksize=batch_size)
    
    for chunk in chunk_iter:
        # Extract columns from this chunk
        protein_pairs = []
        for _, row in chunk.iterrows():
            protein_id1 = str(row['chain_1']).strip()
            protein_id2 = str(row['chain_2']).strip()
            seqxA = str(row['seqxA']).strip()
            seqyA = str(row['seqyA']).strip()
            seqM = str(row['seqM']).strip()
            tm_score = float(row.get('computed_tm', 0.0)) if 'computed_tm' in row else None
            
            # Construct full GCS paths
            pdb1_gcs_path = f"{pdb_folder}/{protein_id1}.pdb"
            pdb2_gcs_path = f"{pdb_folder}/{protein_id2}.pdb"
            
            protein_pairs.append((pdb1_gcs_path, pdb2_gcs_path, seqxA, seqyA, seqM, tm_score))
        
        yield protein_pairs


def save_results(results: List[Dict], output_path: str):
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file
    """
    if not results:
        print("No results to save!")
        return
    
    # Define CSV columns (removed PDB paths)
    fieldnames = [
        'protein_id1', 'protein_id2', 
        'lddt_scores_protein2', 'seqxA', 'seqyA', 'seqM',
        'tm_score', 'num_aligned_residues', 'avg_lddt_protein2'
    ]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Create a copy without PDB paths
            clean_result = {k: v for k, v in result.items() if k in fieldnames}
            writer.writerow(clean_result)
    
    print(f"Results saved to: {output_path}")
    print(f"Processed {len(results)} protein pairs successfully")


def _process_single_pair_standalone(protein_pair: Tuple, bucket_name: str, use_gpu: bool = True) -> Optional[Dict]:
    """
    Process a single protein pair (standalone function for multiprocessing).
    
    Args:
        protein_pair: (pdb1_gcs_path, pdb2_gcs_path, seqxA, seqyA, seqM, tm_score) tuple
        bucket_name: GCS bucket name
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Result dictionary or None if failed
    """
    pdb1_gcs_path, pdb2_gcs_path, seqxA, seqyA, seqM, tm_score = protein_pair
    
    try:
        # Create a temporary extractor for this single pair
        temp_extractor = LDDTExtractor(bucket_name, use_gpu=use_gpu)
        return temp_extractor.process_protein_pair(pdb1_gcs_path, pdb2_gcs_path, seqxA, seqyA, seqM, tm_score)
    except Exception as e:
        print(f"Error processing {pdb1_gcs_path} vs {pdb2_gcs_path}: {e}")
        return None

def process_protein_pairs_parallel_standalone(protein_pairs: List[Tuple], bucket_name: str, 
                                            num_processes: int = None, use_gpu: bool = True) -> List[Dict]:
    """
    Process protein pairs in parallel using multiprocessing (standalone version).
    
    Args:
        protein_pairs: List of (pdb1_gcs_path, pdb2_gcs_path, seqxA, seqyA, seqM, tm_score) tuples
        bucket_name: GCS bucket name
        num_processes: Number of processes to use (default: CPU count)
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        List of result dictionaries
    """
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(protein_pairs))
    
    print(f"Processing {len(protein_pairs)} protein pairs using {num_processes} processes...")
    
    # Create a partial function with the bucket name and GPU setting
    process_func = partial(_process_single_pair_standalone, bucket_name=bucket_name, use_gpu=use_gpu)
    
    # Process in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, protein_pairs),
            total=len(protein_pairs),
            desc="Processing protein pairs"
        ))
    
    # Filter out None results (failed pairs)
    valid_results = [r for r in results if r is not None]
    failed_count = len(results) - len(valid_results)
    
    print(f"Completed: {len(valid_results)} successful, {failed_count} failed")
    return valid_results


def main():
    """
    Main function to extract LDDT scores using provided alignments.
    
    COLAB USAGE:
    1. Make sure your CSV file is in the same GCS bucket as your PDB files
    2. Modify the configuration variables below
    3. Run this script
    """
    # Configuration - modify these variables as needed
    input_csv = "SWISS_MODEL/tm_score_comparison_results.csv"  # Input CSV file with protein pairs and alignments
    bucket_name = "jx-compbio"  # GCS bucket name
    pdb_folder = "SWISS_MODEL/pdbs"  # GCS folder containing PDB files
    output_csv = "results.csv"  # Output CSV file for results
    max_pairs = 10  # Maximum number of pairs to process from CSV (None for all)
    batch_size = 1000  # Number of pairs to process in each batch
    use_gpu = True  # Whether to use GPU acceleration
    num_processes = None  # Number of processes for parallel processing (None = auto)
    
    # Colab-specific file handling
    try:
        from google.colab import files
        print("Running in Colab - files will be saved to Colab environment")
        print("Use files.download() to download results after processing")
    except ImportError:
        print("Not running in Colab - files will be saved to local filesystem")
    
    # Initialize extractor with GPU support
    extractor = LDDTExtractor(bucket_name, use_gpu=use_gpu)
    
    # Read CSV file directly from GCS
    print(f"Reading CSV file from GCS: {input_csv}")
    try:
        csv_file = extractor.read_pdb_from_gcs(input_csv)
        print(f"CSV file opened successfully")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print(f"Make sure the file '{input_csv}' exists in bucket '{bucket_name}'")
        return
    
    # Process protein pairs in batches
    results = []
    failed_pairs = []
    processed_count = 0
    
    print(f"Processing protein pairs in batches of {batch_size}...")
    
    # Process batches
    for batch_idx, protein_pairs_batch in enumerate(load_protein_pairs_with_alignments_batch(csv_file, pdb_folder, batch_size)):
        print(f"Processing batch {batch_idx + 1} ({len(protein_pairs_batch)} pairs)...")
        
        # Limit batch size if max_pairs is set
        if max_pairs and processed_count + len(protein_pairs_batch) > max_pairs:
            protein_pairs_batch = protein_pairs_batch[:max_pairs - processed_count]
        
        # Process batch in parallel
        batch_results = process_protein_pairs_parallel_standalone(protein_pairs_batch, bucket_name, num_processes, use_gpu)
        results.extend(batch_results)
        
        processed_count += len(protein_pairs_batch)
        
        # Save intermediate results after each batch
        temp_output = output_csv.replace('.csv', f'_temp_batch_{batch_idx + 1}.csv')
        save_results(results, temp_output)
        print(f"Batch {batch_idx + 1} completed. Intermediate results saved: {temp_output}")
        print(f"Processed so far: {processed_count} pairs, Successful: {len(results)}")
        
        # Check if we've reached max_pairs limit
        if max_pairs and processed_count >= max_pairs:
            break
    
    # Close CSV file
    csv_file.close()
    
    # Save final results
    save_results(results, output_csv)
    
    # Report statistics
    print(f"\nProcessing completed!")
    print(f"Total pairs processed: {processed_count}")
    print(f"Successful pairs: {len(results)}")
    print(f"Failed pairs: {processed_count - len(results)}")
    
    print(f"\nResults saved to: {output_csv}")
    print("UPDATED: This version uses parallel processing and GPU acceleration!")
    
    # Colab-specific download instructions
    try:
        from google.colab import files
        print("\nTo download results from Colab, run:")
        print("files.download('results.csv')")
        print("Or for intermediate results:")
        print("files.download('results_temp_batch_1.csv')")
    except ImportError:
        print("\nResults saved to local filesystem")


if __name__ == "__main__":
    main() 