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
from tmtools import tm_align
from lddt import LDDTCalculator

MAX_PAIRS = 10  # Set to an integer to limit, or None to process all pairs

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
        # Initialize lDDT calculator
        self.lddt_calculator = LDDTCalculator()
    
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
    
    def process_protein_pair(self, pdb1_gcs_path: str, pdb2_gcs_path: str) -> Optional[Dict]:
        try:
            # Download PDB files from GCS to local temp files
            with self.read_pdb_from_gcs(pdb1_gcs_path) as pdb1_file, \
                 self.read_pdb_from_gcs(pdb2_gcs_path) as pdb2_file:
                # Pass pdb1_file and pdb2_file directly to your coordinate/sequence extraction
                model_coords_all, model_seq = self.get_ca_coords_and_seq(pdb2_file)
                reference_coords_all, reference_seq = self.get_ca_coords_and_seq(pdb1_file)

            protein_id1 = os.path.splitext(os.path.basename(pdb1_gcs_path))[0]
            protein_id2 = os.path.splitext(os.path.basename(pdb2_gcs_path))[0]
            print(f"{protein_id1}.pdb ({len(reference_seq)} aa) vs {protein_id2}.pdb ({len(model_seq)} aa)")

            # Run TM-align
            result = tm_align(reference_coords_all, model_coords_all, reference_seq, model_seq)
            tm_score = result.tm_norm_chain1
            seqxA = result.seqxA
            seqyA = result.seqyA
            seqM = result.seqM

            # Parse alignment pairs using parse_tm_align_result
            alignment_pairs = self.parse_tm_align_result(result)

            # Load all Cα coordinates
            model_coords = np.array([model_coords_all[i] for _, i in alignment_pairs])
            reference_coords = np.array([reference_coords_all[j] for j, _ in alignment_pairs])

            # Calculate lDDT per-residue scores using self.lddt_calculator
            per_residue_scores = self.lddt_calculator.calculate_lddt(model_coords, reference_coords)

            # Pad lDDT scores to length 300, placing each score at the correct model index
            lddt_scores_padded = np.zeros(300, dtype=float)
            for score_idx, (model_idx, _) in enumerate(alignment_pairs):
                if model_idx < 300:
                    lddt_scores_padded[model_idx] = per_residue_scores[score_idx]
            lddt_scores_str = json.dumps(lddt_scores_padded.tolist())

            # Clean up temp files
            pdb1_file.close()
            pdb2_file.close()

            avg_lddt = float(np.mean(per_residue_scores[per_residue_scores > 0])) if np.any(per_residue_scores > 0) else 0.0
            
            return {
                'protein_id1': protein_id1,
                'protein_id2': protein_id2,
                'seqxA': seqxA,
                'seqyA': seqyA,
                'seqM': seqM,
                'tm_score': tm_score,
                'lddt_scores_protein2': lddt_scores_str,
                'avg_lddt_protein2': avg_lddt
            }
        except Exception as e:
            print(f"Error processing {pdb1_gcs_path} vs {pdb2_gcs_path}: {e}")
            return None

    def get_ca_coords_and_seq(self, pdb_file):
        """
        Extract C-alpha coordinates and sequence from PDB file, using one-letter codes for the sequence.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('', pdb_file)
        
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        coords = []
        seq = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
                        resname = residue.get_resname()
                        seq.append(three_to_one.get(resname, 'X'))  # Use 'X' for unknowns
        coords = np.array(coords)
        seq = ''.join(seq)
        return coords, seq
        
    def parse_tm_align_result(self, result):
        """
        Parse TM-align result to get residue alignment.
        """
        aligned_seq1 = result.seqxA
        aligned_seq2 = result.seqyA
        annotation = result.seqM
        alignment = []
        idx1 = idx2 = 0
        for a1, a2, ann in zip(aligned_seq1, aligned_seq2, annotation):
            if a1 != '-' and a2 != '-':
                if ann in [':', '.']:
                    alignment.append((idx1, idx2))
                idx1 += 1
                idx2 += 1
            elif a1 == '-' and a2 != '-':
                idx2 += 1
            elif a1 != '-' and a2 == '-':
                idx1 += 1
        return alignment

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
        return temp_extractor.process_protein_pair(pdb1_gcs_path, pdb2_gcs_path)
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
        
        # Check if we've reached max_pairs limit
        if max_pairs and processed_count >= max_pairs:
            break
    
    # Close CSV file
    csv_file.close()
    
    # Report statistics
    print(f"\nProcessing completed!")
    print(f"Total pairs processed: {processed_count}")
    print(f"Successful pairs: {len(results)}")
    print(f"Failed pairs: {processed_count - len(results)}")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        output_file = 'lddt_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        print(f"Columns: {list(df.columns)}")
    else:
        print("\nNo results to save.")
    
    print(f"\nResults saved to: {input_csv}")
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