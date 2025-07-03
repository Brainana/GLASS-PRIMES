#!/usr/bin/env python3

# Google Cloud authentication for Colab
try:
    from google.colab import auth
    auth.authenticate_user()
except ImportError:
    print("Not running in Colab - skipping authentication")
    print("Make sure you have proper GCS credentials set up")

import multiprocessing as mp
import pandas as pd
import numpy as np
import os
import json
import pickle
import hashlib
import tempfile
import fcntl
from typing import List, Dict, Tuple, Optional
from google.cloud import storage
from Bio.PDB import PDBParser
from tqdm import tqdm
import csv
from functools import partial
from tmtools import tm_align
from lddt import LDDTCalculator
from google.cloud import bigquery
import time
import base64

START_LINE = 200  # Line to start from
max_pairs = 1000  # Maximum number of pairs to process from CSV (None for all)
client = bigquery.Client(project="mit-primes-464001") # BigQuery client
bucket_name = "jx-compbio"  # GCS bucket name
batch_size = 1000  # Number of pairs to process in each batch
use_gpu = True  # Whether to use GPU acceleration
num_processes = None  # Number of processes for parallel processing (None = auto)


# Try to import PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
    import torch.multiprocessing as torch_mp
except ImportError:
    TORCH_AVAILABLE = False

PDB_INFO_TABLE = "mit-primes-464001.primes_data.pdb_info"

# Global GCS client and bucket
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

# Helper to fetch PDB info from BigQuery
def fetch_pdb_info_batch(protein_ids, bq_client=None, table_id=PDB_INFO_TABLE):
    # Use IN clause for better performance
    query = f"SELECT id, coords, seq FROM `{table_id}` WHERE id IN ({protein_ids})"
    result = bq_client.query(query)
    coords_seq = {}
    for row in result:
        # row["coords"] is returned as bytes by BigQuery client for BYTES columns
        coords_bytes = row["coords"]
        seq = row["seq"]
        coords = np.frombuffer(coords_bytes, dtype=np.float32).reshape(-1, 3)
        coords_seq[row["id"]] = (coords, seq)
    return coords_seq

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
        
        # Initialize GPU device if available
        if self.use_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None
        
        # Cache for PDB structures to avoid re-reading
        self.structure_cache = {}
        self.max_cache_size = 100  # Limit cache size to prevent memory issues
        
        # Initialize lDDT calculator
        self.lddt_calculator = LDDTCalculator()
    
    def read_from_gcs(self, gcs_path: str) -> str:
        """
        Read PDB file directly from GCS without downloading.
        
        Args:
            gcs_path: GCS path to PDB file (e.g., 'pdb_folder/Q1GF61.pdb')
            
        Returns:
            File-like object for reading the PDB file
        """
        # Get blob and return as file-like object
        blob = bucket.blob(gcs_path)
        return blob.open('r')


    def score_protein_pair(self, protein_id1: str, protein_id2: str, pdb_info_dict=None) -> Optional[Dict]:
        try:
            # Fetch CA coords and seq from provided dict (batch-fetched)
            if pdb_info_dict is not None:
                reference_coords_all, reference_seq = pdb_info_dict[protein_id1]
                model_coords_all, model_seq = pdb_info_dict[protein_id2]
            else:
                reference_coords_all, reference_seq = fetch_pdb_info(protein_id1)
                model_coords_all, model_seq = fetch_pdb_info(protein_id2)

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
            
            return {
                'id1': protein_id1,
                'id2': protein_id2,
                'tm_score': tm_score,
                'seqxA': seqxA,
                'seqM': seqM,
                'seqyA': seqyA,
                'lddt_scores': lddt_scores_padded.tolist()
            }
        except Exception as e:
            print(f"Error processing {protein_id1} vs {protein_id2}: {e}")
            return None

    def parse_tm_align_result(self, result):
        """
        Parse TM-align result to get residue alignment.
        """
        alignment = []
        idx1 = idx2 = 0
        for a1, a2, ann in zip(result.seqxA, result.seqyA, result.seqM):
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

def load_protein_pairs_simple_batch(csv_file, pdb_folder: str, batch_size: int = 32, start_line: int = 0, max_pairs: int = None) -> List[Tuple[str, str]]:
    """
    Load protein pairs from CSV file in batches, extracting only protein IDs.
    This version doesn't load pre-computed alignments - TM-align will be run fresh.
    
    Args:
        csv_file: File-like object for CSV file with protein pairs
        pdb_folder: GCS folder containing PDB files
        batch_size: Number of rows to process at once
        start_line: Line number (0-based) to start processing from
        max_pairs: Maximum number of pairs to process (None for all)
    Yields:
        List of (pdb1_gcs_path, pdb2_gcs_path) tuples
    """
    # Read CSV in chunks, skipping lines before start_line
    start_time = time.time()
    chunk_iter = pd.read_csv(csv_file, chunksize=batch_size, skiprows=range(1, start_line+1))
    elapsed_time = time.time() - start_time
    print(f"Reading csv: {elapsed_time:.2f}s")
    current_line = start_line
    pairs_processed = 0
    
    for chunk in chunk_iter:
        protein_pairs = []
        for _, row in chunk.iterrows():
            # Check if we've reached max_pairs limit
            if max_pairs and pairs_processed >= max_pairs:
                break
                
            protein_id1 = str(row['chain_1']).strip()
            protein_id2 = str(row['chain_2']).strip()
            pdb1_gcs_path = f"{pdb_folder}/{protein_id1}.pdb"
            pdb2_gcs_path = f"{pdb_folder}/{protein_id2}.pdb"
            protein_pairs.append((pdb1_gcs_path, pdb2_gcs_path))
            pairs_processed += 1
        
        if protein_pairs:  # Only yield if we have pairs
            yield protein_pairs, current_line
            current_line += len(protein_pairs)
        
        # Break if we've reached max_pairs
        if max_pairs and pairs_processed >= max_pairs:
            break

def batch_check_existing_pairs(client, table_id, pairs):
    """
    Check which pairs already exist in BigQuery using a single batch query.
    Args:
        client: BigQuery client
        table_id: BigQuery table ID
        pairs: List of (id1, id2) tuples to check
    Returns:
        set: Set of (id1, id2) tuples that already exist
    """
    if not pairs:
        return set()
    # Use tuple IN clause for better performance
    query = f"""
        SELECT id1, id2 FROM {table_id}
        WHERE (id1, id2) IN UNNEST({pairs})
    """
    results = client.query(query).result()
    return {(row.id1, row.id2) for row in results}

def process_pair_ids(args):
    protein_id1, protein_id2, pdb_info_dict = args
    # Check if both PDBs exist in GCS before processing
    pdb1_path = f"SWISS_MODEL/pdbs/{protein_id1}.pdb"
    pdb2_path = f"SWISS_MODEL/pdbs/{protein_id2}.pdb"
    blob1 = bucket.blob(pdb1_path)
    blob2 = bucket.blob(pdb2_path)
    if not blob1.exists() or not blob2.exists():
        print(f"Skipping pair {protein_id1}, {protein_id2}: one or both PDB files do not exist in GCS.")
        return None
    start_time = time.time()
    temp_extractor = LDDTExtractor(bucket_name, use_gpu=use_gpu)
    ret = temp_extractor.score_protein_pair(protein_id1, protein_id2, pdb_info_dict=pdb_info_dict)
    elapsed_time = time.time() - start_time
    print(f"{protein_id1} {protein_id2}: {elapsed_time:.2f}")
    return ret

def main():
    """
    Main function to extract LDDT scores using fresh TM-align computations.
    CSV file should contain at least 'chain_1' and 'chain_2' columns with protein IDs.
    """
    # Configuration - modify these variables as needed
    input_csv = "SWISS_MODEL/tm_score_comparison_results.csv"  # Input CSV file with protein pairs (only needs chain_1, chain_2 columns)
    pdb_folder = "SWISS_MODEL/pdbs"  # GCS folder containing PDB files 
    table_id = "mit-primes-464001.primes_data.ground_truth_scores" # BigQuery table id

    # Initialize extractor with GPU support
    extractor = LDDTExtractor(bucket_name, use_gpu=use_gpu)

    # Read CSV file directly from GCS
    try:
        csv_file = extractor.read_from_gcs(input_csv)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print(f"Make sure the file '{input_csv}' exists in bucket '{bucket_name}'")
        return

    print(f"Processing protein pairs in batches of {batch_size}...")

    # Process batches
    processed_count = 0
    successful_count = 0
    failed_count = 0
    last_line_processed = START_LINE + processed_count - 1
    for batch_idx, (protein_pairs_batch, batch_start_line) in enumerate(load_protein_pairs_simple_batch(csv_file, pdb_folder, batch_size, start_line=START_LINE, max_pairs=max_pairs)):
        print(f"Processing batch {batch_idx + 1} ({len(protein_pairs_batch)} pairs)... (starting at line {batch_start_line})")
        last_line_processed = batch_start_line + len(protein_pairs_batch) - 1
        batch_start_time = time.time()
        start_time = time.time()
        # Extract just the IDs for batch existence check
        id_pairs = [
            (os.path.splitext(os.path.basename(p1))[0], os.path.splitext(os.path.basename(p2))[0])
            for p1, p2 in protein_pairs_batch
        ]
        existing_pairs = batch_check_existing_pairs(client, table_id, id_pairs)
        # Filter out pairs that already exist
        filtered_pairs = [
            pair for pair, id_pair in zip(protein_pairs_batch, id_pairs)
            if id_pair not in existing_pairs
        ]
        filtered_id_pairs = [id_pair for pair, id_pair in zip(protein_pairs_batch, id_pairs) if id_pair not in existing_pairs]
        elapsed_time = time.time()-start_time
        print(f"id existence: {elapsed_time:.2f}s")
        if not filtered_pairs:
            print("All pairs in this batch already exist in BigQuery, skipping batch.")
            continue

        start_time = time.time()
        # Batch fetch all unique protein IDs for this batch
        unique_ids = set()
        for id1, id2 in filtered_id_pairs:
            unique_ids.add(id1)
            unique_ids.add(id2)
        pdb_info_dict = fetch_pdb_info_batch(list(unique_ids), bq_client=client)

        # Prepare arguments for each pair (now just the id pairs)
        pool_args = [
            (id1, id2, pdb_info_dict)
            for id1, id2 in filtered_id_pairs
        ]
        elapsed_time = time.time()-start_time
        print(f"pdbs: {elapsed_time:.2f}s")
        # Multiprocess: only process new pairs
        with torch_mp.Pool(processes=num_processes or min(mp.cpu_count(), len(pool_args))) as pool:
            batch_results = pool.map(process_pair_ids, pool_args)
        batch_elapsed = time.time() - batch_start_time
        # Filter out None results (skipped or failed)
        batch_results = [r for r in batch_results if r is not None]
        processed_count += len(filtered_pairs)
        successful_count += len(batch_results)
        failed_count += len(filtered_pairs) - len(batch_results)

        # Insert batch directly into BigQuery
        if batch_results:
            errors = client.insert_rows_json(table_id, batch_results)
            if errors:
                print(f"Encountered errors while inserting rows: {errors}")
            else:
                print(f"Batch {batch_idx + 1} inserted into BigQuery table {table_id}")
        print(f"Progress: {processed_count} pairs processed | {successful_count} successful | {failed_count} failed | Batch time: {batch_elapsed:.2f}s")

        # Check if we've reached max_pairs limit
        if max_pairs and processed_count >= max_pairs:
            break

    # Close CSV file
    csv_file.close()

    print(f"\nProcessing completed!")
    print(f"Total pairs processed: {processed_count}")
    print(f"(Batches inserted into BigQuery as processed)")
    print(f"Last line processed in CSV: {last_line_processed}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main() 