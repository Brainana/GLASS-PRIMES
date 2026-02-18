#!/usr/bin/env python3
"""
Time TM-align calculations from a CSV or Parquet file in GCS, fetching PDB coordinates from GCS.
Reads PDB files from SWISS_MODEL/pdbs/ using chain IDs from the CSV file.
Computes average execution time grouped by amino acid length buckets.
"""
import pandas as pd
import numpy as np
import time
from tmtools import tm_align
from collections import defaultdict
from google.cloud import storage
from google.cloud import bigquery
import gcsfs
from Bio.PDB import PDBParser

# Configuration
key_path = "mit-primes-464001-bfa03c2c5999.json"
storage_client = storage.Client.from_service_account_json(key_path)
bucket_name = "primes-bucket"  # GCS bucket name
bucket = storage_client.bucket(bucket_name)
pdb_info_table = "mit-primes-464001.primes_compbio.proteins"
PDB_FOLDER = "SWISS_MODEL/pdbs"  # GCS folder containing PDB files
input_file = "SWISS_MODEL/swiss_under_300_141M.csv"  # Input file with protein pairs
TEST_SAMPLES = 100  # Number of protein pairs to process
batch_size = 100  # Number of pairs to process in each batch

# Initialize PDB parser
pdb_parser = PDBParser(QUIET=True)

# Amino acid code mapping (3-letter to 1-letter)
three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


def get_length_bucket(length, bucket_size=200):
    """
    Get the bucket label for a given length.
    Buckets are (0,200], (200,400], (400,600], etc.
    """
    bucket_num = (length - 1) // bucket_size
    lower = bucket_num * bucket_size
    upper = (bucket_num + 1) * bucket_size
    return f"({lower},{upper}]"


def check_file_exists(gcs_path: str) -> bool:
    """
    Check if a file exists in GCS.
    
    Args:
        gcs_path: GCS path to file (e.g., 'folder/file.csv')

    Returns:
        bool: True if file exists, False otherwise
    """
    blob = bucket.blob(gcs_path)
    return blob.exists()


def read_from_gcs(gcs_path: str):
    """
    Read file directly from GCS without downloading.
    Supports both CSV and Parquet files.
    
    Args:
        gcs_path: GCS path to file (e.g., 'folder/file.csv' or 'folder/file.parquet')
        
    Returns:
        For CSV: File-like object for reading the file
        For Parquet: pandas DataFrame
    """
    # Check if file exists first
    if not check_file_exists(gcs_path):
        raise FileNotFoundError(
            f"File '{gcs_path}' not found in bucket '{bucket_name}'. "
            f"Please check the file path and ensure the file exists."
        )
    
    # Read CSV file
    blob = bucket.blob(gcs_path)
    return blob.open('r')


def extract_ca_coords_and_sequence(pdb_file):
    """
    Extract Cα coordinates and sequence from a PDB file.
    
    Args:
        pdb_file: File-like object for PDB file
        
    Returns:
        tuple: (ca_coords, sequence) where ca_coords is numpy array and sequence is string
    """
    structure = pdb_parser.get_structure("protein", pdb_file)
    model = next(structure.get_models())
    chain = next(model.get_chains())
    
    ca_coords = []
    sequence = []
    for res in chain:
        if 'CA' in res:
            ca_coords.append(res['CA'].get_coord())
            resname = res.get_resname()
            sequence.append(three_to_one.get(resname, 'X'))  # Convert to 1-letter code
    ca_coords = np.array(ca_coords, dtype=np.float32)
    return ca_coords, "".join(sequence)


def fetch_pdb_info_batch(protein_ids, bq_client=None):
    """
    Fetch PDB info (coordinates and sequences) from GCS PDB files.
    Reads PDB files from SWISS_MODEL/pdbs/{chain_id}.pdb
    
    Args:
        protein_ids: List of chain IDs to fetch
        bq_client: Not used (kept for compatibility)
        
    Returns:
        dict: {chain_id: (coords, seq)} where coords is numpy array and seq is string
    """
    if not protein_ids:
        return {}
    
    coords_seq = {}
    for chain_id in protein_ids:
        try:
            # Construct GCS path: SWISS_MODEL/pdbs/{chain_id}.pdb
            pdb_path = f"{PDB_FOLDER}/{chain_id}.pdb"
            blob = bucket.blob(pdb_path)
            
            if not blob.exists():
                print(f"Warning: PDB file {pdb_path} does not exist in GCS. Skipping {chain_id}.")
                continue
            
            # Read PDB file from GCS
            with blob.open('r') as pdb_file:
                coords, seq = extract_ca_coords_and_sequence(pdb_file)
                coords_seq[chain_id] = (coords, seq)
        except Exception as e:
            print(f"Error processing PDB for {chain_id}: {e}")
            continue
    
    return coords_seq


def compute_tm_score(original_coords, mutant_coords, original_seq, mutant_seq):
    """Compute TM-score and return both the score and the execution time."""
    try:
        # Time the TM-align calculation
        start_time = time.time()
        result = tm_align(original_coords, mutant_coords, original_seq, mutant_seq)
        elapsed_time = time.time() - start_time
        
        # Return TM-score and timing
        return result.tm_norm_chain1, elapsed_time
    except Exception as e:
        print(f"Error computing TM-score: {e}")
        return None, None


def load_protein_pairs_batch(input_data, batch_size, max_pairs):
    """
    Load protein pairs from CSV or Parquet file in batches.
    
    Args:
        input_data: File-like object for CSV file, or pandas DataFrame for Parquet
        batch_size: Number of rows to process at once
        max_pairs: Maximum number of pairs to process
        
    Yields:
        List of (protein_id1, protein_id2) tuples
    """
    pairs_processed = 0
    
    # input_data is a file-like object for CSV
    chunk_iter = pd.read_csv(input_data, chunksize=batch_size)
    for chunk in chunk_iter:
        protein_pairs = []
        for _, row in chunk.iterrows():
            if max_pairs and pairs_processed >= max_pairs:
                break
                
            protein_id1 = str(row['chain_1']).strip()
            protein_id2 = str(row['chain_2']).strip()
            protein_pairs.append((protein_id1, protein_id2))
            pairs_processed += 1
        
        if protein_pairs:
            yield protein_pairs
        
        if max_pairs and pairs_processed >= max_pairs:
            break


def main():
    """Main function to benchmark TM-align timing."""
    # BigQuery client not needed for reading PDBs from GCS, but kept for potential future use
    # bq_client = bigquery.Client.from_service_account_json(key_path)
    
    # Read file from GCS
    try:
        input_data = read_from_gcs(input_file)
        print(f"Reading from GCS: {input_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nTo list available files in the bucket, you can use:")
        print(f"  gsutil ls gs://{bucket_name}/")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        print(f"Make sure the file '{input_file}' exists in bucket '{bucket_name}'")
        return
    
    print(f"Processing up to {TEST_SAMPLES} protein pairs...")
    
    # Dictionary to store timing data by length bucket: {bucket_label: [times]}
    bucket_times = defaultdict(list)
    bucket_lengths = defaultdict(list)  # Store actual lengths for reference
    
    pairs_processed = 0
    
    # Process batches
    for batch_idx, protein_pairs_batch in enumerate(load_protein_pairs_batch(input_data, batch_size, TEST_SAMPLES)):
        if pairs_processed >= TEST_SAMPLES:
            break
        
        print(f"\nProcessing batch {batch_idx + 1} ({len(protein_pairs_batch)} pairs)...")
        
        # Fetch all unique protein IDs for this batch
        unique_ids = set()
        for id1, id2 in protein_pairs_batch:
            unique_ids.add(id1)
            unique_ids.add(id2)
        
        # Fetch PDB info from GCS (SWISS_MODEL/pdbs/)
        start_time = time.time()
        pdb_info_dict = fetch_pdb_info_batch(list(unique_ids))
        fetch_time = time.time() - start_time
        print(f"Fetched {len(pdb_info_dict)} proteins from GCS in {fetch_time:.2f}s")
        
        # Process each pair
        for protein_id1, protein_id2 in protein_pairs_batch:
            if pairs_processed >= TEST_SAMPLES:
                break
            
            try:
                # Check if we have the required protein data
                if protein_id1 not in pdb_info_dict or protein_id2 not in pdb_info_dict:
                    print(f"Skipping {protein_id1} vs {protein_id2}: missing PDB info")
                    continue
                
                # Get coordinates and sequences
                coords1, seq1 = pdb_info_dict[protein_id1]
                coords2, seq2 = pdb_info_dict[protein_id2]
                
                # Trim longer protein to match shorter one if shapes don't match
                if coords1.shape[0] != coords2.shape[0]:
                    len1_orig = coords1.shape[0]
                    len2_orig = coords2.shape[0]
                    min_len = min(len1_orig, len2_orig)
                    if len1_orig > len2_orig:
                        # Trim coords1 and seq1
                        coords1 = coords1[:min_len]
                        seq1 = seq1[:min_len]
                        print(f"Trimmed {protein_id1} from {len1_orig} to {min_len} residues to match {protein_id2}")
                    else:
                        # Trim coords2 and seq2
                        coords2 = coords2[:min_len]
                        seq2 = seq2[:min_len]
                        print(f"Trimmed {protein_id2} from {len2_orig} to {min_len} residues to match {protein_id1}")
                
                # Get amino acid length from coordinates (number of CA atoms)
                aa_length = len(coords1)
                
                # Skip proteins over 2000 amino acids
                if aa_length > 2000:
                    continue
                
                bucket = get_length_bucket(aa_length)
                
                print(f"\nProcessing {protein_id1} vs {protein_id2} (length: {aa_length}, bucket: {bucket}):")
                
                # Compute TM-score and time it
                tm_score, tm_time = compute_tm_score(coords1, coords2, seq1, seq2)
                
                if tm_score is not None and tm_time is not None:
                    print(f"  TM-score: {tm_score:.4f}")
                    print(f"  TM-align time: {tm_time:.4f} seconds")
                    bucket_times[bucket].append(tm_time)
                    bucket_lengths[bucket].append(aa_length)
                    pairs_processed += 1
                else:
                    print(f"  Failed to compute TM-score")
                    
            except Exception as e:
                print(f"Error processing {protein_id1} vs {protein_id2}: {e}")
    
    # Close file if it's a file-like object (CSV)
    if hasattr(input_data, 'close'):
        input_data.close()
    
    # Print summary statistics grouped by bucket
    if bucket_times:
        print(f"\n{'='*70}")
        print(f"TM-ALIGN TIMING STATISTICS BY AMINO ACID LENGTH BUCKET")
        print(f"{'='*70}")
        
        # Sort buckets by their lower bound
        sorted_buckets = sorted(bucket_times.keys(), key=lambda x: int(x.split(',')[0][1:]))
        
        all_times = []  # For overall statistics
        
        for bucket in sorted_buckets:
            times = bucket_times[bucket]
            lengths = bucket_lengths[bucket]
            all_times.extend(times)
            
            print(f"\nBucket: {bucket} amino acids")
            print(f"  Number of calculations: {len(times)}")
            print(f"  Length range: {min(lengths)} - {max(lengths)} amino acids")
            print(f"  Average time: {np.mean(times):.4f} seconds")
            print(f"  Median time: {np.median(times):.4f} seconds")
            print(f"  Min time: {np.min(times):.4f} seconds")
            print(f"  Max time: {np.max(times):.4f} seconds")
            print(f"  Std deviation: {np.std(times):.4f} seconds")
        
        # Print overall statistics
        print(f"\n{'='*70}")
        print(f"OVERALL STATISTICS (All Buckets Combined)")
        print(f"{'='*70}")
        print(f"Total calculations: {len(all_times)}")
        print(f"Average time: {np.mean(all_times):.4f} seconds")
        print(f"Median time: {np.median(all_times):.4f} seconds")
        print(f"Min time: {np.min(all_times):.4f} seconds")
        print(f"Max time: {np.max(all_times):.4f} seconds")
        print(f"Std deviation: {np.std(all_times):.4f} seconds")
        print(f"{'='*70}")
    else:
        print("\nNo successful TM-align calculations to time.")


if __name__ == "__main__":
    main()

