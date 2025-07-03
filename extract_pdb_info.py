import pandas as pd
import numpy as np
from google.cloud import storage, bigquery
from Bio.PDB import PDBParser
import io
import multiprocessing as mp
import os
import base64
import time
try:
    from google.colab import auth
    auth.authenticate_user()
except ImportError:
    print("Not running in Colab - skipping authentication")
    print("Make sure you have proper GCS credentials set up")

BUCKET_NAME = "jx-compbio" 
PDB_FOLDER = "SWISS_MODEL/pdbs"
INPUT_CSV = "SWISS_MODEL/tm_score_comparison_results.csv"  # Input CSV file with protein pairs (only needs chain_1, chain_2 columns)
BATCH_SIZE = 1000
TABLE_ID = "mit-primes-464001.primes_data.pdb_info"
MAX_ROWS = 1000000
START_LINE = 0  # Set this to the line number to start from (0-based, not counting header)
# Add amino acid code mapping
three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}
parser = PDBParser(QUIET=True)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
bq_client = bigquery.Client(project="mit-primes-464001")


def extract_ca_coords_and_sequence(pdb):
    structure = parser.get_structure("protein", pdb)
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
    return ca_coords.tobytes(), "".join(sequence)


def process_id(pid):
    pdb_path = f"{PDB_FOLDER}/{pid}.pdb"
    blob = bucket.blob(pdb_path)
    if not blob.exists():
        print(f"PDB file {pdb_path} does not exist in GCS. Skipping {pid}.")
        return None
    try:
        pdb = blob.open('r')
        coords_bytes, sequence = extract_ca_coords_and_sequence(pdb)
        # Base64-encode for BigQuery BYTES column via insert_rows_json
        coords_b64 = base64.b64encode(coords_bytes).decode('utf-8')
        return {
            "id": pid,
            "coords": coords_b64,  # base64 string for BigQuery BYTES column
            "seq": sequence
        }
    except Exception as e:
        print(f"Error processing {pid}: {e}")
        return None


def get_num_workers():
    cpu_count = os.cpu_count() or 1
    print(f"Detected {cpu_count} CPUs. Using all for num_workers.")
    return cpu_count


def batch_check_existing_ids(bq_client, table_id, ids):
    query = f"SELECT id FROM `{table_id}` WHERE id IN UNNEST({ids})"
    results = bq_client.query(query).result()
    return {row.id for row in results}


def process_csv_in_batches_gcs(batch_size=32, max_rows=100, num_workers=None, start_line=0):
    processed_rows = 0
    blob = bucket.blob(INPUT_CSV)
    last_percent = 0
    start_time = time.time()
    with blob.open("rb") as f:
        # Use skiprows to skip lines before start_line (skip header + start_line rows)
        skiprows = range(1, start_line + 1) if start_line > 0 else None
        for chunk in pd.read_csv(f, chunksize=batch_size, skiprows=skiprows):
            if processed_rows + len(chunk) > max_rows:
                chunk = chunk.iloc[:max_rows - processed_rows]
            ids = list(set(chunk['chain_1']).union(set(chunk['chain_2'])))
            # Duplicate check: only process new IDs
            check_start = time.time()
            existing_ids = batch_check_existing_ids(bq_client, TABLE_ID, ids)
            check_elapsed = time.time() - check_start
            print(f"Checked for existing IDs in BigQuery. {len(existing_ids)} found. (Elapsed: {check_elapsed:.2f}s)")
            new_ids = [pid for pid in ids if pid not in existing_ids]
            if len(new_ids) > 0:
                results = []
                batch_size_ids = len(new_ids)
                last_batch_percent = 0
                with mp.Pool(processes=num_workers) as pool:
                    for i, result in enumerate(pool.imap_unordered(process_id, new_ids), 1):
                        if result is not None:
                            results.append(result)
                        percent = int(100 * i / batch_size_ids)
                        if (percent // 10) > (last_batch_percent // 10) or percent == 100:
                            bar = '[' + '#' * (percent // 10) + '-' * (10 - percent // 10) + ']'
                            print(f"Batch progress: {bar} {percent}% ({i}/{batch_size_ids})")
                            last_batch_percent = percent
                rows_to_insert = results
                if rows_to_insert:
                    errors = bq_client.insert_rows_json(TABLE_ID, rows_to_insert)
                    if errors:
                        print(f"BigQuery insert errors: {errors}")
            processed_rows += len(chunk)
            print(f"Processed {processed_rows} rows so far...")
            if processed_rows >= max_rows:
                print(f"Reached max_rows limit: {max_rows}")
                break


if __name__ == "__main__":
    num_workers = get_num_workers()
    # Example: process at most 100 rows, using environment-optimized num_workers
    process_csv_in_batches_gcs(batch_size=BATCH_SIZE, max_rows=MAX_ROWS, num_workers=num_workers, start_line=START_LINE) 