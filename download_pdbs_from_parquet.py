#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import pyarrow.parquet as pq

# === GLOBAL VARIABLES: edit these before running ===
INPUT_PARQUET = "/ocean/projects/bio250072p/jxu23/swiss_under_1000_320M.parquet"  # path to your parquet file
OUTPUT_DIR = "/ocean/projects/bio250072p/jxu23/pdbs"      # Local directory to save PDB files
START_ROW = 100                                    # Row to start processing from (0-based, 0 = first row after header)
MAX_PAIRS = 300000                                 # maximum number of pairs to process from parquet
COL1 = "chain_1"                                 # name of first column in parquet
COL2 = "chain_2"                                 # name of second column in parquet

# Download settings - OPTIMIZED FOR SPEED
DOWNLOAD_DELAY = 1                               # delay between downloads (seconds)
TIMEOUT = 30                                     # timeout for requests
MAX_WORKERS = 4                                  # parallel downloads
CHUNK_SIZE = 8192                                # chunk size for downloading files
MAX_RETRIES = 3                                  # Try up to 4 times total
RETRY_DELAY = 5.0                                # Base delay for exponential backoff
# ==================================================

# Try to import requests, otherwise use urllib
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False

# Thread-safe counters
class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self):
        with self._lock:
            return self._value

def create_session():
    """Create an optimized requests session with connection pooling"""
    if not HAS_REQUESTS:
        return None
    
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    # Configure adapter with connection pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

def download_pdb_from_swissmodel(uniprot_id: str, session=None, timeout: int = 30) -> bool:
    """
    Download PDB file from Swiss-Model REST API and save to local directory.
    Returns True if successful, False otherwise.
    """
    url = f"https://swissmodel.expasy.org/repository/uniprot/{uniprot_id}.pdb?provider=swissmodel"
    local_file_path = os.path.join(OUTPUT_DIR, f"{uniprot_id}.pdb")

    # Check if file already exists locally
    if os.path.exists(local_file_path):
        return True

    for attempt in range(MAX_RETRIES + 1):
        try:
            if HAS_REQUESTS and session:
                resp = session.get(url, timeout=timeout, stream=True)
                if resp.status_code == 200:
                    # Save to local file
                    with open(local_file_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            f.write(chunk)
                    return True
                elif resp.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    print(f"⚠️  Rate limited (429) for {uniprot_id}, waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES + 1}")
                    time.sleep(wait_time)
                    continue
                elif resp.status_code == 404:
                    print(f"⚠️  PDB not found (404) for {uniprot_id}")
                    return False
                else:
                    print(f"Warning: HTTP {resp.status_code} for ID {uniprot_id}", file=sys.stderr)
                    return False
            else:
                # Fallback to urllib
                req = urllib.request.Request(url)
                try:
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        if resp.status == 200:
                            content = resp.read()
                            with open(local_file_path, 'wb') as f:
                                f.write(content)
                            return True
                        elif resp.status == 429:
                            # Rate limited - wait and retry
                            wait_time = RETRY_DELAY * (2 ** attempt)
                            print(f"⚠️  Rate limited (429) for {uniprot_id}, waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES + 1}")
                            time.sleep(wait_time)
                            continue
                        elif resp.status == 404:
                            print(f"⚠️  PDB not found (404) for {uniprot_id}")
                            return False
                        else:
                            print(f"Warning: HTTP {resp.status} for ID {uniprot_id}", file=sys.stderr)
                            return False
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        # Rate limited - wait and retry
                        wait_time = RETRY_DELAY * (2 ** attempt)
                        print(f"⚠️  Rate limited (429) for {uniprot_id}, waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES + 1}")
                        time.sleep(wait_time)
                        continue
                    elif e.code == 404:
                        print(f"⚠️  PDB not found (404) for {uniprot_id}")
                        return False
                    else:
                        print(f"Warning: HTTPError {e.code} for ID {uniprot_id}", file=sys.stderr)
                        return False
                except urllib.error.URLError as e:
                    print(f"Warning: URLError {e.reason} for ID {uniprot_id}", file=sys.stderr)
                    return False
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait_time = RETRY_DELAY * (2 ** attempt)
                print(f"⚠️  Exception for {uniprot_id}, waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES + 1}: {e}")
                time.sleep(wait_time)
                continue
            else:
                print(f"Warning: Exception downloading {uniprot_id} after {MAX_RETRIES + 1} attempts: {e}", file=sys.stderr)
                return False

    return False

def download_worker(args):
    """Worker function for parallel downloads"""
    uniprot_id, session, counter, total_count = args
    idx = counter.increment()
    thread_id = threading.get_ident()  # Get unique thread ID

    print(f"[Thread {thread_id}] [{idx}/{total_count}] Downloading {uniprot_id}...", end=' ', flush=True)

    start_time = time.perf_counter()
    success = download_pdb_from_swissmodel(uniprot_id, session)
    end_time = time.perf_counter()

    if success:
        print(f"✅ OK ({end_time - start_time:.2f}s)")
        return True
    else:
        print("❌ Failed")
        return False

def collect_ids_from_parquet(parquet_path: str, col1: str, col2: str, max_pairs: int, start_row: int = 0) -> "tuple[set, int, int]":
    """
    Read the parquet file at parquet_path, expecting columns col1 and col2.
    Process up to max_pairs rows starting from start_row and collect unique IDs from those two columns.
    Uses pyarrow to read in chunks to avoid loading entire file into memory.
    
    Returns:
        tuple: (ids: set, pairs_processed: int, last_row: int)
    """
    ids = set()
    pairs_processed = 0
    last_row = start_row - 1  # Will be updated as we process rows
    
    # Open parquet file without loading into memory
    parquet_file = pq.ParquetFile(parquet_path)
    
    # Get total number of rows
    total_rows = parquet_file.metadata.num_rows
    
    # Check if start_row is valid
    if start_row < 0:
        start_row = 0
    if start_row >= total_rows:
        raise ValueError(f"start_row {start_row} is beyond the number of rows ({total_rows}) in the parquet file")
    
    # Determine which row groups to read
    # We'll read row groups that contain our target rows
    row_groups_to_read = []
    current_row = 0
    
    for i in range(parquet_file.num_row_groups):
        row_group_metadata = parquet_file.metadata.row_group(i)
        row_group_size = row_group_metadata.num_rows
        row_group_end = current_row + row_group_size
        
        # Check if this row group overlaps with our target range
        if row_group_end > start_row:
            row_groups_to_read.append(i)
        
        # Stop if we've covered enough rows
        if row_group_end >= start_row + max_pairs:
            break
        
        current_row = row_group_end
    
    # Read and process row groups
    current_global_row = 0
    for row_group_idx in row_groups_to_read:
        # Read this row group
        table = parquet_file.read_row_group(row_group_idx)
        df_chunk = table.to_pandas()
        
        # Verify columns exist
        if col1 not in df_chunk.columns or col2 not in df_chunk.columns:
            raise ValueError(f"Parquet file must contain columns '{col1}' and '{col2}'")
        
        # Process rows in this chunk
        for local_idx, (_, row) in enumerate(df_chunk.iterrows()):
            global_row_idx = current_global_row + local_idx
            
            # Skip rows before start_row
            if global_row_idx < start_row:
                continue
            
            # Check if we've reached max_pairs limit
            if pairs_processed >= max_pairs:
                break
            
            # Process this row
            for col in (col1, col2):
                val = str(row.get(col, "")).strip()
                if val and val != 'nan':
                    ids.add(val)
            
            pairs_processed += 1
            last_row = global_row_idx  # Update last processed row
            print(f"Processed row {global_row_idx}")
        
        current_global_row += len(df_chunk)
        
        # Break if we've processed enough pairs
        if pairs_processed >= max_pairs:
            break
    
    return ids, pairs_processed, last_row

def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def main():
    start_time = time.time()

    print(f"📂 Input Parquet: {INPUT_PARQUET}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print(f"⚡ Download settings: {MAX_WORKERS} workers, {DOWNLOAD_DELAY}s delay, {TIMEOUT}s timeout")
    print(f"📍 Starting at row: {START_ROW}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect IDs from parquet file
    try:
        ids, pairs_processed, last_row = collect_ids_from_parquet(INPUT_PARQUET, COL1, COL2, MAX_PAIRS, START_ROW)
    except Exception as e:
        print(f"Error while reading parquet file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"📊 Processed {pairs_processed} pairs (limit {MAX_PAIRS})")
    print(f"📍 Last row processed: {last_row}")
    print(f"🔍 Collected {len(ids)} unique IDs")
    print(f"💾 Downloading PDBs from Swiss-Model to: {OUTPUT_DIR}")
    print("-" * 60)

    # Check existing files in local directory
    existing_files = set()
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith('.pdb'):
                # Extract protein ID from filename
                protein_id = filename.replace(".pdb", "")
                existing_files.add(protein_id)

    ids_to_download = [id for id in sorted(ids) if id not in existing_files]
    skip_count = len(ids) - len(ids_to_download)

    if skip_count > 0:
        print(f"⏭️  Skipping {skip_count} already existing files in local directory")

    if not ids_to_download:
        print("✅ All files already exist in local directory!")
        return

    print(f"🚀 Starting parallel download of {len(ids_to_download)} files with {MAX_WORKERS} workers...")

    # Create optimized session
    session = create_session() if HAS_REQUESTS else None

    # Setup counters
    counter = ThreadSafeCounter()
    success_count = 0
    fail_count = 0

    # Download files in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Prepare arguments for workers
        args_list = [(id, session, counter, len(ids_to_download))
                    for id in ids_to_download]

        # Submit all tasks
        future_to_id = {executor.submit(download_worker, args): args[0]
                       for args in args_list}

        # Process completed tasks
        for future in as_completed(future_to_id):
            if future.result():
                success_count += 1
            else:
                fail_count += 1

    # Final summary
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"🎉 Download Complete!")
    print(f"✅ Success: {success_count}")
    print(f"⏭️  Skipped: {skip_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📍 Last row processed: {last_row}")
    print(f"💡 To resume, set START_ROW = {last_row + 1}")
    print(f"📁 Files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"⏱️  Total time: {format_time(total_time)}")
    if len(ids_to_download) > 0:
        print(f"🚀 Average time per file: {total_time/len(ids_to_download):.2f}s")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

