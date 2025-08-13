#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
import pandas as pd
import numpy as np
import os
import random
import pyarrow.parquet as pq

# Environment configuration
ENVIRONMENT = os.getenv('PRIMES_ENV').lower()  # Default to gcp if not set
if ENVIRONMENT not in ['gcp', 'psc']:
    raise ValueError(f"PRIMES_ENV must be either 'gcp' or 'psc', got '{ENVIRONMENT}'")

# Conditional imports based on environment
if ENVIRONMENT == 'gcp':
    import gcsfs
else:
    # For PSC environment, we'll use local file system
    pass

class SiameseParquetDataset(Dataset):
    def __init__(self, data_folder, max_len, gcs_project=None, key_path=None):
        """
        Args:
            data_folder: Data folder path 
                - For GCP: GCS folder path (e.g., 'gs://bucket/folder/')
                - For PSC: Local directory path (e.g., '/path/to/data/')
            max_len: Maximum length to pad/truncate protein embeddings
            gcs_project: (Optional) GCP project for GCS access (only used in GCP mode)
            key_path: (Optional) path to GCP service account key (only used in GCP mode)
        """
        self.environment = ENVIRONMENT
        self.data_folder = data_folder.rstrip('/') + '/'
        self.max_len = max_len
        self.gcs_project = gcs_project
        self.key_path = key_path
        self.fs = None  # Will be set per worker
        self.all_parquet_uris = None
        self.uri_to_row_count = {} 
        self.file_row_counts = None
        self.cumulative_rows = None
        self.total_rows = None
        self.current_file_idx = None
        self.current_df = None

        # Enumerate all Parquet files in the folder (do this once in main process)
        if self.environment == 'gcp':
            tempfs = gcsfs.GCSFileSystem(project=self.gcs_project, token=self.key_path)
            self.all_parquet_uris = [f'gs://{path}' for path in tempfs.ls(self.data_folder) if path.endswith('.parquet')]
        else:
            self.all_parquet_uris = [
                os.path.join(self.data_folder, f) 
                for f in os.listdir(self.data_folder) 
                if f.endswith('.parquet')
            ]

        for uri in self.all_parquet_uris:
            if self.environment == 'gcp':
                with tempfs.open(uri) as f:
                    pf = pq.ParquetFile(f)
                    n = pf.metadata.num_rows
                    self.uri_to_row_count[uri] = n
            else:
                # PSC environment: read local file
                pf = pq.ParquetFile(uri)
                n = pf.metadata.num_rows
                self.uri_to_row_count[uri] = n
        
        if self.environment == 'gcp':
            tempfs = None

        self._build_shuffled_index_cache()

    def _build_shuffled_index_cache(self):
        # Shuffle the URIs
        random.shuffle(self.all_parquet_uris)
        # Build file_row_counts and cumulative_rows based on shuffled uris
        self.file_row_counts = []
        self.cumulative_rows = []
        total = 0
        for uri in self.all_parquet_uris:
            n = self.uri_to_row_count[uri]
            self.file_row_counts.append(n)
            total += n
            self.cumulative_rows.append(total)
        self.total_rows = total

    def _get_fs(self):
        if self.environment == 'gcp':
            if self.fs is None:
                self.fs = gcsfs.GCSFileSystem(project=self.gcs_project, token=self.key_path)
            return self.fs
        else:
            # PSC environment: no file system needed for local files
            return None

    def __len__(self):
        return self.total_rows

    def _find_file_and_local_idx(self, idx):
        for file_idx, cum in enumerate(self.cumulative_rows):
            if idx < cum:
                if file_idx == 0:
                    local_idx = idx
                else:
                    local_idx = idx - self.cumulative_rows[file_idx - 1]
                return file_idx, local_idx
        raise IndexError("Index out of range")

    def __getitem__(self, idx):
        file_idx, local_idx = self._find_file_and_local_idx(idx)
        uri = self.all_parquet_uris[file_idx]
        
        if self.current_file_idx != file_idx:
            if self.environment == 'gcp':
                fs = self._get_fs()
                self.current_df = pd.read_parquet(uri, filesystem=fs)
            else:
                # PSC environment: read local file directly
                self.current_df = pd.read_parquet(uri)
            self.current_file_idx = file_idx
            
        row = self.current_df.iloc[local_idx]
        emb1 = self._to_numpy(row['embeddings1'])
        emb2 = self._to_numpy(row['embeddings2'])
        lddt_scores = self._to_numpy(row['lddt_scores'])
        return {
            'embeddings1': emb1,
            'embeddings2': emb2,
            'tm_score': float(row['tm_score']),
            'lddt_scores': lddt_scores,
            'seqxA': row['seqxA'],
            'seqyA': row['seqyA'],
            'seqM': row['seqM'],
        }

    def _to_numpy(self, val):
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, bytes):
            # Convert bytes to numpy array and reshape to 2D
            arr = np.frombuffer(val, dtype=np.float32)
            # ProtTrans embeddings are 1024-dimensional per residue
            embedding_dim = 1024
            if len(arr) % embedding_dim == 0:
                seq_len = len(arr) // embedding_dim
                return arr.reshape(seq_len, embedding_dim)
            else:
                # If not divisible by 1024, try to infer the sequence length
                # This might happen if the embedding was truncated
                seq_len = len(arr) // embedding_dim
                if seq_len > 0:
                    return arr[:seq_len * embedding_dim].reshape(seq_len, embedding_dim)
                else:
                    # Fallback: return as 1D array
                    return arr
        if isinstance(val, list):
            # Handle list of floats (like lddt_scores)
            return np.array(val, dtype=np.float32)
        return np.array(val)

def siamese_collate_fn(batch, max_len):
    def pad_and_mask(arr, max_len):
        arr = arr[:max_len]
        pad_width = max_len - arr.shape[0]
        if pad_width > 0:
            arr = np.pad(arr, ((0, pad_width), (0, 0)), mode='constant')
            mask = np.concatenate([np.ones(arr.shape[0] - pad_width), np.zeros(pad_width)])
        else:
            mask = np.ones(max_len)
        return arr, mask

    emb1s, emb2s, mask1s, mask2s, tm_scores, lddt_scores, seqxAs, seqyAs, seqMs = [], [], [], [], [], [], [], [], []
    for item in batch:
        e1, m1 = pad_and_mask(item['embeddings1'], max_len)
        e2, m2 = pad_and_mask(item['embeddings2'], max_len)
        emb1s.append(e1)
        emb2s.append(e2)
        mask1s.append(m1)
        mask2s.append(m2)
        tm_scores.append(item['tm_score'])

        lddt = item['lddt_scores']
        if lddt is None or len(lddt) < max_len:
            lddt = np.zeros(max_len, dtype=np.float32)
        
        lddt_scores.append(lddt)
        seqxAs.append(item['seqxA'])
        seqyAs.append(item['seqyA'])
        seqMs.append(item['seqM'])

    # Convert lists to numpy arrays before creating tensors to avoid the warning
    emb1s_array = np.array(emb1s, dtype=np.float32)
    emb2s_array = np.array(emb2s, dtype=np.float32)
    mask1s_array = np.array(mask1s, dtype=np.float32)
    mask2s_array = np.array(mask2s, dtype=np.float32)
    lddt_scores_array = np.array(lddt_scores, dtype=np.float32)

    return {
        'embeddings1': torch.tensor(emb1s_array, dtype=torch.float32),
        'embeddings2': torch.tensor(emb2s_array, dtype=torch.float32),
        'mask1': torch.tensor(mask1s_array, dtype=torch.float32),
        'mask2': torch.tensor(mask2s_array, dtype=torch.float32),
        'tm_score': torch.tensor(tm_scores, dtype=torch.float32),
        'lddt_scores': torch.tensor(lddt_scores_array, dtype=torch.float32),
        'seqxA': seqxAs,
        'seqyA': seqyAs,
        'seqM': seqMs,
    }

# Example usage:
# For GCP:
# dataset = SiameseParquetDataset('gs://your-bucket/your-folder/', max_len=300, gcs_project='your-gcp-project', key_path='path/to/key.json')
# For PSC:
# dataset = SiameseParquetDataset('/path/to/local/data/', max_len=300)
# loader = DataLoader(dataset, batch_size=32, collate_fn=lambda b: siamese_collate_fn(b, max_len=300), num_workers=4) 