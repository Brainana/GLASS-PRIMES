#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import gcsfs
import os

class SiameseParquetDataset(Dataset):
    def __init__(self, bucket_name, folder, max_len, key_path, gcs_project=None):
        """
        Args:
            bucket_name: GCS bucket name
            folder: GCS folder path within the bucket
            max_len: Maximum length to pad/truncate protein embeddings
            gcs_project: (Optional) GCP project for GCS access
            key_path: Path to GCS service account key JSON file
        """
        if not key_path:
            raise ValueError("key_path is required for GCS authentication")
        
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Key file not found: {key_path}")
        
        gcs_folder = f"gs://{bucket_name}/{folder}/"
        self.gcs_folder = gcs_folder.rstrip('/') + '/'
        self.max_len = max_len
        self.key_path = key_path
        self.gcs_project = gcs_project

        # Use gcsfs to enumerate all Parquet files in the folder
        # Use service account key for authentication
        self.fs = gcsfs.GCSFileSystem(project=gcs_project, token=key_path)
        self.parquet_uris = [f'gs://{path}' for path in self.fs.ls(self.gcs_folder) if path.endswith('.parquet')]
        self.parquet_uris.sort()  # Ensure consistent order

        # Lazy load: just get row counts for indexing
        self.file_row_counts = []
        self.cumulative_rows = []
        total = 0
        for uri in self.parquet_uris:
            df = pd.read_parquet(uri, columns=["tm_score"], filesystem=self.fs)
            n = len(df)
            self.file_row_counts.append(n)
            total += n
            self.cumulative_rows.append(total)
        self.total_rows = total
        self.current_file_idx = None
        self.current_df = None

    def __len__(self):
        return self.total_rows

    def _find_file_and_local_idx(self, idx):
        for file_idx, cum in enumerate(self.cumulative_rows):
            if idx < cum:
                if file_idx == 0:
                    local_idx = idx
                else:
                    local_idx = idx - self.cumulative_rows[file_idx - 1]
                return int(file_idx), int(local_idx)
        raise IndexError("Index out of range")

    def __getitem__(self, idx):
        file_idx, local_idx = self._find_file_and_local_idx(idx)
        
        # Ensure local_idx is an integer
        local_idx = int(local_idx)
        
        uri = self.parquet_uris[file_idx]
        if self.current_file_idx != file_idx:
            # Create a new filesystem instance for each file to avoid fork-safety issues
            fs = gcsfs.GCSFileSystem(project=self.gcs_project, token=self.key_path)
            self.current_df = pd.read_parquet(uri, filesystem=fs)
            self.current_file_idx = file_idx
        
        # Debug: print indices for first few items
        if idx < 3:
            print(f"Item {idx}: file_idx={file_idx}, local_idx={local_idx}, df_shape={self.current_df.shape}")
        
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
                    # Fallback: create a dummy 2D array with one row
                    return arr.reshape(1, -1)
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
        
        # Handle lddt_scores properly - pad to max_len if needed
        lddt = item['lddt_scores']
        if len(lddt) < max_len:
            # Pad with zeros if shorter than max_len
            lddt = np.pad(lddt, (0, max_len - len(lddt)), mode='constant')
        else:
            # Truncate if longer than max_len
            lddt = lddt[:max_len]
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
# dataset = SiameseParquetDataset('gs://your-bucket/your-folder/', max_len=300)
# loader = DataLoader(dataset, batch_size=32, collate_fn=lambda b: siamese_collate_fn(b, max_len=300)) 