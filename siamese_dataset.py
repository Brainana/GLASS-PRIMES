import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import io
from google.cloud import bigquery
from transformers import AutoTokenizer, AutoModel
import torch


class SiameseProteinDataset(Dataset):
    """
    Dataset for Siamese protein network training.
    Loads protein pairs directly from BigQuery and generates ProtBERT embeddings on-demand.
    """
    def __init__(self, max_samples, bq_client=None, ground_truth_table=None, 
                 max_seq_len=300, prottrans_dim=1024, data_batch_size=32):
        """
        Initialize dataset.
        
        Args:
            max_samples: Maximum number of samples to load (None for all)
            bq_client: BigQuery client
            ground_truth_table: BigQuery table name
            max_seq_len: Maximum sequence length for padding
            prottrans_dim: Dimension of ProtTrans embeddings
            data_batch_size: Number of pairs to process in each batch
        """
        self.max_samples = max_samples
        self.bq_client = bq_client
        self.ground_truth_table = ground_truth_table
        self.max_seq_len = max_seq_len
        self.prottrans_dim = prottrans_dim
        self.data_batch_size = data_batch_size
        
        # Initialize ProtBERT
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.protbert_model = AutoModel.from_pretrained("Rostlab/prot_bert", use_safetensors=True).to(device)
        self.protbert_model.eval()
        self.device = device
        
        # Batch cache to avoid reloading the same batch multiple times
        self.batch_cache = None
        self.current_batch_idx = None
        
    def _get_protbert_embeddings(self, sequence, pad_len=300):
        """Generate ProtBERT embeddings for a sequence."""
        # Remove whitespace and uppercase (ProtBERT expects space-separated, uppercase)
        seq = sequence.replace(' ', '').replace('-', '').upper()
        seq_spaced = ' '.join(list(seq))
        inputs = self.tokenizer(seq_spaced, return_tensors='pt')
        with torch.no_grad():
            outputs = self.protbert_model(**{k: v.to(self.device) for k, v in inputs.items()})
            emb = outputs.last_hidden_state.squeeze(0)  # (seq_len, 1024)
        # Pad to (pad_len, 1024)
        if emb.size(0) < pad_len:
            pad = torch.zeros(pad_len - emb.size(0), emb.size(1), device=emb.device)
            emb = torch.cat([emb, pad], dim=0)
        else:
            emb = emb[:pad_len]
        return emb.cpu().numpy()
    
    def _load_batch(self, batch_idx):
        """Load a specific batch of data directly from BigQuery."""
        # Calculate start and end rows for this batch
        start_row = batch_idx * self.data_batch_size
        end_row = start_row + self.data_batch_size
        
        print(f"Loading batch {batch_idx + 1} (rows {start_row}-{end_row})...")
        
        try:
            # Query BigQuery for this batch
            query = f'''
                SELECT id1, id2, tm_score, lddt_scores, seqxA, seqM, seqyA
                FROM `{self.ground_truth_table}`
                LIMIT {self.data_batch_size} OFFSET {start_row}
            '''
            
            bq_results = self.bq_client.query(query).result()
            
            batch_data = []
            
            # Process each pair in the batch
            for row in bq_results:
                try:
                    tm_score = float(row.tm_score) if row.tm_score is not None else 0.0
                    
                    # lddt_scores are stored as ARRAY<FLOAT64> in BigQuery
                    lddt_scores = np.array(row.lddt_scores, dtype=np.float32)
                    
                    # Pad or truncate lddt_scores to ensure consistent size
                    if len(lddt_scores) < 300:
                        # Pad with zeros if too short
                        lddt_scores = np.pad(lddt_scores, (0, 300 - len(lddt_scores)), mode='constant', constant_values=0.0)
                    
                    seqxA = row.seqxA
                    seqM = row.seqM
                    seqyA = row.seqyA
                    
                    # Generate ProtBERT embeddings for both proteins (right-padded to 300)
                    prot1_embeddings = self._get_protbert_embeddings(seqxA)
                    prot2_embeddings = self._get_protbert_embeddings(seqyA)
                    
                    # Create masks (1 for actual residues, 0 for padding)
                    prot1_mask = np.ones(self.max_seq_len)
                    prot2_mask = np.ones(self.max_seq_len)
                    
                    # Adjust masks based on actual sequence length
                    actual_len1 = len(seqxA.replace('-', ''))
                    actual_len2 = len(seqyA.replace('-', ''))
                    
                    if actual_len1 < self.max_seq_len:
                        prot1_mask[actual_len1:] = 0
                    if actual_len2 < self.max_seq_len:
                        prot2_mask[actual_len2:] = 0
                    
                    processed_data = {
                        'protein1_id': str(row.id1),
                        'protein2_id': str(row.id2),
                        'prot1_embeddings': prot1_embeddings,
                        'prot2_embeddings': prot2_embeddings,
                        'prot1_mask': prot1_mask,
                        'prot2_mask': prot2_mask,
                        'tm_score': tm_score,
                        'lddt_scores': lddt_scores,
                        'seqxA': seqxA,
                        'seqyA': seqyA,
                        'seqM': seqM
                    }
                    
                    batch_data.append(processed_data)
                    
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            print(f"Batch {batch_idx + 1}: Loaded {len(batch_data)} protein pairs.")
            return batch_data
            
        except Exception as e:
            print(f"Error loading batch {batch_idx}: {e}")
            return []
    
    def __len__(self):
        return self.max_samples
    
    def __getitem__(self, idx):
        # Calculate which batch this index belongs to
        batch_idx = idx // self.data_batch_size
        local_idx = idx % self.data_batch_size
        
        # Load the batch if not already cached or if we've moved to a new batch
        if batch_idx != self.current_batch_idx:
            # Clear cache and load new batch
            self.batch_cache = None
            self.batch_cache = self._load_batch(batch_idx)
            self.current_batch_idx = batch_idx
        
        batch_data = self.batch_cache
        
        data = batch_data[local_idx]
        
        return {
            'prot1_embeddings': torch.FloatTensor(data['prot1_embeddings']),
            'prot2_embeddings': torch.FloatTensor(data['prot2_embeddings']),
            'prot1_mask': torch.FloatTensor(data['prot1_mask']),
            'prot2_mask': torch.FloatTensor(data['prot2_mask']),
            'lddt_scores': torch.FloatTensor(data['lddt_scores']),
            'tm_score': torch.FloatTensor([data['tm_score']]),
            'seqxA': data['seqxA'],
            'seqM': data['seqM'],
            'seqyA': data['seqyA']
        } 