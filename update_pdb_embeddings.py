#!/usr/bin/env python3

import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from google.cloud import bigquery
import json
import time
import os
from tqdm import tqdm
import base64

# Configuration
key_path = "mit-primes-464001-bfa03c2c5999.json"
TABLE_ID = "mit-primes-464001.primes_data.pdb_info"

# ProtBERT configuration
MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
model = T5EncoderModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def get_protbert_embedding_batch(sequences, pad_len=300):
    """
    Generate ProtBERT embeddings for a batch of sequences.
    
    Args:
        sequences: List of protein sequences
        pad_len: Length to pad/truncate to (default 300)
    
    Returns:
        List of numpy arrays with shape (pad_len, 1024)
    """
    # Preprocess sequences
    processed_sequences = []
    for seq in sequences:
        seq = seq.replace(' ', '').replace('-', '').upper()
        seq_spaced = ' '.join(list(seq))
        processed_sequences.append(seq_spaced)
    
    # Tokenize batch
    inputs = tokenizer(processed_sequences, return_tensors='pt', padding=True, truncation=True, max_length=300)
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state  # (batch_size, seq_len, 1024)
    
    # Handle padding for each sequence
    padded_embeddings = []
    for i in range(embeddings.size(0)):
        emb = embeddings[i]  # (seq_len, 1024)
        
        # Pad or truncate to pad_len
        if emb.size(0) < pad_len:
            pad = torch.zeros(pad_len - emb.size(0), emb.size(1), device=emb.device)
            emb = torch.cat([emb, pad], dim=0)
        else:
            emb = emb[:pad_len]
        
        # Convert to bytes for BigQuery storage
        emb_bytes = emb.cpu().numpy().tobytes()
        padded_embeddings.append(emb_bytes)
    
    return padded_embeddings

def fetch_sequences_batch(client, protein_ids, batch_size=100):
    """
    Fetch sequences from BigQuery for a batch of protein IDs.
    
    Args:
        client: BigQuery client
        protein_ids: List of protein IDs
        batch_size: Size of each batch
    
    Returns:
        Dictionary mapping protein_id to sequence
    """
    if not protein_ids:
        return {}
    
    # Create IN clause for query
    id_list = ', '.join([f"'{id_}'" for id_ in protein_ids])
    query = f"""
        SELECT id, seq 
        FROM `{TABLE_ID}` 
        WHERE id IN ({id_list})
    """
    
    results = client.query(query).result()
    return {row.id: row.seq for row in results}

def check_existing_embeddings(client, protein_ids):
    """
    Check which protein IDs already have embeddings.
    
    Args:
        client: BigQuery client
        protein_ids: List of protein IDs
    
    Returns:
        Set of protein IDs that already have embeddings
    """
    if not protein_ids:
        return set()
    
    id_list = ', '.join([f"'{id_}'" for id_ in protein_ids])
    query = f"""
        SELECT id 
        FROM `{TABLE_ID}` 
        WHERE id IN ({id_list}) 
        AND embeddings IS NOT NULL
    """
    
    results = client.query(query).result()
    return {row.id for row in results}

def update_embeddings_batch(client, updates_batch):
    """
    Update embeddings for a batch of proteins using a single MERGE statement.
    
    Args:
        client: BigQuery client
        updates_batch: List of dicts with 'id' and 'embeddings' keys
    """
    if not updates_batch:
        return
    
    # Convert all embeddings to base64 strings
    rows_data = []
    for update in updates_batch:
        protein_id = update['id']
        embeddings_bytes = update['embeddings']
        embeddings_b64 = base64.b64encode(embeddings_bytes).decode('utf-8')
        rows_data.append({
            'id': protein_id,
            'embeddings_b64': embeddings_b64
        })
    
    # Create a single MERGE statement with base64 strings
    merge_query = f"""
    MERGE `{TABLE_ID}` T
    USING UNNEST(@rows) AS S
    ON T.id = S.id
    WHEN MATCHED THEN
      UPDATE SET embeddings = FROM_BASE64(S.embeddings_b64)
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "rows",
                "STRUCT<id STRING, embeddings_b64 STRING>",
                rows_data
            )
        ]
    )
    
    client.query(merge_query, job_config=job_config).result()

def main():
    """
    Main function to update pdb_info table with ProtBERT embeddings.
    """
    print(f"Starting embedding update for table: {TABLE_ID}")
    print(f"Using device: {device}")
    
    # Initialize BigQuery client
    client = bigquery.Client.from_service_account_json(key_path)
    
    # Get total count of proteins that need embeddings
    count_query = f"""
        SELECT COUNT(*) as total_count
        FROM `{TABLE_ID}`
        WHERE embeddings IS NULL
    """
    total_count = list(client.query(count_query).result())[0].total_count
    print(f"Found {total_count} proteins that need embeddings")
    
    if total_count == 0:
        print("All proteins already have embeddings!")
        return
    
    # Configuration
    batch_size = 10  # Process 10 sequences at a time (good for GPU)
    
    # Process in batches
    processed_count = 0
    successful_count = 0
    failed_count = 0
    
    # Get all protein IDs that need embeddings
    ids_query = f"""
        SELECT id 
        FROM `{TABLE_ID}`
        WHERE embeddings IS NULL
        ORDER BY id
    """
    
    protein_ids = [row.id for row in client.query(ids_query).result()]
    
    print(f"Processing {len(protein_ids)} proteins in batches of {batch_size}...")
    
    # Process in batches
    for i in tqdm(range(0, len(protein_ids), batch_size), desc="Processing batches"):
        batch_ids = protein_ids[i:i + batch_size]
        
        try:
            # Check which ones already have embeddings (in case of restart)
            existing_embeddings = check_existing_embeddings(client, batch_ids)
            new_ids = [id_ for id_ in batch_ids if id_ not in existing_embeddings]
            
            if not new_ids:
                print(f"Batch {i//batch_size + 1}: All proteins already have embeddings")
                continue
            
            # Fetch sequences for this batch
            sequences_dict = fetch_sequences_batch(client, new_ids)
            
            if not sequences_dict:
                print(f"Batch {i//batch_size + 1}: No sequences found")
                continue
            
            # Generate embeddings
            sequences = list(sequences_dict.values())
            protein_ids_batch = list(sequences_dict.keys())
            
            embeddings_batch = get_protbert_embedding_batch(sequences)
            
            # Prepare updates
            updates = []
            for protein_id, embedding_bytes in zip(protein_ids_batch, embeddings_batch):
                updates.append({
                    'id': protein_id,
                    'embeddings': embedding_bytes
                })
            
            # Update BigQuery
            update_embeddings_batch(client, updates)
            
            processed_count += len(new_ids)
            successful_count += len(updates)
            
            print(f"Batch {i//batch_size + 1}: Processed {len(updates)} proteins")
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            failed_count += len(batch_ids)
            continue
    
    print(f"\nProcessing completed!")
    print(f"Total processed: {processed_count}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")

if __name__ == "__main__":
    main() 