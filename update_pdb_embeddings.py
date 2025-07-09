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
import pickle
import os
import json

# Configuration
PROJECT = "mit-primes-464001"
DATASET = "primes_data"
TABLE = "pdb_info"
TABLE_ID = f"{PROJECT}.{DATASET}.{TABLE}"

# New embeddings table
EMBEDDINGS_TABLE = "embeddings"
EMBEDDINGS_TABLE_ID = f"{PROJECT}.{DATASET}.{EMBEDDINGS_TABLE}"

# BigQuery authentication
key_path = "mit-primes-464001-bfa03c2c5999.json"

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

def check_existing_embeddings_in_embeddings_table(client, protein_ids):
    """
    Check which protein IDs already have embeddings in the embeddings table.
    
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
        FROM `{EMBEDDINGS_TABLE_ID}` 
        WHERE id IN ({id_list})
        AND embeddings IS NOT NULL
    """
    
    results = client.query(query).result()
    return {row.id for row in results}

def save_embeddings_batch(embeddings_batch, batch_num):
    """
    Save embeddings to local files for batch loading.
    
    Args:
        embeddings_batch: List of dicts with 'id' and 'embeddings' keys
        batch_num: Batch number for file naming
    
    Returns:
        Path to saved file
    """
    if not embeddings_batch:
        return None
    
    # Create embeddings directory
    os.makedirs('embeddings_temp', exist_ok=True)
    
    # Convert embeddings to base64 strings for JSON serialization
    json_batch = []
    for item in embeddings_batch:
        json_batch.append({
            'id': item['id'],
            'embeddings': base64.b64encode(item['embeddings']).decode('utf-8')
        })
    
    # Save batch to newline-delimited JSON file (one JSON object per line)
    filename = f'embeddings_temp/batch_{batch_num:06d}.json'
    with open(filename, 'w') as f:
        for item in json_batch:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved batch {batch_num} to {filename}")
    return filename

def upload_files_to_bigquery(client, file_paths):
    """
    Upload multiple files to BigQuery using load jobs.
    
    Args:
        client: BigQuery client
        file_paths: List of file paths to upload
    """
    if not file_paths:
        return
    
    start_time = time.time()
    print(f"Uploading {len(file_paths)} files to BigQuery...")
    
    # Configure the load job
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        schema=[
            bigquery.SchemaField("id", "STRING"),
            bigquery.SchemaField("embeddings", "BYTES")
        ],
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    
    successful_uploads = 0
    failed_uploads = 0
    
    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as source_file:
                job = client.load_table_from_file(
                    source_file, EMBEDDINGS_TABLE_ID, job_config=job_config
                )
                job.result()  # Wait for the job to complete
                successful_uploads += 1
                print(f"Successfully uploaded {file_path}")
                
                # Clean up the file after successful upload
                os.remove(file_path)
                
        except Exception as e:
            failed_uploads += 1
            print(f"Failed to upload {file_path}: {e}")
    
    print(f"Upload completed: {successful_uploads} successful, {failed_uploads} failed")
    elapsed_time = time.time() - start_time
    print(f"{elapsed_time:.2f}s")
    
    # Clean up empty directory if all files were processed
    if os.path.exists('embeddings_temp') and not os.listdir('embeddings_temp'):
        os.rmdir('embeddings_temp')

def main(max_embeddings=None):
    """
    Main function to generate and insert ProtBERT embeddings into embeddings table.
    
    Args:
        max_embeddings: Maximum number of embeddings to generate (for testing)
    """
    print(f"Starting embedding generation for table: {EMBEDDINGS_TABLE_ID}")
    print(f"Using device: {device}")
    if max_embeddings:
        print(f"Limiting to {max_embeddings} embeddings for testing")
    
    # Initialize BigQuery client
    client = bigquery.Client.from_service_account_json(key_path)
    
    # Get total count of proteins that need embeddings
    count_query = f"""
        SELECT COUNT(*) as total_count
        FROM `{TABLE_ID}` pdb
        WHERE NOT EXISTS (
            SELECT 1 FROM `{EMBEDDINGS_TABLE_ID}` emb 
            WHERE emb.id = pdb.id
        )
    """
    total_count = list(client.query(count_query).result())[0].total_count
    print(f"Found {total_count} proteins that need embeddings")
    
    if total_count == 0:
        print("All proteins already have embeddings!")
        return
    
    # Configuration
    batch_size = 256  # Increased from 32 - more GPU efficient
    upload_batch_size = 1  # Increased from 10 - fewer BigQuery jobs
    
    # Process in batches
    processed_count = 0
    successful_count = 0
    failed_count = 0
    saved_files = []
    
    # Get all protein IDs that need embeddings
    ids_query = f"""
        SELECT pdb.id 
        FROM `{TABLE_ID}` pdb
        WHERE NOT EXISTS (
            SELECT 1 FROM `{EMBEDDINGS_TABLE_ID}` emb 
            WHERE emb.id = pdb.id
        )
        ORDER BY pdb.id
    """
    
    protein_ids = [row.id for row in client.query(ids_query).result()]
    
    # Apply max_embeddings limit if specified
    if max_embeddings and len(protein_ids) > max_embeddings:
        protein_ids = protein_ids[:max_embeddings]
        print(f"Limited to {max_embeddings} proteins for testing")
    
    print(f"Processing {len(protein_ids)} proteins in batches of {batch_size}...")
    
    # Process in batches
    for i in tqdm(range(0, len(protein_ids), batch_size), desc="Processing batches"):
        batch_ids = protein_ids[i:i + batch_size]
        
        try:
            # Fetch sequences for this batch (all IDs in batch need embeddings)
            sequences_dict = fetch_sequences_batch(client, batch_ids)
            
            if not sequences_dict:
                print(f"Batch {i//batch_size + 1}: No sequences found")
                continue
            
            # Generate embeddings
            sequences = list(sequences_dict.values())
            protein_ids_batch = list(sequences_dict.keys())
            
            embeddings_batch = get_protbert_embedding_batch(sequences)
            
            # Prepare embeddings for saving
            embeddings_to_save = []
            for protein_id, embedding_bytes in zip(protein_ids_batch, embeddings_batch):
                embeddings_to_save.append({
                    'id': protein_id,
                    'embeddings': embedding_bytes
                })
            
            # Save embeddings to local file
            batch_num = i // batch_size + 1
            saved_file = save_embeddings_batch(embeddings_to_save, batch_num)
            
            if saved_file:
                saved_files.append(saved_file)
                processed_count += len(batch_ids)
                successful_count += len(embeddings_to_save)
                print(f"Batch {batch_num}: Saved {len(embeddings_to_save)} embeddings")
                
                # Upload files to BigQuery when we have enough
                if len(saved_files) >= upload_batch_size:
                    upload_files_to_bigquery(client, saved_files)
                    saved_files = []  # Clear the list after upload
            else:
                failed_count += len(batch_ids)
                print(f"Batch {batch_num}: Failed to save embeddings")
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            failed_count += len(batch_ids)
            continue
    
    # Upload any remaining files
    if saved_files:
        print(f"Uploading final {len(saved_files)} files...")
        upload_files_to_bigquery(client, saved_files)
    
    print(f"\nProcessing completed!")
    print(f"Total processed: {processed_count}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")

if __name__ == "__main__":
    # For testing, limit to 10 embeddings
    main(max_embeddings=251349)
    
    # For full run, use:
    # main() 