import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from collections import defaultdict

from siamese_inference import SiameseInference
from siamese_parquet_dataset import SiameseParquetDataset, siamese_collate_fn

# --- Configuration Variables ---
MODEL_PATH = '08.31.pth'  # Path to the trained model checkpoint
TEST_SAMPLES = 10000                           # Number of test samples
NUM_BATCHES = 300                              # Number of batches to test

# GCS Configuration
BUCKET_NAME = 'jx-compbio'
FOLDER = 'temp_folder/'
KEY_PATH = 'mit-primes-464001-bfa03c2c5999.json'
GCS_PROJECT = 'mit-primes-464001'


def get_length_bucket(length, bucket_size=200):
    """
    Get the bucket label for a given length.
    Buckets are (0,200], (200,400], (400,600], etc.
    """
    bucket_num = (length - 1) // bucket_size
    lower = bucket_num * bucket_size
    upper = (bucket_num + 1) * bucket_size
    return f"({lower},{upper}]"


def create_test_dataset(config, test_samples=10000):
    """
    Create a test dataset using SiameseParquetDataset.
    
    Args:
        config: Model configuration
        test_samples: Number of test samples
        
    Returns:
        dataset: Test dataset
    """
    dataset = SiameseParquetDataset(
        gcs_folder=f'gs://{BUCKET_NAME}/{FOLDER}',
        max_len=config['max_seq_len'],
        gcs_project=GCS_PROJECT,
        key_path=KEY_PATH
    )
    
    full_dataset_size = len(dataset)
    print(f"Full dataset size from GCS: {full_dataset_size} samples")
    
    # Limit the dataset to test_samples if needed
    if len(dataset) > test_samples:
        # Create a subset of the dataset
        indices = torch.randperm(len(dataset))[:test_samples].tolist()  # Convert to list of ints
        dataset = torch.utils.data.Subset(dataset, indices)
    
    return dataset


def test_model(model_path, test_samples=10000, num_batches=300):
    """
    Test the trained transformer model and time only the similarity search step.
    
    Args:
        model_path: Path to the trained model checkpoint
        test_samples: Number of test samples
        num_batches: Number of batches to test
    """
    # Initialize SiameseInference
    inference = SiameseInference(model_path)
    config = inference.config
    
    # Create test dataset
    test_dataset = create_test_dataset(config, test_samples)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        collate_fn=lambda batch: siamese_collate_fn(batch, config['max_seq_len']),
        num_workers=0
    )
    
    print(f"\nBenchmarking similarity search on {test_samples} samples...")
    print(f"Device: {inference.device}")
    print(f"Dataset size: {len(test_dataset)}")
    
    # Dictionary to store timing data by length bucket: {bucket_label: [times]}
    bucket_times = defaultdict(list)
    bucket_lengths = defaultdict(list)  # Store actual lengths for reference
    
    # Test on batches
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
        if batch_idx >= num_batches:
            break
            
        # Move data to device
        prot1_embeddings = batch['embeddings1'].to(inference.device)
        prot2_embeddings = batch['embeddings2'].to(inference.device)
        prot1_mask = batch['mask1'].to(inference.device)
        prot2_mask = batch['mask2'].to(inference.device)
        
        batch_size = prot1_embeddings.size(0)
        
        # Process each sample in the batch individually to get per-sample timing
        for i in range(batch_size):
            # Get sequence length from mask (number of non-padded positions)
            seq_len1 = int(prot1_mask[i].sum().item())
            seq_len2 = int(prot2_mask[i].sum().item())
            
            # Use the average length or max length for bucketing
            avg_length = (seq_len1 + seq_len2) // 2
            
            # Skip proteins over 1000 amino acids
            if avg_length > 1000:
                continue
            
            bucket = get_length_bucket(avg_length)
            
            # Extract single sample
            prot1_emb = prot1_embeddings[i:i+1]  # Keep batch dimension
            prot2_emb = prot2_embeddings[i:i+1]
            prot1_m = prot1_mask[i:i+1]
            prot2_m = prot2_mask[i:i+1]
            
            # Time only the similarity search step
            start_time = time.time()
            predicted_scores = inference.compute_similarity_scores(
                prot1_emb, prot2_emb, prot1_m, prot2_m
            )
            elapsed_time = time.time() - start_time
            
            bucket_times[bucket].append(elapsed_time)
            bucket_lengths[bucket].append(avg_length)
    
    # Print summary statistics grouped by bucket
    if bucket_times:
        print(f"\n{'='*70}")
        print(f"SIMILARITY SEARCH TIMING STATISTICS BY AMINO ACID LENGTH BUCKET")
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
        print("\nNo successful similarity search calculations to time.")


def main():
    """
    Main function to benchmark similarity search timing.
    """
    # Use the configuration variables defined at the top
    test_model(MODEL_PATH, TEST_SAMPLES, NUM_BATCHES)


if __name__ == "__main__":
    main()

