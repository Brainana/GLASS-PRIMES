import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from siamese_inference import SiameseInference
from siamese_dataset import SiameseProteinDataset

# --- BigQuery setup ---
from google.cloud import bigquery
key_path = "mit-primes-464001-bfa03c2c5999.json"
bq_client = bigquery.Client.from_service_account_json(key_path)
ground_truth_table = "mit-primes-464001.primes_data.ground_truth_scores"

# --- Configuration Variables ---
MODEL_PATH = 'siamese_transformer_best.pth'  # Path to the trained model checkpoint
TEST_SAMPLES = 100                           # Number of test samples
NUM_BATCHES = 3                              # Number of batches to test


def create_test_dataset(config, test_samples=100):
    """
    Create a test dataset with a different range of samples.
    
    Args:
        config: Model configuration
        test_samples: Number of test samples
        
    Returns:
        dataset: Test dataset
    """
    dataset = SiameseProteinDataset(
        max_samples=test_samples,
        bq_client=bq_client,
        ground_truth_table=ground_truth_table,
        max_seq_len=config['max_seq_len'],
        prottrans_dim=config['prottrans_dim'],
        data_batch_size=config.get('data_batch_size', 32)
    )
    
    return dataset


def evaluate_predictions(predicted_scores, true_tm_scores, true_lddt_scores, seqxA_list, seqM_list, seqyA_list):
    """
    Evaluate predictions against ground truth.
    
    Args:
        predicted_scores: Model predictions (global_sim, residue_sims)
        true_tm_scores: Ground truth TM scores
        true_lddt_scores: Ground truth LDDT scores
        seqxA_list, seqM_list, seqyA_list: Alignment sequences
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    global_sims, residue_sims = predicted_scores
    
    metrics = {
        'tm_correlation': 0.0,
        'tm_mae': 0.0,
        'lddt_correlation': 0.0,
        'lddt_mae': 0.0,
        'total_pairs': len(true_tm_scores),
        'valid_pairs': 0
    }
    
    tm_predictions = []
    tm_ground_truth = []
    lddt_predictions = []
    lddt_ground_truth = []
    
    for i in range(len(true_tm_scores)):
        tm_score = true_tm_scores[i].item()
        
        # Skip pairs with very low TM scores (too chaotic)
        if tm_score < 0.1:
            continue
            
        metrics['valid_pairs'] += 1
        
        # Global TM prediction
        pred_tm = global_sims[i].item()
        tm_predictions.append(pred_tm)
        tm_ground_truth.append(tm_score)
        
        # Per-residue LDDT prediction (for high-quality alignments)
        if tm_score >= 0.4 and residue_sims is not None:
            # Parse alignment to get aligned residues
            from alignment_utils import parse_alignment_from_sequences
            alignment = parse_alignment_from_sequences(
                seqxA_list[i], seqM_list[i], seqyA_list[i]
            )
            
            if alignment:
                # Get actual sequence lengths
                actual_len1 = len(seqxA_list[i].replace('-', ''))
                actual_len2 = len(seqyA_list[i].replace('-', ''))
                
                # Filter valid alignment pairs
                valid_pairs = []
                for m_idx, r_idx in alignment:
                    if m_idx < actual_len1 and r_idx < actual_len2 and m_idx < residue_sims.size(1) and r_idx < residue_sims.size(1):
                        valid_pairs.append((m_idx, r_idx))
                
                if valid_pairs:
                    # Get predicted and true LDDT scores for aligned residues
                    model_indices, ref_indices = zip(*valid_pairs)
                    pred_lddt = residue_sims[i, list(model_indices)].cpu().numpy()
                    true_lddt = true_lddt_scores[i, list(model_indices)].cpu().numpy()
                    
                    lddt_predictions.extend(pred_lddt)
                    lddt_ground_truth.extend(true_lddt)
    
    # Compute TM metrics
    if tm_predictions:
        tm_corr = np.corrcoef(tm_predictions, tm_ground_truth)[0, 1]
        tm_mae = np.mean(np.abs(np.array(tm_predictions) - np.array(tm_ground_truth)))
        metrics['tm_correlation'] = tm_corr if not np.isnan(tm_corr) else 0.0
        metrics['tm_mae'] = tm_mae
    
    # Compute LDDT metrics
    if lddt_predictions:
        lddt_corr = np.corrcoef(lddt_predictions, lddt_ground_truth)[0, 1]
        lddt_mae = np.mean(np.abs(np.array(lddt_predictions) - np.array(lddt_ground_truth)))
        metrics['lddt_correlation'] = lddt_corr if not np.isnan(lddt_corr) else 0.0
        metrics['lddt_mae'] = lddt_mae
    
    return metrics


def test_model(model_path, test_samples=100, num_batches=3):
    """
    Test the trained transformer model on new data using SiameseInference class.
    
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
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nTesting transformer model on {test_samples} samples...")
    print(f"Device: {inference.device}")
    
    all_metrics = []
    
    # Test on batches
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing batches")):
        if batch_idx >= num_batches:
            break
            
        # Move data to device
        prot1_embeddings = batch['prot1_embeddings'].to(inference.device)
        prot2_embeddings = batch['prot2_embeddings'].to(inference.device)
        prot1_mask = batch['prot1_mask'].to(inference.device)
        prot2_mask = batch['prot2_mask'].to(inference.device)
        tm_scores = batch['tm_score'].to(inference.device)
        lddt_scores = batch['lddt_scores'].to(inference.device)
        seqxA_list = batch['seqxA']
        seqM_list = batch['seqM']
        seqyA_list = batch['seqyA']
        
        # Get predictions using SiameseInference
        predicted_scores = inference.compute_similarity_scores(
            prot1_embeddings, prot2_embeddings, prot1_mask, prot2_mask
        )
        
        # Evaluate predictions
        metrics = evaluate_predictions(
            predicted_scores, tm_scores, lddt_scores, 
            seqxA_list, seqM_list, seqyA_list
        )
        
        all_metrics.append(metrics)
        
        print(f"\nBatch {batch_idx + 1} Results:")
        print(f"  Valid pairs: {metrics['valid_pairs']}/{metrics['total_pairs']}")
        print(f"  TM Correlation: {metrics['tm_correlation']:.4f}")
        print(f"  TM MAE: {metrics['tm_mae']:.4f}")
        print(f"  LDDT Correlation: {metrics['lddt_correlation']:.4f}")
        print(f"  LDDT MAE: {metrics['lddt_mae']:.4f}")
        
        # Show some example predictions
        if metrics['valid_pairs'] > 0:
            global_sims, _ = predicted_scores
            print(f"  Sample TM predictions:")
            for i in range(min(3, len(global_sims))):
                if tm_scores[i].item() >= 0.1:
                    print(f"    Pair {i}: Pred={global_sims[i].item():.3f}, True={tm_scores[i].item():.3f}")
    
    # Compute average metrics
    if all_metrics:
        avg_metrics = {}
        for key in ['tm_correlation', 'tm_mae', 'lddt_correlation', 'lddt_mae']:
            values = [m[key] for m in all_metrics if m['valid_pairs'] > 0]
            avg_metrics[key] = np.mean(values) if values else 0.0
        
        print(f"\n{'='*50}")
        print(f"AVERAGE RESULTS (TRANSFORMER):")
        print(f"  TM Correlation: {avg_metrics['tm_correlation']:.4f}")
        print(f"  TM MAE: {avg_metrics['tm_mae']:.4f}")
        print(f"  LDDT Correlation: {avg_metrics['lddt_correlation']:.4f}")
        print(f"  LDDT MAE: {avg_metrics['lddt_mae']:.4f}")
        print(f"{'='*50}")
        
        # Save results
        results = {
            'model_path': model_path,
            'model_type': 'transformer',
            'test_samples': test_samples,
            'num_batches': num_batches,
            'average_metrics': avg_metrics,
            'batch_metrics': all_metrics
        }


def main():
    """
    Main function to test the trained model.
    """
    # Use the configuration variables defined at the top
    test_model(MODEL_PATH, TEST_SAMPLES, NUM_BATCHES)


if __name__ == "__main__":
    main() 