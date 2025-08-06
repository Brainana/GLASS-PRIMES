#!/usr/bin/env python3
"""
Evaluate lDDT scores using parquet files.
Loads a trained model and evaluates mean absolute loss between true and predicted lDDT scores.
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from siamese_transformer_model import SiameseTransformerNet
from siamese_parquet_dataset import SiameseParquetDataset, siamese_collate_fn


def load_trained_model(model_path, device, config):
    """Load the trained Siamese transformer model."""
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('config', config)
    
    model = SiameseTransformerNet(
        input_dim=model_config['prottrans_dim'],
        hidden_dim=model_config['hidden_dim'],
        output_dim=model_config['output_dim'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        max_seq_len=model_config['max_seq_len']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model and compute mean absolute loss.
    
    Args:
        model: Trained Siamese transformer model
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predicted_scores = []
    all_true_scores = []
    all_errors = []
    all_abs_errors = []
    
    # TM score evaluation
    all_predicted_tm_scores = []
    all_true_tm_scores = []
    all_tm_errors = []
    all_tm_abs_errors = []
    
    print("Evaluating model...")
    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                batch_count += 1
                print(f"Processing batch {batch_count}/{len(dataloader)}")
                
                # Move batch to device
                emb1 = batch['embeddings1'].to(device)
                emb2 = batch['embeddings2'].to(device)
                mask1 = batch['mask1'].to(device)
                mask2 = batch['mask2'].to(device)
                true_lddt = batch['lddt_scores'].to(device)  # True lDDT scores from parquet files
                tm_scores = batch['tm_score'].cpu().numpy()  # True TM scores from parquet files
                
                # Get model predictions
                new_emb1, new_emb2, global_emb1, global_emb2 = model(emb1, emb2, mask1, mask2)
                
                # Compute TM scores from global embeddings
                global_sim = F.cosine_similarity(global_emb1, global_emb2, dim=1)  # [batch]
                predicted_tm_scores = (global_sim + 1) / 2  # Normalize from [-1,1] to [0,1] range
                predicted_tm_scores = predicted_tm_scores.cpu().numpy()
                
                # Compute per-residue similarity (normalized to [0,1] range)
                new_emb1 = new_emb1.view(new_emb1.size(0), -1, new_emb1.size(-1))  # [batch, seq_len, dim]
                new_emb2 = new_emb2.view(new_emb2.size(0), -1, new_emb2.size(-1))
                
                # Compute cosine similarity for each residue
                per_res_sim = F.cosine_similarity(new_emb1, new_emb2, dim=2)  # [batch, seq_len]
                
                # Normalize from [-1,1] to [0,1] range
                per_res_sim = (per_res_sim + 1) / 2
                
                # Convert to numpy for analysis
                predicted_scores = per_res_sim.cpu().numpy()
                true_scores = true_lddt.cpu().numpy()
                
                # Handle variable sequence lengths and alignment
                for i in range(predicted_scores.shape[0]):
                    # Skip pairs with TM score < 0.7 (not meaningful for lDDT comparison)
                    if tm_scores[i] < 0.7:
                        continue
                        
                    # Get alignment sequences
                    seqxA = batch['seqxA'][i]  # Reference sequence
                    seqyA = batch['seqyA'][i]  # Query sequence  
                    seqM = batch['seqM'][i]    # Alignment mask
                    
                    # Get actual sequence length (non-padded)
                    seq_len = len(seqxA)  # Use reference sequence length
                    
                    # Get the size of the score arrays for this sample
                    pred_size = predicted_scores[i].shape[0]
                    true_size = true_scores[i].shape[0]
                    
                    # Only evaluate positions that are aligned (not gaps) and within bounds
                    aligned_positions = []
                    for j, (ref_char, query_char, mask_char) in enumerate(zip(seqxA, seqyA, seqM)):
                        # Make sure we don't exceed the bounds of our score arrays
                        if j >= pred_size or j >= true_size:
                            break
                        # Only include positions where both sequences have residues (not gaps)
                        if ref_char != '-' and query_char != '-' and mask_char != '-':
                            aligned_positions.append(j)
                    
                    if len(aligned_positions) == 0:
                        print(f"Warning: No aligned positions found for sample {i}")
                        continue
                    
                    # Extract scores for aligned positions only
                    pred_seq = predicted_scores[i][aligned_positions]
                    true_seq = true_scores[i][aligned_positions]
                    
                    # Print predicted and true scores for the first pair
                    if batch_count == 1 and i == 0:
                        print(f"=== DEBUG INFO ===")
                        print(f"TM Score: {tm_scores[i]:.6f}")
                        print(f"Number of aligned positions: {len(aligned_positions)}")
                        print(f"True scores shape: {true_scores[i].shape}")
                        print(f"True scores (first 10): {true_scores[i][:10]}")
                        print(f"True scores min/max: {true_scores[i].min():.6f}/{true_scores[i].max():.6f}")
                        print(f"True scores mean: {true_scores[i].mean():.6f}")
                        print(f"Aligned true scores min/max: {true_seq.min():.6f}/{true_seq.max():.6f}")
                        print(f"Aligned true scores mean: {true_seq.mean():.6f}")
                        print(f"Number of zero true scores: {np.sum(true_seq == 0)}")
                        print(f"Number of non-zero true scores: {np.sum(true_seq != 0)}")
                        print(f"================================\n")
                        
                        # Print every aligned lDDT score difference
                        print(f"=== ALL ALIGNED lDDT SCORE DIFFERENCES ===")
                        for pos_idx, (pred_score, true_score) in enumerate(zip(pred_seq, true_seq)):
                            diff = pred_score - true_score
                            abs_diff = abs(diff)
                            print(f"Position {aligned_positions[pos_idx]}: Pred={pred_score:.6f}, True={true_score:.6f}, Diff={diff:.6f}, Abs_Diff={abs_diff:.6f}")
                        print(f"==========================================\n")
                    
                    # Compute errors
                    errors = pred_seq - true_seq
                    abs_errors = np.abs(errors)
                    
                    # Store lDDT results
                    all_predicted_scores.extend(pred_seq)
                    all_true_scores.extend(true_seq)
                    all_errors.extend(errors)
                    all_abs_errors.extend(abs_errors)
                    
                    # Store TM score results (for all pairs, not just TM >= 0.7)
                    all_predicted_tm_scores.append(predicted_tm_scores[i])
                    all_true_tm_scores.append(tm_scores[i])
                    tm_error = predicted_tm_scores[i] - tm_scores[i]
                    tm_abs_error = abs(tm_error)
                    all_tm_errors.append(tm_error)
                    all_tm_abs_errors.append(tm_abs_error)
                
                print(f"  Batch {batch_count} completed successfully")
                
            except Exception as e:
                print(f"Error processing batch {batch_count}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"Evaluation completed. Processed {batch_count} batches.")
    
    # Convert to numpy arrays
    all_predicted_scores = np.array(all_predicted_scores)
    all_true_scores = np.array(all_true_scores)
    all_errors = np.array(all_errors)
    all_abs_errors = np.array(all_abs_errors)
    
    # Convert TM score arrays
    all_predicted_tm_scores = np.array(all_predicted_tm_scores)
    all_true_tm_scores = np.array(all_true_tm_scores)
    all_tm_errors = np.array(all_tm_errors)
    all_tm_abs_errors = np.array(all_tm_abs_errors)
    
    # Compute lDDT statistics
    mae = np.mean(all_abs_errors)
    mse = np.mean(all_errors ** 2)
    rmse = np.sqrt(mse)
    std_error = np.std(all_errors)
    std_abs_error = np.std(all_abs_errors)
    correlation = np.corrcoef(all_predicted_scores, all_true_scores)[0, 1]
    
    # Compute TM score statistics
    tm_mae = np.mean(all_tm_abs_errors)
    tm_mse = np.mean(all_tm_errors ** 2)
    tm_rmse = np.sqrt(tm_mse)
    tm_std_error = np.std(all_tm_errors)
    tm_std_abs_error = np.std(all_tm_abs_errors)
    tm_correlation = np.corrcoef(all_predicted_tm_scores, all_true_tm_scores)[0, 1]
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'std_error': std_error,
        'std_abs_error': std_abs_error,
        'correlation': correlation,
        'total_residues': len(all_predicted_scores),
        'predicted_scores': all_predicted_scores,
        'true_scores': all_true_scores,
        'errors': all_errors,
        'abs_errors': all_abs_errors,
        # TM score statistics
        'tm_mae': tm_mae,
        'tm_mse': tm_mse,
        'tm_rmse': tm_rmse,
        'tm_std_error': tm_std_error,
        'tm_std_abs_error': tm_std_abs_error,
        'tm_correlation': tm_correlation,
        'total_pairs': len(all_predicted_tm_scores),
        'predicted_tm_scores': all_predicted_tm_scores,
        'true_tm_scores': all_true_tm_scores,
        'tm_errors': all_tm_errors,
        'tm_abs_errors': all_tm_abs_errors
    }


def plot_evaluation_results(results, output_prefix):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # === lDDT Score Plots ===
    # Error distribution
    axes[0, 0].hist(results['errors'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('lDDT Error Distribution')
    axes[0, 0].set_xlabel('Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_xlim(-0.3, 0.3)
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Absolute error distribution
    axes[0, 1].hist(results['abs_errors'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('lDDT Absolute Error Distribution')
    axes[0, 1].set_xlabel('Absolute Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xlim(0, 0.3)
    
    # lDDT Error vs True score scatter
    axes[1, 0].scatter(results['true_scores'], results['errors'], alpha=0.5, s=1)
    axes[1, 0].set_title('lDDT Error vs True Score')
    axes[1, 0].set_xlabel('True lDDT Score')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # === TM Score Plots ===
    # TM Error distribution
    axes[0, 2].hist(results['tm_errors'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].set_title('TM Score Error Distribution')
    axes[0, 2].set_xlabel('Error')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_xlim(-0.5, 0.5)
    axes[0, 2].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # TM Absolute error distribution
    axes[1, 1].hist(results['tm_abs_errors'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('TM Score Absolute Error Distribution')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_xlim(0, 0.5)
    
    # TM Error vs True score scatter
    axes[1, 2].scatter(results['true_tm_scores'], results['tm_errors'], alpha=0.5, s=1)
    axes[1, 2].set_title('TM Error vs True Score')
    axes[1, 2].set_xlabel('True TM Score')
    axes[1, 2].set_ylabel('Error')
    axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation plot to: {output_prefix}_evaluation_results.png")


def main():
    # ===== CONFIGURABLE VARIABLES =====
    MODEL_PATH = '07.26-2000parquet.pth'  # Path to trained model checkpoint
    GCS_FOLDER = 'gs://primes-bucket/testing_data2/'  # GCS folder with parquet files
    GCS_PROJECT = 'mit-primes-464001'  # GCS project ID
    KEY_PATH = 'mit-primes-464001-bfa03c2c5999.json'  # Path to GCS service account key
    BATCH_SIZE = 32  # Batch size for evaluation
    MAX_SEQ_LEN = 300  # Maximum sequence length
    OUTPUT_PREFIX = 'evaluation'  # Output prefix for results
    NUM_WORKERS = 0  # Number of data loading workers (0 for debugging)
    
    # Model configuration
    MODEL_CONFIG = {
        'prottrans_dim': 1024,
        'hidden_dim': 1024,
        'output_dim': 512,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1,
        'max_seq_len': MAX_SEQ_LEN
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_trained_model(MODEL_PATH, device, MODEL_CONFIG)
    
    # Create dataset with only first 3 parquet files
    print(f"Loading dataset from {GCS_FOLDER}...")
    dataset = SiameseParquetDataset(
        gcs_folder=GCS_FOLDER,
        max_len=MAX_SEQ_LEN,
        gcs_project=GCS_PROJECT,
        key_path=KEY_PATH
    )
    
    # Limit to only first 3 parquet files
    MAX_PARQUET_FILES = 3
    if len(dataset.all_parquet_uris) > MAX_PARQUET_FILES:
        print(f"Limiting to first {MAX_PARQUET_FILES} parquet files (out of {len(dataset.all_parquet_uris)})")
        dataset.all_parquet_uris = dataset.all_parquet_uris[:MAX_PARQUET_FILES]
        # Rebuild the index cache with limited files
        dataset._build_shuffled_index_cache()
    else:
        print(f"Using all {len(dataset.all_parquet_uris)} available parquet files")
        dataset._build_shuffled_index_cache()
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: siamese_collate_fn(batch, MAX_SEQ_LEN),
        num_workers=NUM_WORKERS
    )
    
    print(f"DataLoader created successfully with {len(dataloader)} batches")
    
    # Evaluate model
    results = evaluate_model(model, dataloader, device)
    
    # Print results
    print(f"\n=== lDDT EVALUATION RESULTS ===")
    print(f"Total residues evaluated: {results['total_residues']}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Standard Deviation of Errors: {results['std_error']:.4f}")
    print(f"Standard Deviation of Absolute Errors: {results['std_abs_error']:.4f}")
    print(f"Correlation: {results['correlation']:.4f}")
    
    print(f"\n=== TM SCORE EVALUATION RESULTS ===")
    print(f"Total pairs evaluated: {results['total_pairs']}")
    print(f"TM MAE: {results['tm_mae']:.4f}")
    print(f"TM MSE: {results['tm_mse']:.4f}")
    print(f"TM RMSE: {results['tm_rmse']:.4f}")
    print(f"TM Standard Deviation of Errors: {results['tm_std_error']:.4f}")
    print(f"TM Standard Deviation of Absolute Errors: {results['tm_std_abs_error']:.4f}")
    print(f"TM Correlation: {results['tm_correlation']:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame([
        # lDDT metrics
        {
            'metric': 'total_residues',
            'value': results['total_residues'],
            'description': 'Total number of residues evaluated'
        }, {
            'metric': 'mae',
            'value': results['mae'],
            'description': 'lDDT Mean Absolute Error'
        }, {
            'metric': 'mse',
            'value': results['mse'],
            'description': 'lDDT Mean Squared Error'
        }, {
            'metric': 'rmse',
            'value': results['rmse'],
            'description': 'lDDT Root Mean Squared Error'
        }, {
            'metric': 'std_error',
            'value': results['std_error'],
            'description': 'lDDT Standard Deviation of Errors'
        }, {
            'metric': 'std_abs_error',
            'value': results['std_abs_error'],
            'description': 'lDDT Standard Deviation of Absolute Errors'
        }, {
            'metric': 'correlation',
            'value': results['correlation'],
            'description': 'lDDT Pearson Correlation Coefficient'
        },
        # TM score metrics
        {
            'metric': 'total_pairs',
            'value': results['total_pairs'],
            'description': 'Total number of protein pairs evaluated'
        }, {
            'metric': 'tm_mae',
            'value': results['tm_mae'],
            'description': 'TM Score Mean Absolute Error'
        }, {
            'metric': 'tm_mse',
            'value': results['tm_mse'],
            'description': 'TM Score Mean Squared Error'
        }, {
            'metric': 'tm_rmse',
            'value': results['tm_rmse'],
            'description': 'TM Score Root Mean Squared Error'
        }, {
            'metric': 'tm_std_error',
            'value': results['tm_std_error'],
            'description': 'TM Score Standard Deviation of Errors'
        }, {
            'metric': 'tm_std_abs_error',
            'value': results['tm_std_abs_error'],
            'description': 'TM Score Standard Deviation of Absolute Errors'
        }, {
            'metric': 'tm_correlation',
            'value': results['tm_correlation'],
            'description': 'TM Score Pearson Correlation Coefficient'
        }
    ])
    
    results_csv = f"{OUTPUT_PREFIX}_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Saved results to: {results_csv}")
    
    # Create plots
    plot_evaluation_results(results, OUTPUT_PREFIX)
    
    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    main() 