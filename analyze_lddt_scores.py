#!/usr/bin/env python3
"""
Generate predicted and true lDDT scores from an info CSV file.
Creates a new CSV with columns: PID, sequence, mutated_sequence, predicted_lddt_scores, true_lddt_scores, description
Also computes MAE, standard deviation, and other statistics, and plots the loss distribution.
"""
import sys
import numpy as np
import pandas as pd
import base64
import csv
import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer
from siamese_transformer_model import SiameseTransformerNet
from lddt_weighted import LDDTCalculatorWeighted
from pathlib import Path
import matplotlib.pyplot as plt
import ast

if len(sys.argv) != 3:
    print("Usage: python analyze_lddt_scores.py <input_csv> <model_checkpoint>")
    sys.exit(1)

input_csv = sys.argv[1]
model_checkpoint = sys.argv[2]
output_csv = input_csv.replace('.csv', '_lddt_scores.csv')

# Model configuration
MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
PAD_LEN = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def decode_coords(b64str):
    """Decode base64 coordinates."""
    arr = np.frombuffer(base64.b64decode(b64str), dtype=np.float32)
    return arr.reshape(-1, 3)

def get_prott5_embedding(seq, pad_len=PAD_LEN):
    """Get ProtT5 embeddings for a sequence."""
    seq_spaced = ' '.join(list(seq))
    inputs = tokenizer(seq_spaced, return_tensors='pt', padding='max_length', truncation=True, max_length=pad_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = t5_model(**inputs)
        emb = outputs.last_hidden_state[0]  # (seq_len, 1024)
    if emb.size(0) < pad_len:
        pad = torch.zeros(pad_len - emb.size(0), emb.size(1), device=emb.device)
        emb = torch.cat([emb, pad], dim=0)
    else:
        emb = emb[:pad_len]
    attn_mask = inputs['attention_mask'][0]
    return emb, attn_mask

def load_trained_model(model_path, device):
    """Load the trained Siamese transformer model."""
    siamese_checkpoint = torch.load(model_path, map_location=device)
    siamese_config = siamese_checkpoint['config']
    siamese_model = SiameseTransformerNet(
        input_dim=siamese_config['prottrans_dim'],
        hidden_dim=siamese_config['hidden_dim'],
        output_dim=siamese_config['output_dim'],
        nhead=siamese_config['nhead'],
        num_layers=siamese_config['num_layers'],
        dropout=siamese_config['dropout'],
        max_seq_len=siamese_config['max_seq_len']
    )
    siamese_model.load_state_dict(siamese_checkpoint['model_state_dict'])
    siamese_model = siamese_model.to(device)
    siamese_model.eval()
    return siamese_model

def get_predicted_lddt_scores(wild_seq, mutant_seq, model, device):
    """Get predicted lDDT scores using the Siamese model."""
    # Get embeddings
    emb_wt, mask_wt = get_prott5_embedding(wild_seq)
    emb_mut, mask_mut = get_prott5_embedding(mutant_seq)
    emb_wt = emb_wt.to(device)
    emb_mut = emb_mut.to(device)
    mask_wt = mask_wt.to(device)
    mask_mut = mask_mut.to(device)

    # Get model outputs
    model.eval()
    with torch.no_grad():
        emb_wt = emb_wt.unsqueeze(0)
        emb_mut = emb_mut.unsqueeze(0)
        mask_wt = mask_wt.unsqueeze(0)
        mask_mut = mask_mut.unsqueeze(0)
        new_emb1, new_emb2, global_emb1, global_emb2 = model(emb_wt, emb_mut, mask_wt, mask_mut)

    # Per-residue similarity (normalized to [0,1] range)
    new_emb1 = new_emb1.squeeze(0)  # [seq_len, output_dim]
    new_emb2 = new_emb2.squeeze(0)
    per_res_sim = F.cosine_similarity(new_emb1, new_emb2, dim=1).cpu().numpy()  # [seq_len]
    # Normalize from [-1,1] to [0,1] range
    per_res_sim = (per_res_sim + 1) / 2
    # Truncate to wild_seq length
    per_res_sim = per_res_sim[:len(wild_seq)]
    
    return per_res_sim

def compute_statistics(predicted_scores, true_scores):
    """Compute various statistics between predicted and true scores."""
    # Ensure same length
    min_len = min(len(predicted_scores), len(true_scores))
    pred = predicted_scores[:min_len]
    true = true_scores[:min_len]
    
    # Compute errors
    errors = pred - true
    abs_errors = np.abs(errors)
    
    # Statistics
    mae = np.mean(abs_errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    std_error = np.std(errors)
    std_abs_error = np.std(abs_errors)
    
    # Correlation
    correlation = np.corrcoef(pred, true)[0, 1]
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'std_error': std_error,
        'std_abs_error': std_abs_error,
        'correlation': correlation,
        'errors': errors,
        'abs_errors': abs_errors
    }

def plot_loss_distribution(all_errors, all_abs_errors, all_true_scores, all_predicted_scores, output_prefix):
    """Plot the distribution of errors and absolute errors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Error distribution
    axes[0].hist(all_errors, bins=200, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Distribution of Errors (Predicted - True)')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim(-0.0075, 0.0075)  # Set x-axis limits
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Absolute error distribution
    axes[1].hist(all_abs_errors, bins=200, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('Distribution of Absolute Errors')
    axes[1].set_xlabel('Absolute Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim(0, 0.0075)  # Set x-axis limits
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_loss_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss distribution plot to: {output_prefix}_loss_distribution.png")



# Load models
print("Loading ProtT5 model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
t5_model = T5EncoderModel.from_pretrained(MODEL_NAME).to(device)
t5_model.eval()

print(f"Loading Siamese model from {model_checkpoint}...")
siamese_model = load_trained_model(model_checkpoint, device)

print("Loading lDDT calculator...")
lddt_calculator = LDDTCalculatorWeighted(weight_exponent=3.0)

# Read input CSV
df = pd.read_csv(input_csv)
print(f"Processing {len(df)} protein pairs...")

# Initialize lists to collect all data for statistics
all_predicted_scores = []
all_true_scores = []
all_errors = []
all_abs_errors = []
protein_stats = []  # List to store per-protein statistics
processed_variants = 0  # Counter for processed variants

# Create output CSV
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ["PID", "sequence", "mutated_sequence", "predicted_lddt_scores", "true_lddt_scores", "description"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for idx, row in df.iterrows():
        try:
            print(f"\nProcessing {row['PID']} ({idx+1}/{len(df)})...")
            
            # Decode coordinates
            coords = decode_coords(row["coords"])
            mutant_coords = decode_coords(row["mutant_coords"])
            
            if coords.shape != mutant_coords.shape:
                print(f"  Skipping: coordinate shapes don't match ({coords.shape} vs {mutant_coords.shape})")
                continue
            
            # Get sequences
            wild_seq = row["sequence"]
            mutant_seq = row["mutated_sequence"]
            
            # Get predicted lDDT scores
            predicted_lddt_scores = get_predicted_lddt_scores(wild_seq, mutant_seq, siamese_model, device)
            
            # Get true lDDT scores
            true_lddt_scores = lddt_calculator.calculate_lddt(coords, mutant_coords)
            
            # Compute statistics for this protein
            stats = compute_statistics(predicted_lddt_scores, true_lddt_scores)
            
            # Store per-protein statistics
            protein_stats.append({
                'PID': row['PID'],
                'sequence_length': len(wild_seq),
                'predicted_mean_lddt': np.mean(predicted_lddt_scores),
                'true_mean_lddt': np.mean(true_lddt_scores),
                'mae': stats['mae'],
                'mse': stats['mse'],
                'rmse': stats['rmse'],
                'std_error': stats['std_error'],
                'std_abs_error': stats['std_abs_error'],
                'correlation': stats['correlation'],
                'predicted_lddt_scores': list(predicted_lddt_scores),
                'true_lddt_scores': list(true_lddt_scores),
                'description': row.get("description", "")
            })
            
            # Collect data for overall statistics
            all_predicted_scores.extend(predicted_lddt_scores[:len(true_lddt_scores)])
            all_true_scores.extend(true_lddt_scores[:len(predicted_lddt_scores)])
            all_errors.extend(stats['errors'])
            all_abs_errors.extend(stats['abs_errors'])
            processed_variants += 1
            
            # Write to CSV
            writer.writerow({
                "PID": row["PID"],
                "sequence": wild_seq,
                "mutated_sequence": mutant_seq,
                "predicted_lddt_scores": list(predicted_lddt_scores),
                "true_lddt_scores": list(true_lddt_scores),
                "description": row.get("description", "")
            })
            
            # Print summary with statistics
            print(f"  Predicted mean lDDT: {np.mean(predicted_lddt_scores):.4f}")
            print(f"  True mean lDDT: {np.mean(true_lddt_scores):.4f}")
            print(f"  MAE: {stats['mae']:.4f}")
            print(f"  RMSE: {stats['rmse']:.4f}")
            print(f"  Correlation: {stats['correlation']:.4f}")
            
        except Exception as e:
            print(f"Error processing {row['PID']}: {e}")
            continue

# Convert to numpy arrays for plotting
all_predicted_scores = np.array(all_predicted_scores)
all_true_scores = np.array(all_true_scores)
all_errors = np.array(all_errors)
all_abs_errors = np.array(all_abs_errors)

# Save per-protein statistics to CSV
protein_stats_csv = output_csv.replace('.csv', '_protein_statistics.csv')
protein_stats_df = pd.DataFrame(protein_stats)
protein_stats_df.to_csv(protein_stats_csv, index=False)
print(f"Saved per-protein statistics to: {protein_stats_csv}")

# Compute and print overall statistics (not saved)
overall_stats = compute_statistics(all_predicted_scores, all_true_scores)

print(f"\n=== OVERALL STATISTICS ===")
print(f"Total protein variants processed: {processed_variants}")
print(f"Total residues processed: {len(all_predicted_scores)}")
print(f"MAE: {overall_stats['mae']:.4f}")
print(f"MSE: {overall_stats['mse']:.4f}")
print(f"RMSE: {overall_stats['rmse']:.4f}")
print(f"Standard Deviation of Errors: {overall_stats['std_error']:.4f}")
print(f"Standard Deviation of Absolute Errors: {overall_stats['std_abs_error']:.4f}")
print(f"Correlation: {overall_stats['correlation']:.4f}")

# Create loss distribution plot
output_prefix = output_csv.replace('.csv', '')
plot_loss_distribution(all_errors, all_abs_errors, all_true_scores, all_predicted_scores, output_prefix)

print(f"\nGenerated lDDT comparison CSV: {output_csv}")