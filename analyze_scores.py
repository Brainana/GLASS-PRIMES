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
from siamese_transformer_model_v1 import SiameseTransformerNetV1
from lddt_weighted import LDDTCalculatorWeighted
from pathlib import Path
import matplotlib.pyplot as plt
import ast
import subprocess
import tempfile
import os
import glob
import re
import tmtools
from embed_structure_model import trans_basic_block, trans_basic_block_Config

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

def get_predicted_tm_score(wild_seq, mutant_seq, model, device):
    """Get predicted TM score using the Siamese model."""
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

    # Compute TM score from global embeddings
    global_sim = F.cosine_similarity(global_emb1, global_emb2, dim=1)  # [batch]
    predicted_tm_score = (global_sim + 1) / 2  # Normalize from [-1,1] to [0,1] range
    predicted_tm_score = predicted_tm_score.cpu().numpy()[0]  # Get scalar value
    
    return predicted_tm_score

def calculate_tm_score_with_tmvec(wild_seq, mutant_seq, tmvec_model):
    """Calculate TM-score using tm_vec model."""
    try:
        # Get ProtT5 embeddings
        emb_wt, mask_wt = get_prott5_embedding(wild_seq)
        emb_mut, mask_mut = get_prott5_embedding(mutant_seq)
        emb_wt = emb_wt.to(device)
        emb_mut = emb_mut.to(device)
        mask_wt = mask_wt.to(device)
        mask_mut = mask_mut.to(device)
        
        # Add batch dimension
        emb_wt = emb_wt.unsqueeze(0)
        emb_mut = emb_mut.unsqueeze(0)
        mask_wt = mask_wt.unsqueeze(0)
        mask_mut = mask_mut.unsqueeze(0)
        
        # Get tm_vec embeddings
        tmvec_model.eval()
        with torch.no_grad():
            out1_tmvec = tmvec_model.forward(emb_wt, src_mask=None, src_key_padding_mask=(mask_wt == 0))
            out2_tmvec = tmvec_model.forward(emb_mut, src_mask=None, src_key_padding_mask=(mask_mut == 0))
            
            # Calculate cosine similarity
            tm_score = torch.nn.functional.cosine_similarity(out1_tmvec, out2_tmvec).item()
            
            return tm_score
            
    except Exception as e:
        print(f"Error calculating TM score with tm_vec: {e}")
        return 0.0

def calculate_tm_score_with_tmalign(coords1, coords2, seq1, seq2):
    """Calculate TM-score using tmtools library with coordinates and sequences."""
    try:
        # Convert coordinates to numpy arrays if needed
        coords1_np = np.array(coords1, dtype=np.float64)
        coords2_np = np.array(coords2, dtype=np.float64)
        
        # Use tmtools to calculate TM-score from coordinates and sequences
        tm_result = tmtools.tm_align(coords1_np, coords2_np, seq1, seq2)
        tm_score = tm_result.tm_norm_chain1
        
        return tm_score
        
    except Exception as e:
        print(f"Error calculating TM score with tmtools: {e}")
        return 0.0

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

def compute_tm_statistics(predicted_tm_scores, true_tm_scores):
    """Compute statistics for TM scores."""
    # Filter out invalid scores (0 or negative)
    valid_indices = (true_tm_scores > 0) & (predicted_tm_scores > 0)
    if not np.any(valid_indices):
        return {
            'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'std_error': 0.0, 
            'std_abs_error': 0.0, 'correlation': 0.0, 'valid_count': 0
        }
    
    pred_valid = predicted_tm_scores[valid_indices]
    true_valid = true_tm_scores[valid_indices]
    
    # Compute errors
    errors = pred_valid - true_valid
    abs_errors = np.abs(errors)
    
    # Statistics
    mae = np.mean(abs_errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    std_error = np.std(errors)
    std_abs_error = np.std(abs_errors)
    
    # Correlation
    correlation = np.corrcoef(pred_valid, true_valid)[0, 1]
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'std_error': std_error,
        'std_abs_error': std_abs_error,
        'correlation': correlation,
        'valid_count': len(pred_valid)
    }

def plot_loss_distribution(all_errors, all_abs_errors, all_true_scores, all_predicted_scores, 
                          all_tm_errors, all_tm_abs_errors, all_true_tm_scores, all_predicted_tm_scores, output_prefix):
    """Plot the distribution of errors and absolute errors for both lDDT and TM scores."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # === lDDT Score Plots ===
    # lDDT Error distribution
    axes[0, 0].hist(all_errors, bins=200, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('lDDT Error Distribution')
    axes[0, 0].set_xlabel('Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_xlim(-0.02, 0.02)  # Set x-axis limits
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # lDDT Absolute error distribution
    axes[0, 1].hist(all_abs_errors, bins=200, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('lDDT Absolute Error Distribution')
    axes[0, 1].set_xlabel('Absolute Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xlim(0, 0.02)  # Set x-axis limits
    
    # lDDT Predicted vs True scatter
    axes[0, 2].scatter(all_true_scores, all_predicted_scores, alpha=0.5, s=1)
    axes[0, 2].set_title('lDDT Predicted vs True')
    axes[0, 2].set_xlabel('True lDDT Score')
    axes[0, 2].set_ylabel('Predicted lDDT Score')
    axes[0, 2].plot([0, 1], [0, 1], 'r--', alpha=0.7)  # Perfect prediction line
    
    # === TM Score Plots ===
    # TM Error distribution
    axes[1, 0].hist(all_tm_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('TM Score Error Distribution')
    axes[1, 0].set_xlabel('Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_xlim(-0.3, 0.3)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # TM Absolute error distribution
    axes[1, 1].hist(all_tm_abs_errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('TM Score Absolute Error Distribution')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_xlim(0, 0.3)
    
    # TM Predicted vs True scatter
    axes[1, 2].scatter(all_true_tm_scores, all_predicted_tm_scores, alpha=0.5, s=1)
    axes[1, 2].set_title('TM Score Predicted vs True')
    axes[1, 2].set_xlabel('True TM Score')
    axes[1, 2].set_ylabel('Predicted TM Score')
    axes[1, 2].plot([0, 1], [0, 1], 'r--', alpha=0.7)  # Perfect prediction line
    
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

print("Loading tm_vec model...")
config = trans_basic_block_Config()
tmvec_model = trans_basic_block.load_from_checkpoint('tm_vec_swiss_model.ckpt', config=config)
tmvec_model.eval()
tmvec_model.freeze()
tmvec_model = tmvec_model.to(device)

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
# TM score tracking
all_predicted_tm_scores = []  # Siamese model predictions
all_tmvec_tm_scores = []      # tm_vec model predictions
all_true_tm_scores = []       # TM-align ground truth
all_tm_errors = []            # Siamese model errors
all_tm_abs_errors = []        # Siamese model absolute errors
all_tmvec_errors = []         # tm_vec model errors
all_tmvec_abs_errors = []     # tm_vec model absolute errors
protein_stats = []  # List to store per-protein statistics
processed_variants = 0  # Counter for processed variants

# Create output CSV
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ["PID", "sequence", "mutated_sequence", "predicted_lddt_scores", "true_lddt_scores", "siamese_tm_score", "tmvec_tm_score", "tmalign_tm_score", "description"]
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
            
            # Get predicted TM scores from both models
            siamese_tm_score = get_predicted_tm_score(wild_seq, mutant_seq, siamese_model, device)
            tmvec_tm_score = calculate_tm_score_with_tmvec(wild_seq, mutant_seq, tmvec_model)
            
            # Get true lDDT scores
            true_lddt_scores = lddt_calculator.calculate_lddt(coords, mutant_coords)
            
            # Calculate true TM score using TM-align
            tmalign_tm_score = calculate_tm_score_with_tmalign(coords, mutant_coords, wild_seq, mutant_seq)
            
            # Compute statistics for this protein
            stats = compute_statistics(predicted_lddt_scores, true_lddt_scores)
            
            # Check if TM-align score is valid (not 0 or None)
            if tmalign_tm_score <= 0:
                print(f"  Skipping TM score analysis for {row['PID']}: invalid TM-align score ({tmalign_tm_score})")
                # Still process lDDT scores but skip TM score analysis
                siamese_tm_error = 0.0
                siamese_tm_abs_error = 0.0
                tmvec_tm_error = 0.0
                tmvec_tm_abs_error = 0.0
                valid_tm_score = False
            else:
                # Compute TM score errors for both models
                siamese_tm_error = siamese_tm_score - tmalign_tm_score
                siamese_tm_abs_error = abs(siamese_tm_error)
                tmvec_tm_error = tmvec_tm_score - tmalign_tm_score
                tmvec_tm_abs_error = abs(tmvec_tm_error)
                valid_tm_score = True
            
            # Store per-protein statistics
            protein_stats.append({
                'PID': row['PID'],
                'sequence_length': len(wild_seq),
                'predicted_mean_lddt': np.mean(predicted_lddt_scores),
                'true_mean_lddt': np.mean(true_lddt_scores),
                'siamese_tm_score': siamese_tm_score,
                'tmvec_tm_score': tmvec_tm_score,
                'tmalign_tm_score': tmalign_tm_score,
                'siamese_tm_error': siamese_tm_error,
                'siamese_tm_abs_error': siamese_tm_abs_error,
                'tmvec_tm_error': tmvec_tm_error,
                'tmvec_tm_abs_error': tmvec_tm_abs_error,
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
            # Collect TM score data only if valid
            if valid_tm_score:
                all_predicted_tm_scores.append(siamese_tm_score)
                all_tmvec_tm_scores.append(tmvec_tm_score)
                all_true_tm_scores.append(tmalign_tm_score)
                all_tm_errors.append(siamese_tm_error)
                all_tm_abs_errors.append(siamese_tm_abs_error)
                all_tmvec_errors.append(tmvec_tm_error)
                all_tmvec_abs_errors.append(tmvec_tm_abs_error)
            processed_variants += 1
            
            # Write to CSV
            writer.writerow({
                "PID": row["PID"],
                "sequence": wild_seq,
                "mutated_sequence": mutant_seq,
                "predicted_lddt_scores": list(predicted_lddt_scores),
                "true_lddt_scores": list(true_lddt_scores),
                "siamese_tm_score": siamese_tm_score,
                "tmvec_tm_score": tmvec_tm_score,
                "tmalign_tm_score": tmalign_tm_score,
                "description": row.get("description", "")
            })
            
            # Print summary with statistics
            print(f"  Predicted mean lDDT: {np.mean(predicted_lddt_scores):.4f}")
            print(f"  True mean lDDT: {np.mean(true_lddt_scores):.4f}")
            print(f"  Siamese TM score: {siamese_tm_score:.4f}")
            print(f"  tm_vec TM score: {tmvec_tm_score:.4f}")
            print(f"  TM-align TM score: {tmalign_tm_score:.4f}")
            if valid_tm_score:
                print(f"  Siamese TM error: {siamese_tm_error:.4f}")
                print(f"  tm_vec TM error: {tmvec_tm_error:.4f}")
            print(f"  lDDT MAE: {stats['mae']:.4f}")
            print(f"  lDDT RMSE: {stats['rmse']:.4f}")
            print(f"  lDDT Correlation: {stats['correlation']:.4f}")
            
        except Exception as e:
            print(f"Error processing {row['PID']}: {e}")
            continue

# Convert to numpy arrays for plotting
all_predicted_scores = np.array(all_predicted_scores)
all_true_scores = np.array(all_true_scores)
all_errors = np.array(all_errors)
all_abs_errors = np.array(all_abs_errors)
# Convert TM score arrays
all_predicted_tm_scores = np.array(all_predicted_tm_scores)
all_tmvec_tm_scores = np.array(all_tmvec_tm_scores)
all_true_tm_scores = np.array(all_true_tm_scores)
all_tm_errors = np.array(all_tm_errors)
all_tm_abs_errors = np.array(all_tm_abs_errors)
all_tmvec_errors = np.array(all_tmvec_errors)
all_tmvec_abs_errors = np.array(all_tmvec_abs_errors)

# Save per-protein statistics to CSV
protein_stats_csv = output_csv.replace('.csv', '_protein_statistics.csv')
protein_stats_df = pd.DataFrame(protein_stats)
protein_stats_df.to_csv(protein_stats_csv, index=False)
print(f"Saved per-protein statistics to: {protein_stats_csv}")

# Compute and print overall statistics (not saved)
overall_stats = compute_statistics(all_predicted_scores, all_true_scores)
overall_siamese_tm_stats = compute_tm_statistics(all_predicted_tm_scores, all_true_tm_scores)
overall_tmvec_tm_stats = compute_tm_statistics(all_tmvec_tm_scores, all_true_tm_scores)

print(f"\n=== OVERALL STATISTICS ===")
print(f"Total protein variants processed: {processed_variants}")
print(f"Total residues processed: {len(all_predicted_scores)}")
print(f"Valid TM score pairs: {overall_siamese_tm_stats['valid_count']}")

print(f"\n=== lDDT STATISTICS ===")
print(f"lDDT MAE: {overall_stats['mae']:.4f}")
print(f"lDDT MSE: {overall_stats['mse']:.4f}")
print(f"lDDT RMSE: {overall_stats['rmse']:.4f}")
print(f"lDDT Standard Deviation of Errors: {overall_stats['std_error']:.4f}")
print(f"lDDT Standard Deviation of Absolute Errors: {overall_stats['std_abs_error']:.4f}")
print(f"lDDT Correlation: {overall_stats['correlation']:.4f}")

print(f"\n=== TM SCORE STATISTICS (vs TM-align) ===")
print(f"Siamese Model:")
print(f"  TM MAE: {overall_siamese_tm_stats['mae']:.4f}")
print(f"  TM MSE: {overall_siamese_tm_stats['mse']:.4f}")
print(f"  TM RMSE: {overall_siamese_tm_stats['rmse']:.4f}")
print(f"  TM Standard Deviation of Errors: {overall_siamese_tm_stats['std_error']:.4f}")
print(f"  TM Standard Deviation of Absolute Errors: {overall_siamese_tm_stats['std_abs_error']:.4f}")
print(f"  TM Correlation: {overall_siamese_tm_stats['correlation']:.4f}")

print(f"\ntm_vec Model:")
print(f"  TM MAE: {overall_tmvec_tm_stats['mae']:.4f}")
print(f"  TM MSE: {overall_tmvec_tm_stats['mse']:.4f}")
print(f"  TM RMSE: {overall_tmvec_tm_stats['rmse']:.4f}")
print(f"  TM Standard Deviation of Errors: {overall_tmvec_tm_stats['std_error']:.4f}")
print(f"  TM Standard Deviation of Absolute Errors: {overall_tmvec_tm_stats['std_abs_error']:.4f}")
print(f"  TM Correlation: {overall_tmvec_tm_stats['correlation']:.4f}")

# Create loss distribution plot
output_prefix = output_csv.replace('.csv', '')
plot_loss_distribution(all_errors, all_abs_errors, all_true_scores, all_predicted_scores, 
                      all_tm_errors, all_tm_abs_errors, all_true_tm_scores, all_predicted_tm_scores, output_prefix)

print(f"\nGenerated scores comparison CSV: {output_csv}")