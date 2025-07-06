import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
import json
import io
from tqdm import tqdm

from siamese_transformer_model import SiameseTransformerNet
from siamese_dataset import SiameseProteinDataset
from siamese_loss import TMLDDTLoss

# --- BigQuery setup ---
from google.cloud import bigquery
key_path = "mit-primes-464001-bfa03c2c5999.json"  # Update if needed
bq_client = bigquery.Client.from_service_account_json(key_path)
ground_truth_table = "mit-primes-464001.primes_data.ground_truth_scores"  # Update if needed


def train_transformer_siamese_network(max_samples=None, bq_client=None, ground_truth_table=None, config=None):
    """
    Train the transformer-based Siamese network with batched data loading.
    
    Args:
        max_samples: Maximum number of samples to load (None for all)
        bq_client: BigQuery client
        ground_truth_table: BigQuery table name
        config: Training configuration dictionary
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = SiameseProteinDataset(
        max_samples=max_samples,
        bq_client=bq_client,
        ground_truth_table=ground_truth_table, 
        max_seq_len=config['max_seq_len'],
        prottrans_dim=config['prottrans_dim'],
        data_batch_size=config.get('data_batch_size', 32)
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0  # Set to 0 to avoid multiprocessing issues with BigQuery
    )
    
    # Create model
    model = SiameseTransformerNet(
        input_dim=config['prottrans_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    # Create custom loss function
    criterion = TMLDDTLoss(
        alpha=config['alpha'],
        beta=config['beta']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0.0
        total_residue_loss = 0.0
        total_global_loss = 0.0
        pairs_processed = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            prot1_embeddings = batch['prot1_embeddings'].to(device)
            prot2_embeddings = batch['prot2_embeddings'].to(device)
            prot1_mask = batch['prot1_mask'].to(device)
            prot2_mask = batch['prot2_mask'].to(device)
            tm_scores = batch['tm_score'].to(device)
            lddt_scores = batch['lddt_scores'].to(device)
            seqxA_list = batch['seqxA']
            seqM_list = batch['seqM']
            seqyA_list = batch['seqyA']
            
            # Count actual pairs in this batch
            batch_size = len(seqxA_list)
            pairs_processed += batch_size
            
            # Forward pass with masks
            new_emb1, new_emb2, global_emb1, global_emb2 = model(
                prot1_embeddings, prot2_embeddings, prot1_mask, prot2_mask
            )
            
            # Compute loss
            loss, loss_dict = criterion(
                new_emb1, new_emb2, global_emb1, global_emb2,
                tm_scores, lddt_scores, seqxA_list, seqM_list, seqyA_list
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_residue_loss += loss_dict['residue_loss'].item()
            total_global_loss += loss_dict['global_loss'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Residue': f'{loss_dict["residue_loss"].item():.4f}',
                'Global': f'{loss_dict["global_loss"].item():.4f}',
                'Pairs': pairs_processed
            })
        
        # Compute average losses
        avg_loss = total_loss / pairs_processed
        avg_residue_loss = total_residue_loss / pairs_processed
        avg_global_loss = total_global_loss / pairs_processed
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{config["num_epochs"]}:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Residue Loss: {avg_residue_loss:.4f}')
        print(f'  Global Loss: {avg_global_loss:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, config['model_save_path'])
            print(f'Best model saved with loss: {best_loss:.4f}')
    
    return model

def main():
    """
    Main training function for transformer-based Siamese network.
    """
    # Configuration
    config = {
        'prottrans_dim': 1024,      # Dimension of ProtTrans embeddings
        'max_seq_len': 300,         # Maximum sequence length for padding
        'hidden_dim': 512,          # Hidden dimension of the transformer
        'output_dim': 512,          # Output dimension of new embeddings
        'nhead': 4,                 # Number of attention heads
        'num_layers': 2,            # Number of transformer layers
        'dropout': 0.1,             # Dropout rate
        'batch_size': 32,           # Batch size for training
        'data_batch_size': 32,      # Batch size for data loading from CSV/BigQuery
        'num_epochs': 16,           # Number of training epochs
        'learning_rate': 1e-4,      # Learning rate
        'weight_decay': 1e-5,       # Weight decay
        'max_grad_norm': 1.0,       # Gradient clipping
        'alpha': 0.7,               # Weight for per-residue loss
        'beta': 0.3,                # Weight for global loss
        'num_workers': 4,           # Number of data loading workers
        'model_save_path': 'siamese_transformer_best.pth'  # Path to save best model
    }
    
    # Load data from CSV and BigQuery
    print(f"Training SiameseTransformerNet with custom TM/LDDT loss...")
    print("Loss strategy:")
    print("- TM < 0.1: No loss (too chaotic)")
    print("- TM 0.1-0.4: TM score only")
    print("- TM 0.4-1.0: TM + LDDT scores")
    model = train_transformer_siamese_network(max_samples=100, bq_client=bq_client, ground_truth_table=ground_truth_table, config=config)
    print("Training completed!")
    print(f"Best model saved to: {config['model_save_path']}")

if __name__ == "__main__":
    main() 