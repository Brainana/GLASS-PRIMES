#!/usr/bin/env python3
"""
Training script for transformer-based Siamese network using Parquet data from GCS.

This script loads training data from Parquet files stored in GCS and trains
the Siamese transformer model with TM/LDDT loss.

Usage:
    python siamese_transformer_train_parquet.py
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import gcsfs
from tqdm import tqdm

from siamese_transformer_model import SiameseTransformerNet
from siamese_parquet_dataset import SiameseParquetDataset, siamese_collate_fn
from siamese_loss import TMLDDTLoss


def train_transformer_siamese_network_parquet(config=None, resume_from=None):
    """
    Train the transformer-based Siamese network using Parquet data from GCS.
    
    Args:
        config: Training configuration dictionary
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset from Parquet files
    dataset = SiameseParquetDataset(
        gcs_folder=config.get('gcs_folder', 'gs://primes-bucket/testing_data_weight5/'),
        max_len=config.get('max_seq_len', 300),
        gcs_project=config.get('gcs_project', None),
        key_path=config.get('key_path', 'mit-primes-464001-bfa03c2c5999.json')
    )
    
    print(f"Dataset has {len(dataset)} samples")
    
    # Create model
    print(f"num_layers: {config['num_layers']}")
    model = SiameseTransformerNet(
        input_dim=config['prottrans_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        fs = gcsfs.GCSFileSystem(project=config['gcs_project'], token=config['key_path'])
        with fs.open(resume_from, 'rb') as f:
            checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed model and optimizer from epoch {start_epoch}")
    
    # Training loop
    best_loss = float('inf')
    training_history = []
    
    for epoch in range(start_epoch, config['num_epochs']):
        # reshuffle Dataset cache at the start of each epoch
        dataset._build_shuffled_index_cache()
        # re-create DataLoader for the new shuffled data
        dataloader = DataLoader(
            dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=False,
            collate_fn=lambda batch: siamese_collate_fn(batch, config.get('max_seq_len', 300)),
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )

        model.train()
        total_loss = 0.0
        total_residue_loss = 0.0
        total_global_loss = 0.0
        batches_processed = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            emb1 = batch['embeddings1'].to(device)
            emb2 = batch['embeddings2'].to(device)
            mask1 = batch['mask1'].to(device)
            mask2 = batch['mask2'].to(device)
            tm_scores = batch['tm_score'].to(device)
            lddt_scores = batch['lddt_scores'].to(device)
            seqxA_list = batch['seqxA']
            seqM_list = batch['seqM']
            seqyA_list = batch['seqyA']

            # Forward pass with masks
            new_emb1, new_emb2, global_emb1, global_emb2 = model(
                emb1, emb2, mask1, mask2
            )

            # Compute loss, passing the thresholds
            loss, loss_dict = criterion(
                new_emb1, new_emb2, global_emb1, global_emb2,
                tm_scores, lddt_scores, seqxA_list, seqM_list, seqyA_list,
                min_tm_score_for_global=config['min_tm_score_for_global'],
                min_tm_score_for_lddt=config['min_tm_score_for_lddt']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            optimizer.step()
            
            # Update metrics
            batches_processed += 1
            total_loss += loss.item()
            total_residue_loss += loss_dict['residue_loss'].item()
            total_global_loss += loss_dict['global_loss'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Residue': f'{loss_dict["residue_loss"].item():.4f}',
                'Global': f'{loss_dict["global_loss"].item():.4f}',
                'Batch': f'{batch_idx+1}/{len(dataloader)}'
            })
            
        # Compute average losses
        avg_loss = total_loss / batches_processed
        avg_residue_loss = total_residue_loss / batches_processed
        avg_global_loss = total_global_loss / batches_processed
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Record training history
        epoch_history = {
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'avg_residue_loss': avg_residue_loss,
            'avg_global_loss': avg_global_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'batches_processed': batches_processed
        }
        training_history.append(epoch_history)
        
        # Save training history JSON to GCS every epoch
        gcs_history_path = 'gcs://primes-bucket/models/model_history.json'
        fs = gcsfs.GCSFileSystem(project=config['gcs_project'], token=config['key_path'])
        with fs.open(gcs_history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f'Saved training history to {gcs_history_path}')
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{config["num_epochs"]}:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Residue Loss: {avg_residue_loss:.4f}')
        print(f'  Global Loss: {avg_global_loss:.4f}')
        print(f'  Batches Processed: {batches_processed}')
        print()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            fs = gcsfs.GCSFileSystem(project=config['gcs_project'], token=config['key_path'])
            with fs.open('gcs://primes-bucket/models/model.pth', 'wb') as f:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': config
                }, f)
                print(f'Best model saved with loss: {best_loss:.4f}')
    
    return model, training_history


def main():
    """
    Main training function for transformer-based Siamese network using Parquet data.
    """
    # Configuration
    config = {
        'prottrans_dim': 1024,      # Dimension of ProtTrans embeddings
        'max_seq_len': 300,         # Maximum sequence length for padding
        'hidden_dim': 1024,         # Hidden dimension of the transformer
        'output_dim': 512,          # Output dimension of new embeddings
        'nhead': 4,                 # Number of attention heads
        'num_layers': 2,            # Number of transformer layers
        'dropout': 0.1,             # Dropout rate
        'batch_size': 16,           # Batch size for training
        'num_workers': 0,           # Number of workers for data loading
        'num_epochs': 5,            # Number of training epochs
        'learning_rate': 1e-4,      # Learning rate
        'weight_decay': 1e-5,       # Weight decay
        'max_grad_norm': 1.0,       # Gradient clipping
        'alpha': 0.7,               # Weight for per-residue loss
        'beta': 0.3,                # Weight for global loss
        'gcs_folder': 'gs://primes-bucket/testing_data_weight5/', # GCS folder containing Parquet files
        'gcs_project': 'mit-primes-464001',        # GCS project (optional)
        'key_path': 'mit-primes-464001-bfa03c2c5999.json',  # GCS service account key path (required)
        'min_tm_score_for_global': 0.1,           # Minimum TM-score for global loss
        'min_tm_score_for_lddt': 0.7,             # Minimum TM-score for lDDT loss
    }
    
    print(f"Training SiameseTransformerNet with Parquet data from GCS...")
    print("Loss strategy:")
    print("- TM < 0.1: No loss (too chaotic)")
    print("- TM 0.1-0.7: TM score only")
    print("- TM 0.7-1.0: TM + LDDT scores")
    
    model, history = train_transformer_siamese_network_parquet(config=config, resume_from=None)
    
    print("Training completed!")
    print(f"Training history saved to GCS: {gcs_history_path}")


if __name__ == "__main__":
    main() 