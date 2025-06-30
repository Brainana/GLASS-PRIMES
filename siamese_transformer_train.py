import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
import json
from tqdm import tqdm

from siamese_transformer_model import SiameseTransformerNet, LightweightTransformerNet
from siamese_dataset import SiameseProteinDataset
from siamese_loss import SiameseLDDTLoss

def create_sample_data(num_samples=10, seq_len=300, prottrans_dim=1024):
    """
    Create sample data for testing the training pipeline.
    In practice, you would load real ProtTrans embeddings and PDB files.
    """
    protein_data = []
    
    for i in range(num_samples):
        # Create dummy ProtTrans embeddings
        prot1_embeddings = np.random.randn(seq_len, prottrans_dim).astype(np.float32)
        prot2_embeddings = np.random.randn(seq_len, prottrans_dim).astype(np.float32)
        
        # Use existing PDB files for testing
        prot1_pdb = "Q1GF61.pdb"
        prot2_pdb = "Q3K3S2.pdb"
        
        protein_data.append({
            'protein1_id': f'protein_{i}_1',
            'protein2_id': f'protein_{i}_2',
            'prot1_embeddings': prot1_embeddings,
            'prot2_embeddings': prot2_embeddings,
            'prot1_pdb': prot1_pdb,
            'prot2_pdb': prot2_pdb
        })
    
    return protein_data

def load_data_from_csv(csv_path, max_samples=None):
    """
    Load protein data from CSV with ground truth LDDT and TM scores.
    
    Args:
        csv_path: Path to CSV file with results from extract_lddt_scores.py
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        List of protein pair data dictionaries
    """
    print(f"Loading data from {csv_path}...")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    if max_samples:
        df = df.head(max_samples)
    
    protein_data = []
    
    for idx, row in df.iterrows():
        try:
            # Extract protein IDs
            protein1_id = row['protein_id1']
            protein2_id = row['protein_id2']
            
            # Extract TM score
            tm_score = float(row['tm_score']) if pd.notna(row['tm_score']) else 0.0
            
            # Extract LDDT scores (JSON string)
            lddt_scores_str = row['lddt_scores_protein2']
            lddt_scores = json.loads(lddt_scores_str) if isinstance(lddt_scores_str, str) else []
            
            # Extract alignment sequences
            seqxA = row['seqxA']
            seqyA = row['seqyA']
            seqM = row['seqM']
            
            # Create dummy ProtTrans embeddings (in practice, you'd load real ones)
            seq_len = len(seqyA.replace('-', ''))  # Length of target protein
            prottrans_dim = 1024
            
            # Create embeddings for both proteins
            prot1_embeddings = np.random.randn(seq_len, prottrans_dim).astype(np.float32)
            prot2_embeddings = np.random.randn(seq_len, prottrans_dim).astype(np.float32)
            
            # Create PDB file paths
            prot1_pdb = f"{protein1_id}.pdb"
            prot2_pdb = f"{protein2_id}.pdb"
            
            protein_data.append({
                'protein1_id': protein1_id,
                'protein2_id': protein2_id,
                'prot1_embeddings': prot1_embeddings,
                'prot2_embeddings': prot2_embeddings,
                'prot1_pdb': prot1_pdb,
                'prot2_pdb': prot2_pdb,
                'tm_score': tm_score,
                'lddt_scores': np.array(lddt_scores, dtype=np.float32),
                'seqxA': seqxA,
                'seqyA': seqyA,
                'seqM': seqM
            })
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    print(f"Loaded {len(protein_data)} protein pairs")
    return protein_data

class CustomTMLDDTLoss(torch.nn.Module):
    """
    Custom loss function that combines TM and LDDT scores based on TM score ranges.
    """
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # Weight for per-residue loss
        self.beta = beta    # Weight for global loss
        
    def forward(self, new_emb1, new_emb2, global_emb1, global_emb2, 
                tm_scores, lddt_scores, alignment_mask, alignment):
        """
        Compute loss based on TM score ranges.
        
        Args:
            new_emb1, new_emb2: Per-residue embeddings
            global_emb1, global_emb2: Global embeddings
            tm_scores: TM scores for each protein pair
            lddt_scores: LDDT scores for each residue
            alignment_mask: Mask for aligned residues
            alignment: Alignment information
        """
        batch_size = tm_scores.shape[0]
        total_loss = 0.0
        total_residue_loss = 0.0
        total_global_loss = 0.0
        valid_pairs = 0
        
        for i in range(batch_size):
            tm_score = tm_scores[i].item()
            
            # Skip pairs with TM score < 0.1 (too chaotic)
            if tm_score < 0.1:
                continue
            
            valid_pairs += 1
            
            # Compute cosine similarity for global embeddings
            global_sim = torch.cosine_similarity(global_emb1[i], global_emb2[i], dim=0)
            
            # Target similarity based on TM score
            target_sim = torch.tensor(tm_score, device=global_sim.device)
            
            # Global loss (always use TM score)
            global_loss = torch.nn.functional.mse_loss(global_sim, target_sim)
            
            # Per-residue loss
            if tm_score >= 0.4:
                # Use combination of TM and LDDT scores
                residue_loss = self._compute_residue_loss_with_lddt(
                    new_emb1[i], new_emb2[i], lddt_scores[i], alignment_mask[i], alignment[i]
                )
            else:
                # Use only TM score for per-residue loss
                residue_loss = self._compute_residue_loss_tm_only(
                    new_emb1[i], new_emb2[i], tm_score, alignment_mask[i]
                )
            
            # Combine losses
            pair_loss = self.alpha * residue_loss + self.beta * global_loss
            
            total_loss += pair_loss
            total_residue_loss += residue_loss
            total_global_loss += global_loss
        
        if valid_pairs == 0:
            return torch.tensor(0.0, device=new_emb1.device, requires_grad=True), {
                'residue_loss': torch.tensor(0.0, device=new_emb1.device),
                'global_loss': torch.tensor(0.0, device=new_emb1.device)
            }
        
        avg_loss = total_loss / valid_pairs
        avg_residue_loss = total_residue_loss / valid_pairs
        avg_global_loss = total_global_loss / valid_pairs
        
        return avg_loss, {
            'residue_loss': avg_residue_loss,
            'global_loss': avg_global_loss
        }
    
    def _compute_residue_loss_with_lddt(self, emb1, emb2, lddt_scores, alignment_mask, alignment):
        """Compute per-residue loss using LDDT scores."""
        # Use LDDT scores as target similarities
        target_scores = torch.tensor(lddt_scores, device=emb1.device, dtype=torch.float32)
        
        # Compute cosine similarities between aligned residues
        aligned_emb1 = emb1[alignment_mask]
        aligned_emb2 = emb2[alignment_mask]
        aligned_targets = target_scores[alignment_mask]
        
        if len(aligned_emb1) == 0:
            return torch.tensor(0.0, device=emb1.device)
        
        similarities = torch.cosine_similarity(aligned_emb1, aligned_emb2, dim=1)
        loss = torch.nn.functional.mse_loss(similarities, aligned_targets)
        
        return loss
    
    def _compute_residue_loss_tm_only(self, emb1, emb2, tm_score, alignment_mask):
        """Compute per-residue loss using only TM score."""
        # Use TM score as target for all aligned residues
        target_score = torch.tensor(tm_score, device=emb1.device, dtype=torch.float32)
        
        # Compute cosine similarities between aligned residues
        aligned_emb1 = emb1[alignment_mask]
        aligned_emb2 = emb2[alignment_mask]
        
        if len(aligned_emb1) == 0:
            return torch.tensor(0.0, device=emb1.device)
        
        similarities = torch.cosine_similarity(aligned_emb1, aligned_emb2, dim=1)
        target_scores = target_score.expand(len(similarities))
        loss = torch.nn.functional.mse_loss(similarities, target_scores)
        
        return loss

def train_transformer_siamese_network(protein_data, config):
    """
    Train the transformer-based Siamese network.
    
    Args:
        protein_data: List of protein pair data
        config: Training configuration dictionary
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = SiameseProteinDataset(
        protein_data=protein_data,
        max_seq_len=config['max_seq_len'],
        prottrans_dim=config['prottrans_dim']
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
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
    criterion = CustomTMLDDTLoss(
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
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            prot1_embeddings = batch['prot1_embeddings'].to(device)
            prot2_embeddings = batch['prot2_embeddings'].to(device)
            prot1_mask = batch['prot1_mask'].to(device)
            prot2_mask = batch['prot2_mask'].to(device)
            tm_scores = batch['tm_scores'].to(device)
            lddt_scores = batch['lddt_scores'].to(device)
            alignment_mask = batch['alignment_mask'].to(device)
            alignment = batch['alignment']
            
            # Forward pass with masks
            new_emb1, new_emb2, global_emb1, global_emb2 = model(
                prot1_embeddings, prot2_embeddings, prot1_mask, prot2_mask
            )
            
            # Compute loss
            loss, loss_dict = criterion(
                new_emb1, new_emb2, global_emb1, global_emb2,
                tm_scores, lddt_scores, alignment_mask, alignment
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
                'Global': f'{loss_dict["global_loss"].item():.4f}'
            })
        
        # Compute average losses
        avg_loss = total_loss / len(dataloader)
        avg_residue_loss = total_residue_loss / len(dataloader)
        avg_global_loss = total_global_loss / len(dataloader)
        
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
        'batch_size': 32,           # Batch size
        'num_epochs': 16,           # Number of training epochs
        'learning_rate': 1e-4,      # Learning rate
        'weight_decay': 1e-5,       # Weight decay
        'max_grad_norm': 1.0,       # Gradient clipping
        'alpha': 0.7,               # Weight for per-residue loss
        'beta': 0.3,                # Weight for global loss
        'num_workers': 0,           # Number of data loading workers
        'model_save_path': 'siamese_transformer_best.pth'  # Path to save best model
    }
    
    # Load data from CSV
    csv_path = "results.csv"  # Path to your results CSV from extract_lddt_scores.py
    protein_data = load_data_from_csv(csv_path, max_samples=100)  # Load first 100 samples
    
    print(f"Training SiameseTransformerNet with custom TM/LDDT loss...")
    print("Loss strategy:")
    print("- TM < 0.1: No loss (too chaotic)")
    print("- TM 0.1-0.4: TM score only")
    print("- TM 0.4-1.0: TM + LDDT scores")
    
    model = train_transformer_siamese_network(protein_data, config)
    
    print("Training completed!")
    print(f"Best model saved to: {config['model_save_path']}")

if __name__ == "__main__":
    main() 