import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseLDDTLoss(nn.Module):
    """
    Loss function for Siamese network using LDDT local scores as ground truth.
    Combines per-residue LDDT loss with global similarity loss.
    """
    def __init__(self, alpha=0.7, beta=0.3, margin=0.1):
        """
        Initialize loss function.
        
        Args:
            alpha: Weight for per-residue LDDT loss
            beta: Weight for global similarity loss
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
    
    def forward(self, new_emb1, new_emb2, global_emb1, global_emb2, 
                lddt_scores, alignment_mask, alignment):
        """
        Compute loss for Siamese network.
        
        Args:
            new_emb1, new_emb2: Per-residue embeddings [batch_size, seq_len, output_dim]
            global_emb1, global_emb2: Global embeddings [batch_size, output_dim]
            lddt_scores: Padded LDDT scores [batch_size, seq_len]
            alignment_mask: Mask for aligned residues [batch_size, seq_len]
            alignment: List of (i, j) tuples for aligned residue pairs
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        batch_size = new_emb1.shape[0]
        
        # Per-residue LDDT loss
        residue_loss = self._compute_residue_loss(
            new_emb1, new_emb2, lddt_scores, alignment_mask, alignment
        )
        
        # Global similarity loss (using TM-score as proxy)
        global_loss = self._compute_global_loss(global_emb1, global_emb2)
        
        # Combine losses
        total_loss = self.alpha * residue_loss + self.beta * global_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'residue_loss': residue_loss,
            'global_loss': global_loss
        }
        
        return total_loss, loss_dict
    
    def _compute_residue_loss(self, new_emb1, new_emb2, lddt_scores, alignment_mask, alignment):
        """
        Compute per-residue loss using LDDT scores.
        """
        batch_size, seq_len, output_dim = new_emb1.shape
        
        # Compute cosine similarity for all residue pairs
        # Normalize embeddings
        emb1_norm = F.normalize(new_emb1, p=2, dim=-1)
        emb2_norm = F.normalize(new_emb2, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.bmm(emb1_norm, emb2_norm.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        
        # For each batch, compute loss for aligned residues
        batch_losses = []
        
        for b in range(batch_size):
            batch_alignment = alignment[b] if isinstance(alignment, list) else alignment
            batch_lddt = lddt_scores[b]
            batch_mask = alignment_mask[b]
            
            # Get similarities for aligned residues
            aligned_similarities = []
            aligned_lddt = []
            
            for i, j in batch_alignment:
                if i < seq_len and j < seq_len:
                    sim = similarity[b, i, j]
                    aligned_similarities.append(sim)
                    aligned_lddt.append(batch_lddt[i])
            
            if aligned_similarities:
                aligned_similarities = torch.stack(aligned_similarities)
                aligned_lddt = torch.tensor(aligned_lddt, device=aligned_similarities.device)
                
                # MSE loss between predicted similarity and LDDT score
                residue_loss = F.mse_loss(aligned_similarities, aligned_lddt)
                batch_losses.append(residue_loss)
            else:
                batch_losses.append(torch.tensor(0.0, device=new_emb1.device))
        
        return torch.stack(batch_losses).mean()
    
    def _compute_global_loss(self, global_emb1, global_emb2):
        """
        Compute global similarity loss using contrastive learning.
        """
        # Normalize global embeddings
        emb1_norm = F.normalize(global_emb1, p=2, dim=-1)
        emb2_norm = F.normalize(global_emb2, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(emb1_norm, emb2_norm, dim=-1)
        
        # Contrastive loss: encourage high similarity for similar proteins
        # For now, we'll use a simple MSE loss with target similarity of 0.8
        # In practice, you might want to use the TM-score as the target
        target_similarity = torch.ones_like(similarity) * 0.8
        global_loss = F.mse_loss(similarity, target_similarity)
        
        return global_loss

class ContrastiveLDDTLoss(nn.Module):
    """
    Alternative loss function using contrastive learning with LDDT scores.
    """
    def __init__(self, temperature=0.1, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, global_emb1, global_emb2, lddt_scores, alignment_mask):
        """
        Contrastive loss based on LDDT scores.
        """
        # Normalize embeddings
        emb1_norm = F.normalize(global_emb1, p=2, dim=-1)
        emb2_norm = F.normalize(global_emb2, p=2, dim=-1)
        
        # Compute distance
        distance = torch.norm(emb1_norm - emb2_norm, dim=-1)
        
        # Use average LDDT score as similarity target
        avg_lddt = (lddt_scores * alignment_mask).sum(dim=-1) / alignment_mask.sum(dim=-1).clamp(min=1)
        
        # Contrastive loss: if LDDT is high, distance should be low
        target_distance = 1.0 - avg_lddt  # Convert LDDT to distance
        
        loss = F.mse_loss(distance, target_distance)
        
        return loss 