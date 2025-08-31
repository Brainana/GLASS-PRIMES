import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    """
    Custom transformer encoder layer with layer normalization.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, src, src_key_padding_mask=None):
        # Self-attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, 
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # Feedforward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class SiameseTransformerNet(nn.Module):
    """
    Transformer-based Siamese network for protein embeddings.
    Uses self-attention to capture sequence dependencies and local structure patterns.
    Relies on ProtTrans embeddings' inherent positional information.
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=512, 
                 nhead=4, num_layers=2, dropout=0.1, max_seq_len=300):                 
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        
        # Input projection block: linear
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, nhead, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection block: linear
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Attention pooling for global embedding (last linear outputs 1)
        self.global_attention = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x1, x2, mask1=None, mask2=None):
        """
        Forward pass for Siamese transformer network.
        
        Args:
            x1: Padded ProtTrans embeddings for protein 1 [batch_size, seq_len, input_dim]
            x2: Padded ProtTrans embeddings for protein 2 [batch_size, seq_len, input_dim]
            mask1: Padding mask for protein 1 [batch_size, seq_len] (1 for real, 0 for pad)
            mask2: Padding mask for protein 2 [batch_size, seq_len] (1 for real, 0 for pad)
        
        Returns:
            new_emb1: New embeddings for protein 1 [batch_size, seq_len, output_dim]
            new_emb2: New embeddings for protein 2 [batch_size, seq_len, output_dim]
            global_emb1: Global embedding for protein 1 [batch_size, output_dim]
            global_emb2: Global embedding for protein 2 [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x1.shape
        
        # Process protein 1
        new_emb1, global_emb1 = self._process_protein(x1, mask1)
        
        # Process protein 2
        new_emb2, global_emb2 = self._process_protein(x2, mask2)
        
        return new_emb1, new_emb2, global_emb1, global_emb2
    
    def _process_protein(self, x, mask=None):
        """
        Process a single protein through the transformer.
        
        Args:
            x: Protein embeddings [batch_size, seq_len, input_dim]
            mask: Padding mask [batch_size, seq_len]
        
        Returns:
            new_emb: Per-residue embeddings [batch_size, seq_len, output_dim]
            global_emb: Global embedding [batch_size, output_dim]
        """
        # Input projection block 
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # Apply transformer layers (no positional encoding needed)
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=(mask == 0) if mask is not None else None)
        
        # Output projection block 
        new_emb = self.output_projection(x)  # [batch_size, seq_len, output_dim]
        
        # Global embedding with attention pooling
        global_emb = self._attention_pool(new_emb, mask)
        
        return new_emb, global_emb
    
    def _attention_pool(self, embeddings, mask=None):
        """
        Attention pooling to get global embedding from per-residue embeddings.
        
        Args:
            embeddings: [batch_size, seq_len, output_dim]
            mask: [batch_size, seq_len] (1 for real, 0 for pad)
        
        Returns:
            global_embedding: [batch_size, output_dim]
        """
        # Compute attention weights
        attention_weights = self.global_attention(embeddings)  # [batch_size, seq_len, 1]
        
        # Apply mask if provided
        if mask is not None:
            attention_weights = attention_weights.masked_fill(
                mask.unsqueeze(-1) == 0, float('-inf')
            )
        
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        global_embedding = torch.sum(embeddings * attention_weights, dim=1)  # [batch_size, output_dim]
        return global_embedding
    
    def get_protein_embedding(self, x, mask=None):
        """
        Get embedding for a single protein (for inference).
        
        Args:
            x: ProtTrans embeddings for a single protein [seq_len, input_dim]
            mask: Padding mask [seq_len] (optional)
        
        Returns:
            global_embedding: Global embedding [output_dim]
        """
        # Add batch dimension
        x = x.unsqueeze(0)  # [1, seq_len, input_dim]
        if mask is not None:
            mask = mask.unsqueeze(0)  # [1, seq_len]
        
        # Process through transformer
        _, global_emb = self._process_protein(x, mask)
        return global_emb.squeeze(0)  # [output_dim] 