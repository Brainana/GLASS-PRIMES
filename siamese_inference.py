import torch
import numpy as np
from siamese_transformer_model import SiameseTransformerNet
import pickle
import os
import torch.nn.functional as F


class SiameseInference:
    """
    Class for inference with trained Siamese transformer models.
    """
    def __init__(self, model_path):
        """
        Initialize the inference class.
        
        Args:
            model_path: Path to the trained model checkpoint
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and config
        self.model, self.config = self._load_trained_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Transformer model loaded from {model_path}")
        print(f"Using device: {self.device}")
    
    def _load_trained_model(self, model_path):
        """
        Load the trained transformer model from checkpoint.
        
        Args:
            model_path: Path to the .pth checkpoint file
            
        Returns:
            model: Loaded model
            config: Model configuration
        """
        print(f"Loading transformer model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
        
        # Create SiameseTransformerNet model
        model = SiameseTransformerNet(
            input_dim=config['prottrans_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len']
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Training loss: {checkpoint['loss']:.4f}")
        print(f"Epoch: {checkpoint['epoch']}")
        
        return model, config
    
    def compute_similarity_scores(self, prot1_emb, prot2_emb, prot1_mask=None, prot2_mask=None):
        """
        Compute similarity scores between protein embeddings.
        
        Args:
            prot1_emb, prot2_emb: Protein embeddings
            prot1_mask, prot2_mask: Attention masks
            
        Returns:
            global_sim: Global similarity score
            residue_sims: Per-residue similarity scores
        """
        with torch.no_grad():
            # Forward pass through transformer with masks
            new_emb1, new_emb2, global_emb1, global_emb2 = self.model(
                prot1_emb, prot2_emb, prot1_mask, prot2_mask
            )
            
            # Compute global similarity
            global_sim = F.cosine_similarity(global_emb1, global_emb2, dim=1)
            
            # Compute per-residue similarities
            residue_sims = F.cosine_similarity(new_emb1, new_emb2, dim=2)
            
            return global_sim, residue_sims
    
    def create_mask_from_length(self, seq_len, max_len=None):
        """
        Create a mask from sequence length.
        
        Args:
            seq_len: Actual sequence length
            max_len: Maximum sequence length (if None, uses seq_len)
            
        Returns:
            mask: Mask tensor [max_len] (1 for real residues, 0 for padding)
        """
        if max_len is None:
            max_len = seq_len
        
        mask = torch.zeros(max_len, device=self.device)
        mask[:seq_len] = 1
        return mask
    
    def get_protein_embedding(self, prottrans_embeddings, mask=None, seq_len=None):
        """
        Get embedding for a single protein (for inference).
        
        Args:
            prottrans_embeddings: ProtTrans embeddings [seq_len, prottrans_dim]
            mask: Optional mask [seq_len] (1 for real residues, 0 for padding)
            seq_len: Optional actual sequence length (used to create mask if mask is None)
            
        Returns:
            embedding: Protein embedding [output_dim]
        """
        # For transformer model, we need to handle masks properly
        total_len = prottrans_embeddings.size(0)
        
        if mask is None:
            if seq_len is not None:
                # Create mask from actual sequence length
                mask = self.create_mask_from_length(seq_len, total_len)
            else:
                # If no mask or seq_len provided, assume all positions are real (no padding)
                mask = torch.ones(total_len, device=self.device)
        else:
            # Ensure mask is on the correct device
            mask = mask.to(self.device)
        
        # Add batch dimension
        embeddings = prottrans_embeddings.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0)
        
        # Process through transformer
        _, _, global_emb, _ = self.model(embeddings, embeddings, mask, mask)
        return global_emb.squeeze(0)  # [output_dim]


class VectorDatabase:
    """
    Simple vector database for storing and searching protein embeddings.
    """
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.embeddings = []
        self.protein_ids = []
    
    def add_protein(self, protein_id, embedding):
        """
        Add a protein embedding to the database.
        
        Args:
            protein_id: Unique identifier for the protein
            embedding: Protein embedding vector
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embedding.shape[0]}")
        
        self.embeddings.append(embedding)
        self.protein_ids.append(protein_id)
    
    def search_similar(self, query_embedding, top_k=10):
        """
        Search for similar proteins using cosine similarity.
        
        Args:
            query_embedding: Query protein embedding
            top_k: Number of top similar proteins to return
        
        Returns:
            results: List of (protein_id, similarity_score) tuples
        """
        if not self.embeddings:
            return []
        
        # Convert to numpy arrays
        embeddings_array = np.array(self.embeddings)
        query_array = query_embedding.reshape(1, -1)
        
        # Compute cosine similarities
        similarities = self._cosine_similarity(query_array, embeddings_array)
        
        # Get top-k results
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            protein_id = self.protein_ids[idx]
            similarity = similarities[0][idx]
            results.append((protein_id, similarity))
        
        return results
    
    def _cosine_similarity(self, a, b):
        """
        Compute cosine similarity between two arrays.
        """
        # Normalize vectors
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        
        # Compute similarity
        return np.dot(a_norm, b_norm.T)
    
    def save_database(self, filepath):
        """
        Save the database to a file.
        """
        data = {
            'embeddings': self.embeddings,
            'protein_ids': self.protein_ids,
            'embedding_dim': self.embedding_dim
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_database(self, filepath):
        """
        Load the database from a file.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.protein_ids = data['protein_ids']
        self.embedding_dim = data['embedding_dim']

def main():
    """
    Example usage of the inference pipeline.
    """
    # Configuration (should match training config)
    config = {
        'prottrans_dim': 1024,
        'max_seq_len': 300,
        'hidden_dim': 512,
        'output_dim': 512,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1
    }
    
    # Initialize inference
    model_path = 'siamese_transformer_best.pth'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    inference = SiameseInference(model_path)
    
    # Initialize vector database
    db = VectorDatabase(config['output_dim'])
    
    # Example: Generate embeddings for some proteins
    print("Generating embeddings for sample proteins...")
    
    # Create sample ProtTrans embeddings (in practice, load real ones)
    sample_proteins = {
        'protein_1': torch.randn(150, config['prottrans_dim']),
        'protein_2': torch.randn(200, config['prottrans_dim']),
        'protein_3': torch.randn(180, config['prottrans_dim']),
    }
    
    # Generate new embeddings and add to database
    for protein_id, embeddings in sample_proteins.items():
        new_embedding = inference.get_protein_embedding(embeddings, seq_len=embeddings.size(0))
        db.add_protein(protein_id, new_embedding.cpu().numpy())
        print(f"Added {protein_id} to database")
    
    # Example search
    print("\nPerforming similarity search...")
    query_embedding = inference.get_protein_embedding(sample_proteins['protein_1'], seq_len=150)
    results = db.search_similar(query_embedding.cpu().numpy(), top_k=3)
    
    print("Top similar proteins:")
    for protein_id, similarity in results:
        print(f"  {protein_id}: {similarity:.4f}")
    
    # Save database
    db.save_database('protein_embeddings_db.pkl')
    print("\nDatabase saved to 'protein_embeddings_db.pkl'")

if __name__ == "__main__":
    main() 