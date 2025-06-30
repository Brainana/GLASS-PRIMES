import torch
import numpy as np
from siamese_model import SiameseResidueNet
import pickle
import os

class ProteinEmbeddingGenerator:
    """
    Class for generating new protein embeddings using the trained Siamese network.
    """
    def __init__(self, model_path, config):
        """
        Initialize the embedding generator.
        
        Args:
            model_path: Path to the trained model checkpoint
            config: Model configuration dictionary
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Load model
        self.model = SiameseResidueNet(
            input_dim=config['prottrans_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim']
        ).to(self.device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")
    
    def pad_embeddings(self, embeddings, max_seq_len):
        """
        Pad embeddings to a fixed length.
        
        Args:
            embeddings: ProtTrans embeddings [seq_len, prottrans_dim]
            max_seq_len: Maximum sequence length for padding
        
        Returns:
            padded_embeddings: Padded embeddings [max_seq_len, prottrans_dim]
        """
        seq_len, dim = embeddings.shape
        
        if seq_len >= max_seq_len:
            # Truncate
            padded = embeddings[:max_seq_len]
        else:
            # Pad with zeros
            padded = np.zeros((max_seq_len, dim))
            padded[:seq_len] = embeddings
        
        return padded
    
    def generate_embedding(self, prottrans_embeddings):
        """
        Generate new embedding for a single protein.
        
        Args:
            prottrans_embeddings: ProtTrans embeddings [seq_len, prottrans_dim]
        
        Returns:
            new_embedding: New protein embedding [output_dim]
        """
        # Pad embeddings
        padded_embeddings = self.pad_embeddings(
            prottrans_embeddings, self.config['max_seq_len']
        )
        
        # Convert to tensor
        embeddings_tensor = torch.FloatTensor(padded_embeddings).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            new_embedding = self.model.get_protein_embedding(embeddings_tensor)
        
        return new_embedding.cpu().numpy()
    
    def generate_embeddings_batch(self, prottrans_embeddings_list):
        """
        Generate embeddings for a batch of proteins.
        
        Args:
            prottrans_embeddings_list: List of ProtTrans embeddings
        
        Returns:
            new_embeddings: List of new protein embeddings
        """
        new_embeddings = []
        
        for embeddings in prottrans_embeddings_list:
            new_emb = self.generate_embedding(embeddings)
            new_embeddings.append(new_emb)
        
        return new_embeddings

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
        'max_seq_len': 512,
        'hidden_dim': 256,
        'output_dim': 128
    }
    
    # Initialize embedding generator
    model_path = 'siamese_model_best.pth'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    generator = ProteinEmbeddingGenerator(model_path, config)
    
    # Initialize vector database
    db = VectorDatabase(config['output_dim'])
    
    # Example: Generate embeddings for some proteins
    print("Generating embeddings for sample proteins...")
    
    # Create sample ProtTrans embeddings (in practice, load real ones)
    sample_proteins = {
        'protein_1': np.random.randn(150, config['prottrans_dim']).astype(np.float32),
        'protein_2': np.random.randn(200, config['prottrans_dim']).astype(np.float32),
        'protein_3': np.random.randn(180, config['prottrans_dim']).astype(np.float32),
    }
    
    # Generate new embeddings and add to database
    for protein_id, embeddings in sample_proteins.items():
        new_embedding = generator.generate_embedding(embeddings)
        db.add_protein(protein_id, new_embedding)
        print(f"Added {protein_id} to database")
    
    # Example search
    print("\nPerforming similarity search...")
    query_embedding = generator.generate_embedding(sample_proteins['protein_1'])
    results = db.search_similar(query_embedding, top_k=3)
    
    print("Top similar proteins:")
    for protein_id, similarity in results:
        print(f"  {protein_id}: {similarity:.4f}")
    
    # Save database
    db.save_database('protein_embeddings_db.pkl')
    print("\nDatabase saved to 'protein_embeddings_db.pkl'")

if __name__ == "__main__":
    main() 