import pickle
import numpy as np
import torch

def create_example_pickle(
    output_path,
    num_samples=100,
    embedding_dim=1024,
    tm_score_range=(0.0, 1.0)
):
    data = []
    for _ in range(num_samples):
        # Example: 128 residues, 1024-dim embedding
        embed1 = torch.randn(128, embedding_dim)
        embed2 = torch.randn(128, embedding_dim)
        tm_score = float(np.random.uniform(*tm_score_range))
        data.append({
            'Embed_sequence_1': embed1,
            'Embed_sequence_2': embed2,
            'tm_score': tm_score
        })
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {num_samples} samples to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a test pickle file for TM-Vec model evaluation.")
    parser.add_argument("--output", type=str, required=True, help="Output pickle file path")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Embedding dimension")
    args = parser.parse_args()
    create_example_pickle(args.output, args.num_samples, args.embedding_dim) 