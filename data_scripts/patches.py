"""The patch definition, shared by label generation and training."""

import numpy as np

K_NEIGHBORS = 32


def knn_indices(coords, residue_idx, k=K_NEIGHBORS):
    """Indices of the k residues nearest `residue_idx`, nearest first."""
    distances = np.linalg.norm(coords - coords[residue_idx], axis=1)
    k = min(k, len(coords))
    return np.argsort(distances)[:k]


def pad_patch(coords, features, k=K_NEIGHBORS):
    """Pad a patch out to k nodes, returning (coords, features, mask)."""
    num_nodes = len(coords)
    feature_dim = features.shape[-1]

    padded_coords = np.zeros((k, 3), dtype=np.float32)
    padded_features = np.zeros((k, feature_dim), dtype=np.float32)
    mask = np.zeros(k, dtype=bool)

    padded_coords[:num_nodes] = coords
    padded_features[:num_nodes] = features
    mask[:num_nodes] = True

    return padded_coords, padded_features, mask
