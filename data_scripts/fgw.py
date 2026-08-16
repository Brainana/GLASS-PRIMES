import numpy as np

from parse_pdb import parse_pdb

# computes the pairwise distance matrix

def pairwise_dist(x):
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def cosine_feature_dist(F1, F2):
    F1_norm = F1 / (np.linalg.norm(F1, axis=1, keepdims=True) + 1e-9)
    F2_norm = F2 / (np.linalg.norm(F2, axis=1, keepdims=True) + 1e-9)
    similarity = F1_norm @ F2_norm.T
    return 1 - np.clip(similarity, -1, 1)


def normalize_cost_matrix(C):
    positive = C[C > 0]
    if positive.size == 0:
        return C
    return C / np.mean(positive)


def exponential_scaled_distance(value, scale):
    return 1 - np.exp(-value / scale)


# computes the Sinkhorn transport plan

def sinkhorn(C, a, b, eps=0.05, n_iter=30):
    K = np.exp(-C / eps)

    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(n_iter):
        u = a / (K @ v + 1e-9)
        v = b / (K.T @ u + 1e-9)

    return u[:, None] * K * v[None, :]

# computes the FGW loss based off both the structure and the features

def fgw_terms(C1, C2, F1, F2, T, D_feat=None):

    # structural term
    a = T @ np.ones(T.shape[1])
    b = T.T @ np.ones(T.shape[0])

    term_struct = (
        np.sum((C1 ** 2) * np.outer(a, a)) +
        np.sum((C2 ** 2) * np.outer(b, b)) -
        2 * np.trace(C1 @ T @ C2 @ T.T)
    )

    # feature term
    if D_feat is None:
        D_feat = cosine_feature_dist(F1, F2)
    term_feat = np.sum(D_feat * T)

    return term_struct, term_feat


def fgw_loss(C1, C2, F1, F2, T, alpha=0.7, D_feat=None):
    term_struct, term_feat = fgw_terms(C1, C2, F1, F2, T, D_feat=D_feat)
    return alpha * term_struct + (1 - alpha) * term_feat


def fgw_cost_matrix(C1, C2, T):
    a = T.sum(axis=1)
    b = T.sum(axis=0)

    term1 = (C1 ** 2) @ a
    term2 = (C2 ** 2) @ b
    cross = C1 @ T @ C2.T

    return term1[:, None] + term2[None, :] - 2 * cross


def compute_fgw_from_features(
        X,
        Y,
        F1,
        F2,
        alpha=0.7,
        eps=0.05,
        sinkhorn_iter=30,
        inner_sinkhorn_iter=20,
        normalize=True,
        structure_exp_scale=None,
        return_components=False):
    """sinkhorn_iter is the outer FGW loop; inner_sinkhorn_iter the inner one."""
    C1 = pairwise_dist(X)
    C2 = pairwise_dist(Y)
    M_feat = cosine_feature_dist(F1, F2)

    if normalize:
        C1 = normalize_cost_matrix(C1)
        C2 = normalize_cost_matrix(C2)
        M_feat = normalize_cost_matrix(M_feat)

    n, m = len(X), len(Y)
    a = np.ones(n) / n
    b = np.ones(m) / m

    T = np.outer(a, b)

    for _ in range(sinkhorn_iter):
        M_geom = fgw_cost_matrix(C1, C2, T)
        C = alpha * M_geom + (1 - alpha) * M_feat
        T = sinkhorn(C, a, b, eps=eps, n_iter=inner_sinkhorn_iter)

    term_struct, term_feat = fgw_terms(C1, C2, F1, F2, T, D_feat=M_feat)
    if structure_exp_scale is not None:
        term_struct = exponential_scaled_distance(term_struct, structure_exp_scale)
    score = alpha * term_struct + (1 - alpha) * term_feat

    if return_components:
        return score, term_struct, term_feat

    return score

# computes the FGW distance

def compute_fgw(pdb1, pdb2,
                alpha=0.7,
                eps=0.05,
                sinkhorn_iter=30,
                device="cpu",
                model=None,
                batch_converter=None,
                normalize=True,
                structure_exp_scale=None):
    from embed_esm2 import get_esm_embeddings, load_esm

    if model is None or batch_converter is None:
        model, _, batch_converter = load_esm(device=device)

    # load structures
    X, seq1 = parse_pdb(pdb1)
    Y, seq2 = parse_pdb(pdb2)

    # ESM-2 embeddings
    F1 = get_esm_embeddings(seq1, model, batch_converter, device=device)
    F2 = get_esm_embeddings(seq2, model, batch_converter, device=device)

    return compute_fgw_from_features(
        X,
        Y,
        F1,
        F2,
        alpha=alpha,
        eps=eps,
        sinkhorn_iter=sinkhorn_iter,
        normalize=normalize,
        structure_exp_scale=structure_exp_scale,
    )
