"""Sequence-only student."""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

TEACHER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "teacher_model")
if TEACHER_DIR not in sys.path:
    sys.path.insert(0, TEACHER_DIR)

from egnn_model import TMScoreHead  # noqa: E402


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_length=1024):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1).float()
        scale = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_length, hidden_dim)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds positional encoding "
                f"table ({self.encoding.shape[1]})"
            )
        return x + self.encoding[:, : x.shape[1]]


class SequenceStudent(nn.Module):
    def __init__(
        self,
        input_dim=480,
        hidden_dim=256,
        output_dim=128,
        num_layers=4,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        max_length=1024,
        use_tm_head=True,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.positional = SinusoidalPositionalEncoding(hidden_dim, max_length)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.output_dim = output_dim
        self.use_tm_head = use_tm_head
        self.tm_head = TMScoreHead(output_dim) if use_tm_head else None

    def forward(self, features, seq_mask=None):
        """features: [B, L, input_dim] -> per-residue unit vectors [B, L, output_dim]."""
        h = self.input_proj(features)
        h = self.positional(h)

        padding_mask = None
        if seq_mask is not None:
            padding_mask = ~seq_mask.bool()
        h = self.encoder(h, src_key_padding_mask=padding_mask)

        z = F.normalize(self.output_proj(h), dim=-1)
        if seq_mask is not None:
            z = z * seq_mask.unsqueeze(-1)
        return z

    @staticmethod
    def gather_residues(z, residue_idx):
        """z: [B, L, D], residue_idx: [B, K] -> [B, K, D]."""
        index = residue_idx.unsqueeze(-1).expand(-1, -1, z.shape[-1])
        return torch.gather(z, 1, index)

    @staticmethod
    def masked_mean(z, mask):
        """Mask-aware pooling; padded rows hold unit vectors, not zeros."""
        weights = mask.unsqueeze(-1).float()
        pooled = (z * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
        return F.normalize(pooled, dim=-1)

    def predict_tm(self, z1, seq_mask1, z2, seq_mask2):
        """TM from pooled per-residue embeddings; forward_pair inlines this."""
        if not self.use_tm_head or self.tm_head is None:
            return None
        g1 = self.masked_mean(z1, seq_mask1)
        g2 = self.masked_mean(z2, seq_mask2)
        return self.tm_head(g1, g2)

    def forward_pair(
        self,
        features1,
        seq_mask1,
        residue_idx1,
        features2,
        seq_mask2,
        residue_idx2,
        pair_mask=None,
        extra_residue_idx1=None,
        extra_residue_idx2=None,
    ):
        """One protein pair per batch row, with K sampled residue pairs each."""
        z1 = self.forward(features1, seq_mask1)
        z2 = self.forward(features2, seq_mask2)

        residue_z1 = self.gather_residues(z1, residue_idx1)
        residue_z2 = self.gather_residues(z2, residue_idx2)

        global_seq_z1 = self.masked_mean(z1, seq_mask1)
        global_seq_z2 = self.masked_mean(z2, seq_mask2)

        outputs = {
            "residue_z1": residue_z1,
            "residue_z2": residue_z2,
            "cosine_similarity": (residue_z1 * residue_z2).sum(dim=-1),
            "global_seq_z1": global_seq_z1,
            "global_seq_z2": global_seq_z2,
        }

        if pair_mask is not None:
            outputs["global_sampled_z1"] = self.masked_mean(residue_z1, pair_mask)
            outputs["global_sampled_z2"] = self.masked_mean(residue_z2, pair_mask)

        if extra_residue_idx1 is not None:
            outputs["extra_z1"] = self.gather_residues(z1, extra_residue_idx1)
        if extra_residue_idx2 is not None:
            outputs["extra_z2"] = self.gather_residues(z2, extra_residue_idx2)

        if self.use_tm_head and self.tm_head is not None:
            outputs["tm_score_pred"] = self.tm_head(global_seq_z1, global_seq_z2)
        return outputs


def distillation_loss(student_z, teacher_z, mask):
    """1 - cosine, averaged over valid residue slots. Both inputs are unit norm."""
    cosine = (student_z * teacher_z).sum(dim=-1)
    valid = mask.float()
    return ((1.0 - cosine) * valid).sum() / valid.sum().clamp(min=1.0)
