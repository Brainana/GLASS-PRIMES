#!/usr/bin/env python3
"""Evaluate a trained teacher checkpoint on held-out proteins."""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_SCRIPTS_DIR = os.path.join(REPO_ROOT, "data_scripts")
for _path in (HERE, DATA_SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from egnn_model import SiameseEGNNTeacher  # noqa: E402
from common import log  # noqa: E402
from metrics import (  # noqa: E402
    report,
    report_by_bucket,
    report_esm_baseline,
)
from pair_data import (  # noqa: E402
    ProteinDataCache,
    ProteinPairDataset,
    clip_unit,
    collate_pairs,
    esm_baseline_similarity,
    iter_group_buffers,
)
from splits import split_summary  # noqa: E402
from train_teacher import (  # noqa: E402
    CHECKPOINT_DIR,
    CLIP_TARGETS,
    CSV_CHUNK_SIZE,
    EMBEDDING_DIR,
    FGW_CSV,
    FGW_TARGET,
    PDB_DIR,
    forward_batch,
)

# global config
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "teacher_best.pt")
SPLIT = "test"
EVAL_BATCH_SIZE = 16
GROUPS_PER_BUFFER = 256
PROTEIN_CACHE_SIZE = 512
MAX_GROUPS = None  # None for the whole split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint["config"]

    model = SiameseEGNNTeacher(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_layers=config["num_layers"],
        dropout=0.0,
        use_tm_head=config.get("use_tm_head", False),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate(model, cache):
    fgw_pred, tm_pred, tm_true = [], [], []
    targets = {"structure": [], "composite": []}
    esm_baseline = []
    skipped = 0
    buffer_idx = 0

    for groups in iter_group_buffers(
        FGW_CSV,
        split=SPLIT,
        buffer_size=GROUPS_PER_BUFFER,
        chunk_size=CSV_CHUNK_SIZE,
        max_groups=MAX_GROUPS,
    ):
        buffer_idx += 1
        loader = DataLoader(
            ProteinPairDataset(groups, cache),
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_pairs,
        )
        log(f"eval buffer {buffer_idx}: {len(groups)} protein pairs")

        for batch in loader:
            if batch is None:
                skipped += 1
                continue
            try:
                batch = {key: value.to(DEVICE) for key, value in batch.items()}
                outputs = forward_batch(model, batch)
                mask = batch["pair_mask"]

                fgw_pred.append(outputs["cosine_similarity"][mask].cpu().numpy())
                targets["structure"].append(
                    clip_unit(batch["fgw_structure"], CLIP_TARGETS)[mask].cpu().numpy()
                )
                targets["composite"].append(
                    clip_unit(batch["fgw"], CLIP_TARGETS)[mask].cpu().numpy()
                )
                esm_baseline.append(esm_baseline_similarity(batch)[mask].cpu().numpy())
                if "tm_score_pred" in outputs:
                    tm_pred.append(outputs["tm_score_pred"].cpu().numpy())
                    tm_true.append(clip_unit(batch["tm"], CLIP_TARGETS).cpu().numpy())
            except Exception as exc:
                skipped += 1
                log(f"skipped eval batch in buffer {buffer_idx}: {exc}")

    if not fgw_pred:
        raise ValueError("No evaluation examples processed")

    return {
        "fgw_pred": np.concatenate(fgw_pred),
        "targets": {k: np.concatenate(v) for k, v in targets.items()},
        "esm_baseline": np.concatenate(esm_baseline),
        "tm": (np.concatenate(tm_pred), np.concatenate(tm_true)) if tm_pred else None,
        "skipped": skipped,
    }


def main():
    log("starting test_teacher.py")
    log(f"Checkpoint: {CHECKPOINT_PATH}")
    log(f"FGW CSV: {FGW_CSV}")
    log(f"Device: {DEVICE}")
    log(f"Split: {SPLIT} ({split_summary()})")

    if not os.path.exists(FGW_CSV):
        raise FileNotFoundError(FGW_CSV)

    model, checkpoint = load_model(CHECKPOINT_PATH)
    log(
        f"loaded checkpoint: epoch {checkpoint.get('epoch')}, "
        f"train loss {checkpoint.get('loss', float('nan')):.6f}"
    )

    cache = ProteinDataCache(PDB_DIR, EMBEDDING_DIR, max_size=PROTEIN_CACHE_SIZE)
    results = evaluate(model, cache)

    predictions = results["fgw_pred"]
    targets = results["targets"]
    trained_on = checkpoint["config"].get("fgw_target", FGW_TARGET)

    log("")
    log("=" * 60)
    fgw_metrics = report(
        f"FGW structure-only  (trained on: {trained_on})",
        predictions,
        targets["structure"],
    )
    report("FGW composite (fgw_score, 30% ESM echo)", predictions, targets["composite"])

    report_esm_baseline(results["esm_baseline"], predictions, targets)

    if results["tm"] is not None:
        report("TM   (per protein pair)", *results["tm"])
    else:
        log("")
        log("  TM   : this checkpoint has no TM head")

    report_by_bucket(predictions, targets.get(trained_on, targets["structure"]))
    log("")
    log(f"  skipped batches  {results['skipped']}")
    log(f"  cache hit rate   {cache.hit_rate():.1%}")
    log("=" * 60)
    return fgw_metrics


if __name__ == "__main__":
    main()
