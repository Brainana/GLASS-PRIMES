#!/usr/bin/env python3

import os
import sys

import numpy as np
import pandas as pd

from common import (
    build_parser,
    embedding_path,
    log_failures,
    open_appending_writer,
    pdb_path,
    read_done_rows,
    write_progress,
    write_run_manifest,
)
from fgw import compute_fgw_from_features
from parse_pdb import parse_pdb
from patches import K_NEIGHBORS, knn_indices

# global config
TM_DATA_CSV = "/jet/home/jxu23/OCEANDIR/tm_scores.csv"
PDB_DIR = "/jet/home/jxu23/OCEANDIR/pdbs"
EMBEDDING_DIR = "/jet/home/jxu23/OCEANDIR/embeddings"
OUTPUT_CSV = "/jet/home/jxu23/OCEANDIR/fgw_scores.csv"
PROGRESS_FILE = "/jet/home/jxu23/OCEANDIR/fgw_data_progress.txt"

START_ROW = 0
END_ROW = 50000
CSV_CHUNK_SIZE = 1000
FLUSH_EVERY_ROWS = 10
ALIGNED_RESIDUE_STRIDE = 32  # K_NEIGHBORS now comes from patches.py
ALPHA = 0.7
EPS = 0.05
SINKHORN_ITER = 30
STRUCTURE_EXP_SCALE = 0.1

SEQM_VALUES = {":", ".", " "}


def compute_local_fgw(
    coords1: np.ndarray,
    coords2: np.ndarray,
    features1: np.ndarray,
    features2: np.ndarray,
):
    return compute_fgw_from_features(
        coords1,
        coords2,
        features1,
        features2,
        alpha=ALPHA,
        eps=EPS,
        sinkhorn_iter=SINKHORN_ITER,
        structure_exp_scale=STRUCTURE_EXP_SCALE,
        return_components=True,
    )


def iter_aligned_residue_pairs(seqxA: str, seqM: str, seqyA: str):
    if not (len(seqxA) == len(seqM) == len(seqyA)):
        raise ValueError("seqxA, seqM, and seqyA must have the same length")

    idx1 = 0
    idx2 = 0

    for align_pos, (aa1, marker, aa2) in enumerate(zip(seqxA, seqM, seqyA)):
        residue_idx1 = idx1 if aa1 != "-" else None
        residue_idx2 = idx2 if aa2 != "-" else None

        if aa1 != "-":
            idx1 += 1
        if aa2 != "-":
            idx2 += 1

        if marker not in SEQM_VALUES:
            continue
        if residue_idx1 is None or residue_idx2 is None:
            continue

        yield align_pos, residue_idx1, residue_idx2, aa1, marker, aa2


def load_protein_data(uniprot_id: str):
    pdb_file = pdb_path(PDB_DIR, uniprot_id)
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(pdb_file)

    embedding_file = embedding_path(EMBEDDING_DIR, uniprot_id)
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(embedding_file)

    coords, sequence = parse_pdb(pdb_file)
    if len(sequence) == 0:
        raise ValueError(f"No CA atoms found in {pdb_file}")

    embeddings = np.load(embedding_file).astype(np.float32)
    if len(coords) != len(embeddings):
        raise ValueError(
            f"Coordinate/embedding length mismatch for {uniprot_id}: "
            f"{len(coords)} coords vs {len(embeddings)} embeddings"
        )

    return coords, sequence, embeddings


def input_rows(csv_path: str, start_row, end_row, skip_source_rows=frozenset()):
    """Stream tm_scores.csv rows whose SOURCE parquet row is in the window."""
    start_row = 0 if start_row is None else start_row
    line_no = 0

    for df_chunk in pd.read_csv(
        csv_path,
        chunksize=CSV_CHUNK_SIZE,
        keep_default_na=False,
    ):
        for _, row in df_chunk.iterrows():
            source_row = int(row["row"]) if "row" in row else line_no
            line_no += 1

            if source_row < start_row:
                continue
            if end_row is not None and source_row >= end_row:
                continue
            if source_row in skip_source_rows:
                continue

            yield source_row, row


def output_fields():
    return [
        "tm_data_row",
        "source_row",
        "id1",
        "id2",
        "tm_score_norm1",
        "align_pos",
        "residue_idx1",
        "residue_idx2",
        "aa1",
        "seqM",
        "aa2",
        "fgw_score",
        "fgw_structure_term",
        "fgw_feature_term",
        "neighborhood_size1",
        "neighborhood_size2",
    ]


def write_result(
    writer,
    row,
    input_row_idx,
    residue_pair,
    fgw_score,
    structure_term,
    feature_term,
    n1,
    n2,
):
    align_pos, residue_idx1, residue_idx2, aa1, marker, aa2 = residue_pair
    source_row = row["row"] if "row" in row else input_row_idx

    writer.writerow(
        {
            "tm_data_row": input_row_idx,
            "source_row": source_row,
            "id1": row["id1"],
            "id2": row["id2"],
            "tm_score_norm1": row["tm_score_norm1"],
            "align_pos": align_pos,
            "residue_idx1": residue_idx1,
            "residue_idx2": residue_idx2,
            "aa1": aa1,
            "seqM": marker,
            "aa2": aa2,
            "fgw_score": fgw_score,
            "fgw_structure_term": structure_term,
            "fgw_feature_term": feature_term,
            "neighborhood_size1": n1,
            "neighborhood_size2": n2,
        }
    )


def process_tm_row(row, input_row_idx, writer):
    id1 = str(row["id1"]).strip()
    id2 = str(row["id2"]).strip()

    coords1, _, embeddings1 = load_protein_data(id1)
    coords2, _, embeddings2 = load_protein_data(id2)

    seqxA = str(row["seqxA"])
    seqM = str(row["seqM"])
    seqyA = str(row["seqyA"])

    pending = []
    for aligned_pair_idx, residue_pair in enumerate(
        iter_aligned_residue_pairs(seqxA, seqM, seqyA)
    ):
        if aligned_pair_idx % ALIGNED_RESIDUE_STRIDE != 0:
            continue

        _, residue_idx1, residue_idx2, _, _, _ = residue_pair

        indices1 = knn_indices(coords1, residue_idx1)
        indices2 = knn_indices(coords2, residue_idx2)

        fgw_distance, structure_term, feature_term = compute_local_fgw(
            coords1[indices1],
            coords2[indices2],
            embeddings1[indices1],
            embeddings2[indices2],
        )
        fgw_score = 1 - fgw_distance

        pending.append(
            (residue_pair, fgw_score, structure_term, feature_term,
             len(indices1), len(indices2))
        )

    for residue_pair, score, struct, feat, n1, n2 in pending:
        write_result(writer, row, input_row_idx, residue_pair, score, struct, feat, n1, n2)

    return len(pending)


def main():
    args = build_parser(
        "Compute local FGW scores for TM-aligned pairs in a parquet row window.",
        START_ROW,
        END_ROW,
    ).parse_args()
    start_row, end_row = args.start_row, args.end_row

    if not os.path.exists(TM_DATA_CSV):
        print(f"Error: TM data CSV not found: {TM_DATA_CSV}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(PDB_DIR):
        print(f"Error: PDB directory not found: {PDB_DIR}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(EMBEDDING_DIR):
        print(
            f"Error: embedding directory not found: {EMBEDDING_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"TM data: {TM_DATA_CSV}")
    print(f"PDB dir: {PDB_DIR}")
    print(f"Embedding dir: {EMBEDDING_DIR}")
    print(f"Source rows: [{start_row}, {end_row if end_row is not None else 'EOF'})")
    print(f"CSV chunk size: {CSV_CHUNK_SIZE}")
    print(f"Flush every rows: {FLUSH_EVERY_ROWS}")
    print(f"k-NN size: {K_NEIGHBORS}")
    print(f"Aligned residue stride: {ALIGNED_RESIDUE_STRIDE}")
    print(f"Structure exp scale: {STRUCTURE_EXP_SCALE}")
    print(f"Output: {OUTPUT_CSV}")

    skip_source_rows = (
        frozenset() if args.no_resume else read_done_rows(OUTPUT_CSV, "source_row")
    )
    if skip_source_rows:
        print(f"Resume: {len(skip_source_rows)} pairs already scored, skipping them")

    total_pairs = 0
    failures = []
    rows_since_flush = 0
    last_row_processed = None

    output_file, writer = open_appending_writer(OUTPUT_CSV, output_fields())
    with output_file:
        for input_row_idx, row in input_rows(
            TM_DATA_CSV, start_row, end_row, skip_source_rows
        ):
            try:
                scored_pairs = process_tm_row(
                    row,
                    input_row_idx,
                    writer,
                )
                total_pairs += scored_pairs
                print(
                    f"[row {input_row_idx}] {row['id1']} vs {row['id2']}: "
                    f"{scored_pairs} local FGW scores"
                )
                rows_since_flush += 1
                if rows_since_flush >= FLUSH_EVERY_ROWS:
                    output_file.flush()
                    write_progress(PROGRESS_FILE, input_row_idx)
                    rows_since_flush = 0
            except Exception as exc:
                failures.append(exc)
                print(
                    f"[row {input_row_idx}] skipped ({exc})",
                    file=sys.stderr,
                )
            finally:
                last_row_processed = input_row_idx

        output_file.flush()
        if last_row_processed is not None:
            write_progress(PROGRESS_FILE, last_row_processed)

    print("-" * 60)
    print(f"Done. Local FGW scores: {total_pairs}")
    log_failures(failures, total_pairs + len(failures))
    print(f"Results saved to: {os.path.abspath(OUTPUT_CSV)}")
    manifest = write_run_manifest(
        OUTPUT_CSV,
        {
            "stage": "fgw_data",
            "start_row": start_row,
            "end_row": end_row,
            "scores_written": total_pairs,
            "failed": len(failures),
            "resumed_skips": len(skip_source_rows),
            "last_row_processed": last_row_processed,
        },
    )
    print(f"Run recorded in: {manifest}")


if __name__ == "__main__":
    main()
