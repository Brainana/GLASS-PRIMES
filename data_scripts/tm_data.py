#!/usr/bin/env python3

import os
import sys

import pyarrow.parquet as pq
from tmtools import tm_align

from common import (
    build_parser,
    iter_row_groups,
    log_failures,
    open_appending_writer,
    pdb_path,
    read_done_rows,
    write_progress,
    write_run_manifest,
)
from parse_pdb import parse_pdb

# global config 
PARQUET_PATH = "/jet/home/jxu23/OCEANDIR/swiss_under_1000_320M.parquet"
PDB_DIR = "/jet/home/jxu23/OCEANDIR/pdbs"
OUTPUT_CSV = "/jet/home/jxu23/OCEANDIR/tm_scores.csv"
PROGRESS_FILE = "/jet/home/jxu23/OCEANDIR/tm_data_progress.txt"

START_ROW = 32000
END_ROW = 100000

COL1 = "chain_1"
COL2 = "chain_2"

MAX_SEQUENCE_LENGTH = 1000
FLUSH_EVERY_ROWS = 100


def run_tm_align(uniprot_id1: str, uniprot_id2: str):
    path1 = pdb_path(PDB_DIR, uniprot_id1)
    path2 = pdb_path(PDB_DIR, uniprot_id2)

    if not os.path.exists(path1):
        raise FileNotFoundError(path1)
    if not os.path.exists(path2):
        raise FileNotFoundError(path2)

    coords1, seq1 = parse_pdb(path1)
    coords2, seq2 = parse_pdb(path2)

    if len(seq1) == 0:
        raise ValueError(f"No CA atoms found in {path1}")
    if len(seq2) == 0:
        raise ValueError(f"No CA atoms found in {path2}")
    if MAX_SEQUENCE_LENGTH is not None and len(seq1) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"{uniprot_id1} sequence length {len(seq1)} exceeds "
            f"MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}"
        )
    if MAX_SEQUENCE_LENGTH is not None and len(seq2) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"{uniprot_id2} sequence length {len(seq2)} exceeds "
            f"MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}"
        )

    result = tm_align(coords1, coords2, seq1, seq2)

    return {
        "id1": uniprot_id1,
        "id2": uniprot_id2,
        "seq1": seq1,
        "seq2": seq2,
        "tm_score_norm1": float(result.tm_norm_chain1),
        "seqxA": result.seqxA,
        "seqM": result.seqM,
        "seqyA": result.seqyA,
    }


def iter_pairs(parquet_path: str, col1: str, col2: str, start_row: int, end_row,
               skip_rows=frozenset()):
    """Yield (global_parquet_row, id1, id2) over the requested window."""
    parquet_file = pq.ParquetFile(parquet_path)
    start_row = 0 if start_row is None else start_row

    for row_group_idx, group_start in iter_row_groups(
        parquet_file, start_row, end_row
    ):
        table = parquet_file.read_row_group(row_group_idx, columns=[col1, col2])
        df_chunk = table.to_pandas()

        for local_idx, (_, row) in enumerate(df_chunk.iterrows()):
            global_row_idx = group_start + local_idx

            if global_row_idx < start_row:
                continue
            if end_row is not None and global_row_idx >= end_row:
                return
            if global_row_idx in skip_rows:
                continue

            id1 = str(row[col1]).strip()
            id2 = str(row[col2]).strip()
            yield global_row_idx, id1, id2


def output_fields():
    return [
        "id1",
        "id2",
        "seq1",
        "seq2",
        "tm_score_norm1",
        "seqxA",
        "seqM",
        "seqyA",
        "row",
    ]


def main():
    args = build_parser(
        "TM-align every protein pair in a parquet row window.",
        START_ROW,
        END_ROW,
    ).parse_args()
    start_row, end_row = args.start_row, args.end_row

    if not os.path.exists(PARQUET_PATH):
        print(f"Error: parquet file not found: {PARQUET_PATH}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(PDB_DIR):
        print(f"Error: PDB directory not found: {PDB_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Parquet: {PARQUET_PATH}")
    print(f"PDB dir: {PDB_DIR}")
    print(f"Rows: [{start_row}, {end_row if end_row is not None else 'EOF'})")
    print(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Progress file: {PROGRESS_FILE}")
    print(f"Flush every rows: {FLUSH_EVERY_ROWS}")

    skip_rows = frozenset() if args.no_resume else read_done_rows(OUTPUT_CSV, "row")
    if skip_rows:
        print(f"Resume: {len(skip_rows)} pairs already written, skipping them")

    success = 0
    failures = []
    rows_since_flush = 0
    last_row_processed = None

    output_file, writer = open_appending_writer(OUTPUT_CSV, output_fields())
    with output_file:
        for row_idx, id1, id2 in iter_pairs(
            PARQUET_PATH, COL1, COL2, start_row, end_row, skip_rows=skip_rows
        ):
            try:
                result = run_tm_align(id1, id2)
                result["row"] = row_idx
                writer.writerow(result)
                success += 1
                print(
                    f"[row {row_idx}] {id1} vs {id2}: "
                    f"TM={result['tm_score_norm1']:.4f}"
                )
            except Exception as exc:
                failures.append(exc)
                print(
                    f"[row {row_idx}] {id1} vs {id2}: skipped ({exc})",
                    file=sys.stderr,
                )
            finally:
                last_row_processed = row_idx
                rows_since_flush += 1
                if rows_since_flush >= FLUSH_EVERY_ROWS:
                    output_file.flush()
                    write_progress(PROGRESS_FILE, row_idx)
                    rows_since_flush = 0

        output_file.flush()
        if last_row_processed is not None:
            write_progress(PROGRESS_FILE, last_row_processed)

    print("-" * 60)
    print(f"Done. Successful: {success}")
    log_failures(failures, success + len(failures))
    print(f"Last row processed: {last_row_processed}")
    print(f"Results saved to: {os.path.abspath(OUTPUT_CSV)}")
    manifest = write_run_manifest(
        OUTPUT_CSV,
        {
            "stage": "tm_data",
            "start_row": start_row,
            "end_row": end_row,
            "pairs_written": success,
            "failed": len(failures),
            "resumed_skips": len(skip_rows),
            "last_row_processed": last_row_processed,
        },
    )
    print(f"Run recorded in: {manifest}")


if __name__ == "__main__":
    main()
