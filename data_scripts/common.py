"""Small helpers shared across the data scripts."""

import argparse
import datetime
import json
import os


def pdb_path(pdb_dir, uniprot_id):
    return os.path.join(pdb_dir, f"{uniprot_id}.pdb")


def embedding_path(embedding_dir, uniprot_id):
    return os.path.join(embedding_dir, f"{uniprot_id}.npy")


def write_progress(progress_file, row_idx):
    """Record the last row processed, so a shard can be resumed."""
    progress_dir = os.path.dirname(progress_file)
    if progress_dir:
        os.makedirs(progress_dir, exist_ok=True)

    with open(progress_file, "w") as handle:
        handle.write(str(row_idx))


def read_done_rows(csv_path, column):
    """Source rows already present in an output file, for resume."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()

    import csv as _csv

    done = set()
    with open(csv_path, newline="") as handle:
        reader = _csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            return set()
        for row in reader:
            value = row.get(column, "")
            if value not in ("", None):
                try:
                    done.add(int(value))
                except ValueError:
                    continue
    return done


def log(msg):
    print(msg, flush=True)


def add_shard_args(parser, start_default, end_default):
    """The row window every stage shares."""
    parser.add_argument("--start-row", type=int, default=start_default)
    parser.add_argument(
        "--end-row",
        type=lambda v: None if v.lower() in ("none", "eof", "") else int(v),
        default=end_default,
        help="exclusive; 'none' for end of input",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="reprocess entries already present in the output (default: skip them)",
    )
    return parser


def build_parser(description, start_default, end_default):
    parser = argparse.ArgumentParser(description=description)
    return add_shard_args(parser, start_default, end_default)


def iter_row_groups(parquet_file, start_row, end_row):
    """Yield (row_group_index, first_global_row) for groups overlapping the window."""
    offset = 0
    for index in range(parquet_file.num_row_groups):
        num_rows = parquet_file.metadata.row_group(index).num_rows
        if offset + num_rows <= start_row:
            offset += num_rows
            continue
        if end_row is not None and offset >= end_row:
            return
        yield index, offset
        offset += num_rows


def write_run_manifest(output_path, record):
    """Append one JSON line describing this run, beside its output."""
    manifest_path = f"{output_path}.runs.jsonl"
    record = dict(record)
    record.setdefault(
        "finished_at", datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    )
    with open(manifest_path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return manifest_path


def summarise_failures(failures):
    """Turn a list of exceptions into a 'why did I lose rows' histogram."""
    counts = {}
    for exc in failures:
        counts[type(exc).__name__] = counts.get(type(exc).__name__, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def log_failures(failures, total):
    if not failures:
        log("  no failures")
        return
    log(f"  failures: {len(failures)} of {total}")
    for name, count in summarise_failures(failures).items():
        log(f"    {name:<28} {count}")


def open_appending_writer(csv_path, fieldnames):
    """Append-mode CSV writer that writes a header only for a genuinely new file."""
    import csv

    output_dir = os.path.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    handle = open(csv_path, "a", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        handle.flush()
    return handle, writer
