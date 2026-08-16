"""Deterministic train / val / test split, shared by every script."""

import hashlib

HOLDOUT_MODE = "protein"

# bucket ranges over 0-99
TEST_BUCKETS = 5  # buckets [0, 5)
VAL_BUCKETS = 5  # buckets [5, 10), train gets the remaining 90%

SPLITS = ("train", "val", "test")


def stable_bucket(key):
    """Deterministic 0-99 bucket."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def bucket_split(bucket):
    if bucket < TEST_BUCKETS:
        return "test"
    if bucket < TEST_BUCKETS + VAL_BUCKETS:
        return "val"
    return "train"


def protein_split(protein_id):
    return bucket_split(stable_bucket(protein_id))


def row_split(id1, id2):
    """Assign a CSV row to 'train', 'val', 'test', or 'discard'."""
    if HOLDOUT_MODE == "pair":
        return bucket_split(stable_bucket(f"{id1}|{id2}"))

    split1 = protein_split(id1)
    split2 = protein_split(id2)
    return split1 if split1 == split2 else "discard"


def split_summary():
    return (
        f"{HOLDOUT_MODE} split: test={TEST_BUCKETS}% val={VAL_BUCKETS}% "
        f"train={100 - TEST_BUCKETS - VAL_BUCKETS}% of proteins"
    )
