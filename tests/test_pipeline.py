#!/usr/bin/env python3
"""Tests for the teacher/student pipeline."""

import contextlib
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _name in ("data_scripts", "teacher_model", "student_model"):
    _path = os.path.join(REPO_ROOT, _name)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common  # noqa: E402
import patches  # noqa: E402
import splits  # noqa: E402

try:
    import torch

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

try:
    import fgw
    import fgw_data

    HAVE_FGW = True
except ImportError:  # pragma: no cover
    HAVE_FGW = False

if HAVE_TORCH:
    import egnn_model
    import pair_data
    import student_model as student_module
else:  # pragma: no cover
    egnn_model = pair_data = student_module = None

needs_torch = unittest.skipUnless(HAVE_TORCH, "torch not installed")
needs_fgw = unittest.skipUnless(HAVE_FGW, "fgw/fgw_data not importable")


@contextlib.contextmanager
def module_globals(module, **overrides):
    saved = {key: getattr(module, key) for key in overrides}
    for key, value in overrides.items():
        setattr(module, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(module, key, value)


def random_backbone(num_residues=40, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(num_residues, 3))
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    return np.cumsum(steps * 3.8, axis=0).astype(np.float32)


def proper_rotation(seed=0):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


class TestPatchDefinition(unittest.TestCase):
    def test_center_residue_comes_first(self):
        """The encoder reads the patch centre from index 0."""
        coords = random_backbone(50, seed=2)
        for residue_idx in (0, 7, 25, 49):
            with self.subTest(residue_idx=residue_idx):
                self.assertEqual(patches.knn_indices(coords, residue_idx)[0], residue_idx)

    def test_returns_k_neighbors(self):
        coords = random_backbone(100, seed=3)
        indices = patches.knn_indices(coords, 10)
        self.assertEqual(len(indices), patches.K_NEIGHBORS)
        self.assertEqual(len(set(indices.tolist())), len(indices))

    def test_clamps_to_protein_length(self):
        self.assertEqual(len(patches.knn_indices(random_backbone(5, seed=4), 2)), 5)

    def test_neighbors_ordered_by_distance(self):
        coords = random_backbone(60, seed=5)
        indices = patches.knn_indices(coords, 30)
        distances = np.linalg.norm(coords[indices] - coords[30], axis=1)
        self.assertTrue(np.all(np.diff(distances) >= 0))

    @needs_fgw
    def test_label_generation_uses_the_shared_definition(self):
        self.assertIs(fgw_data.knn_indices, patches.knn_indices)
        self.assertIs(fgw_data.K_NEIGHBORS, patches.K_NEIGHBORS)

    def test_full_patch_is_unpadded(self):
        k = patches.K_NEIGHBORS
        coords = random_backbone(k, seed=6)
        features = np.arange(k * 4, dtype=np.float32).reshape(k, 4)
        padded_coords, padded_features, mask = patches.pad_patch(coords, features)
        self.assertTrue(mask.all())
        np.testing.assert_allclose(padded_coords, coords)
        np.testing.assert_allclose(padded_features, features)

    def test_short_patch_is_zero_padded_and_masked(self):
        num_real = 6
        coords = random_backbone(num_real, seed=7)
        features = np.ones((num_real, 4), dtype=np.float32)
        padded_coords, padded_features, mask = patches.pad_patch(coords, features)
        self.assertEqual(mask.sum(), num_real)
        self.assertFalse(mask[num_real:].any())
        np.testing.assert_allclose(padded_coords[:num_real], coords)
        self.assertEqual(padded_coords[num_real:].sum(), 0.0)
        self.assertEqual(padded_features[num_real:].sum(), 0.0)


class TestDataGenerationHelpers(unittest.TestCase):
    """The three stages append to shared outputs over hand-chosen row windows."""

    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp()

    def test_header_is_written_for_a_zero_byte_file(self):
        """A zero-byte file still needs a header written."""
        path = os.path.join(self.dir, "out.csv")
        open(path, "w").close()

        handle, writer = common.open_appending_writer(path, ["a", "b"])
        writer.writerow({"a": 1, "b": 2})
        handle.close()

        self.assertEqual(open(path).readline().strip(), "a,b")

    def test_header_is_not_repeated_on_append(self):
        path = os.path.join(self.dir, "out.csv")
        for value in (1, 2):
            handle, writer = common.open_appending_writer(path, ["a", "b"])
            writer.writerow({"a": value, "b": value})
            handle.close()
        self.assertEqual(open(path).read().count("a,b"), 1)

    def test_read_done_rows_supports_resume(self):
        path = os.path.join(self.dir, "fgw.csv")
        with open(path, "w") as handle:
            handle.write("source_row,v\n10,x\n10,y\n11,z\n")
        self.assertEqual(common.read_done_rows(path, "source_row"), {10, 11})

    def test_read_done_rows_tolerates_missing_file_and_column(self):
        self.assertEqual(
            common.read_done_rows(os.path.join(self.dir, "nope.csv"), "source_row"),
            set(),
        )
        path = os.path.join(self.dir, "other.csv")
        with open(path, "w") as handle:
            handle.write("a,b\n1,2\n")
        self.assertEqual(common.read_done_rows(path, "source_row"), set())

    def test_iter_row_groups_skips_leading_groups(self):
        class Meta:
            def __init__(self, sizes):
                self.sizes = sizes

            def row_group(self, i):
                return type("G", (), {"num_rows": self.sizes[i]})()

        class Parquet:
            def __init__(self, sizes):
                self.num_row_groups = len(sizes)
                self.metadata = Meta(sizes)

        parquet = Parquet([100, 100, 100, 100])
        self.assertEqual(
            [g for g, _ in common.iter_row_groups(parquet, 0, None)], [0, 1, 2, 3]
        )
        self.assertEqual(
            [g for g, _ in common.iter_row_groups(parquet, 250, None)], [2, 3]
        )
        self.assertEqual(
            [(g, o) for g, o in common.iter_row_groups(parquet, 50, 150)],
            [(0, 0), (1, 100)],
        )

    def test_run_manifest_appends_one_json_line_per_run(self):
        import json

        path = os.path.join(self.dir, "out.csv")
        common.write_run_manifest(path, {"stage": "fgw_data", "start_row": 0})
        manifest = common.write_run_manifest(path, {"stage": "fgw_data", "start_row": 50})

        records = [json.loads(line) for line in open(manifest)]
        self.assertEqual([r["start_row"] for r in records], [0, 50])
        self.assertTrue(all("finished_at" in r for r in records))

    def test_failure_histogram_groups_by_exception_type(self):
        counts = common.summarise_failures(
            [ValueError("a"), ValueError("b"), FileNotFoundError("c")]
        )
        self.assertEqual(counts, {"ValueError": 2, "FileNotFoundError": 1})


class TestDataScriptDependencies(unittest.TestCase):
    def test_fgw_imports_without_the_esm_package(self):
        """fgw.py's numerics must not drag in the ESM install."""
        self.assertNotIn("esm", sys.modules)
        import fgw  # noqa: F401

        self.assertNotIn("esm", sys.modules)

    def test_patch_helpers_do_not_require_torch(self):
        """fgw_data runs on CPU nodes; patches.py must stay numpy-only."""
        source = open(os.path.join(REPO_ROOT, "data_scripts", "patches.py")).read()
        self.assertNotIn("import torch", source)
        common = open(os.path.join(REPO_ROOT, "data_scripts", "common.py")).read()
        self.assertNotIn("import torch", common)


class TestSplits(unittest.TestCase):
    IDS = [f"P{i:05d}" for i in range(3000)]

    def pairs(self):
        return list(zip(self.IDS[::2], self.IDS[1::2]))

    def test_deterministic(self):
        pairs = self.pairs()
        first = [splits.row_split(a, b) for a, b in pairs]
        second = [splits.row_split(a, b) for a, b in pairs]
        self.assertEqual(first, second)

    def test_stable_bucket_does_not_use_salted_hash(self):
        """A regression guard: hash() would differ per process."""
        self.assertEqual(splits.stable_bucket("P00001"), splits.stable_bucket("P00001"))
        self.assertTrue(0 <= splits.stable_bucket("anything") < 100)

    def test_proteins_never_cross_splits(self):
        by_split = {}
        for id1, id2 in self.pairs():
            split = splits.row_split(id1, id2)
            if split != "discard":
                by_split.setdefault(split, set()).update([id1, id2])

        names = list(by_split)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = by_split[names[i]] & by_split[names[j]]
                self.assertEqual(overlap, set(), f"{names[i]} vs {names[j]} share proteins")

    def test_every_split_is_populated(self):
        seen = {splits.row_split(a, b) for a, b in self.pairs()}
        for split in splits.SPLITS:
            self.assertIn(split, seen)

    def test_pair_mode_assigns_every_row(self):
        with module_globals(splits, HOLDOUT_MODE="pair"):
            assignments = {splits.row_split(a, b) for a, b in self.pairs()}
        self.assertNotIn("discard", assignments)


@needs_fgw
class TestAlignedResiduePairs(unittest.TestCase):
    def test_gaps_on_both_sides_are_accounted_for(self):
        pairs = list(fgw_data.iter_aligned_residue_pairs("AB-CD", ":::::", "A-BCD"))
        self.assertEqual(
            [(pos, i1, i2) for pos, i1, i2, _, _, _ in pairs],
            [(0, 0, 0), (3, 2, 2), (4, 3, 3)],
        )

    def test_ungapped_alignment_is_identity(self):
        seq = "ACDEFGHIK"
        pairs = list(fgw_data.iter_aligned_residue_pairs(seq, ":" * len(seq), seq))
        self.assertEqual(len(pairs), len(seq))
        for pos, idx1, idx2, _, _, _ in pairs:
            self.assertEqual((idx1, idx2), (pos, pos))

    def test_marker_outside_seqm_values_is_skipped(self):
        pairs = list(fgw_data.iter_aligned_residue_pairs("ABCD", ":X::", "ABCD"))
        self.assertEqual([p[0] for p in pairs], [0, 2, 3])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            list(fgw_data.iter_aligned_residue_pairs("ABC", "::", "ABC"))


@needs_fgw
class TestFgwProperties(unittest.TestCase):
    PARAMS = dict(alpha=0.7, eps=0.05, sinkhorn_iter=10, structure_exp_scale=0.1)

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        cls.X = random_backbone(16, seed=20)
        cls.Y = random_backbone(16, seed=21)
        cls.F1 = rng.normal(size=(16, 32))
        cls.F2 = rng.normal(size=(16, 32))

    def test_identical_patches_have_near_zero_distance(self):
        self.assertLess(
            fgw.compute_fgw_from_features(self.X, self.X, self.F1, self.F1, **self.PARAMS),
            0.05,
        )

    def test_unrelated_patches_are_further(self):
        same = fgw.compute_fgw_from_features(self.X, self.X, self.F1, self.F1, **self.PARAMS)
        different = fgw.compute_fgw_from_features(self.X, self.Y, self.F1, self.F2, **self.PARAMS)
        self.assertGreater(different, same + 0.1)

    def test_distance_is_symmetric(self):
        forward = fgw.compute_fgw_from_features(self.X, self.Y, self.F1, self.F2, **self.PARAMS)
        backward = fgw.compute_fgw_from_features(self.Y, self.X, self.F2, self.F1, **self.PARAMS)
        self.assertAlmostEqual(forward, backward, delta=0.02)

    def test_sinkhorn_plan_has_requested_marginals(self):
        n = 12
        rng = np.random.default_rng(12)
        cost = np.abs(rng.normal(size=(n, n)))
        a = b = np.ones(n) / n
        plan = fgw.sinkhorn(cost, a, b, eps=0.05, n_iter=200)
        self.assertAlmostEqual(plan.sum(), 1.0, places=4)
        np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-3)
        np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-3)

    def test_pairwise_dist_known_case(self):
        points = np.array([[0.0, 0, 0], [3.0, 4, 0]])
        np.testing.assert_allclose(fgw.pairwise_dist(points), [[0, 5], [5, 0]], atol=1e-6)


@needs_torch
class TestEncoderInvariants(unittest.TestCase):
    INPUT_DIM = 16
    NUM_NODES = 12

    def setUp(self):
        torch.manual_seed(0)
        self.encoder = egnn_model.EGNNPatchEncoder(
            input_dim=self.INPUT_DIM, hidden_dim=32, output_dim=8, num_layers=2, dropout=0.0
        ).eval()

    def make_patch(self, seed=0):
        rng = np.random.default_rng(seed)
        coords = torch.from_numpy(random_backbone(self.NUM_NODES, seed=seed)[None])
        features = torch.from_numpy(
            rng.normal(size=(1, self.NUM_NODES, self.INPUT_DIM)).astype(np.float32)
        )
        return features, coords

    def test_output_is_unit_norm(self):
        features, coords = self.make_patch(seed=30)
        with torch.no_grad():
            z = self.encoder(features, coords)
        np.testing.assert_allclose(z.norm(dim=-1).numpy(), 1.0, atol=1e-5)

    def test_invariant_to_rotation_and_translation(self):
        features, coords = self.make_patch(seed=31)
        rotation = torch.from_numpy(proper_rotation(seed=31))
        moved = coords @ rotation.T + torch.tensor([5.0, -3.0, 12.0])
        with torch.no_grad():
            z = self.encoder(features, coords)
            z_moved = self.encoder(features, moved)
        np.testing.assert_allclose(z.numpy(), z_moved.numpy(), atol=1e-4)

    def test_masked_slots_do_not_affect_the_embedding(self):
        num_real = 5
        features, coords = self.make_patch(seed=32)
        mask = torch.zeros(1, self.NUM_NODES, dtype=torch.bool)
        mask[:, :num_real] = True

        clean_f, clean_c = features.clone(), coords.clone()
        clean_f[:, num_real:] = 0.0
        clean_c[:, num_real:] = 0.0
        noisy_f, noisy_c = features.clone(), coords.clone()
        noisy_f[:, num_real:] = 99.0
        noisy_c[:, num_real:] = -42.0

        with torch.no_grad():
            z_clean = self.encoder(clean_f, clean_c, node_mask=mask)
            z_noisy = self.encoder(noisy_f, noisy_c, node_mask=mask)
        np.testing.assert_allclose(z_clean.numpy(), z_noisy.numpy(), atol=1e-5)


@needs_torch
class TestPairMasking(unittest.TestCase):
    INPUT_DIM = 16
    NUM_NODES = 10
    REAL_PAIRS = 3
    PADDED_SLOTS = 2

    def setUp(self):
        torch.manual_seed(1)
        self.model = egnn_model.SiameseEGNNTeacher(
            input_dim=self.INPUT_DIM,
            hidden_dim=32,
            output_dim=8,
            num_layers=2,
            dropout=0.0,
            use_tm_head=True,
        ).eval()

        rng = np.random.default_rng(70)
        shape = (1, self.REAL_PAIRS, self.NUM_NODES, self.INPUT_DIM)
        self.features = torch.from_numpy(rng.normal(size=shape).astype(np.float32))
        self.coords = torch.from_numpy(
            rng.normal(size=(1, self.REAL_PAIRS, self.NUM_NODES, 3)).astype(np.float32) * 3.8
        )
        self.node_mask = torch.ones(1, self.REAL_PAIRS, self.NUM_NODES, dtype=torch.bool)

    def pad(self, tensor, fill):
        extra = list(tensor.shape)
        extra[1] = self.PADDED_SLOTS
        return torch.cat([tensor, torch.full(extra, fill, dtype=tensor.dtype)], dim=1)

    def run_model(self, features, coords, node_mask, pair_mask):
        with torch.no_grad():
            return self.model.forward_protein_pair(
                features, coords, features, coords,
                mask1=node_mask, mask2=node_mask, pair_mask=pair_mask,
            )

    def test_padding_does_not_change_the_tm_prediction(self):
        unpadded = self.run_model(
            self.features, self.coords, self.node_mask,
            torch.ones(1, self.REAL_PAIRS, dtype=torch.bool),
        )

        pair_mask = torch.zeros(1, self.REAL_PAIRS + self.PADDED_SLOTS, dtype=torch.bool)
        pair_mask[:, : self.REAL_PAIRS] = True
        padded = self.run_model(
            self.pad(self.features, 7.0),
            self.pad(self.coords, -5.0),
            self.pad(self.node_mask, False),
            pair_mask,
        )

        np.testing.assert_allclose(
            padded["tm_score_pred"].numpy(), unpadded["tm_score_pred"].numpy(), atol=1e-5
        )
        np.testing.assert_allclose(
            padded["global_z1"].numpy(), unpadded["global_z1"].numpy(), atol=1e-5
        )

    def test_omitting_the_pair_mask_corrupts_pooling(self):
        """Documents why pair_mask is required, not optional."""
        unpadded = self.run_model(
            self.features, self.coords, self.node_mask,
            torch.ones(1, self.REAL_PAIRS, dtype=torch.bool),
        )
        with torch.no_grad():
            unmasked = self.model.forward_protein_pair(
                self.pad(self.features, 7.0), self.pad(self.coords, -5.0),
                self.pad(self.features, 7.0), self.pad(self.coords, -5.0),
                mask1=self.pad(self.node_mask, False),
                mask2=self.pad(self.node_mask, False),
                pair_mask=None,
            )
        difference = float(
            np.abs(unmasked["global_z1"].numpy() - unpadded["global_z1"].numpy()).max()
        )
        self.assertGreater(difference, 1e-3)

    def test_student_pooling_is_also_masked(self):
        student = student_module.SequenceStudent(
            input_dim=self.INPUT_DIM, hidden_dim=32, output_dim=8,
            num_layers=1, num_heads=2, ff_dim=32, dropout=0.0, max_length=64,
        ).eval()

        rng = np.random.default_rng(71)
        real_len = 9
        z = torch.from_numpy(rng.normal(size=(1, 14, 8)).astype(np.float32))
        z = torch.nn.functional.normalize(z, dim=-1)
        mask = torch.zeros(1, 14, dtype=torch.bool)
        mask[:, :real_len] = True

        pooled = student.masked_mean(z, mask)
        expected = torch.nn.functional.normalize(z[:, :real_len].mean(dim=1), dim=-1)
        np.testing.assert_allclose(pooled.numpy(), expected.numpy(), atol=1e-6)


@needs_torch
class TestPairBatching(unittest.TestCase):
    NUM_RESIDUES = 40
    FEATURE_DIM = 8

    def setUp(self):
        import tempfile

        self.num_pairs = 9
        self.rows_per_pair = 7
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        )
        handle.write(
            "tm_data_row,id1,id2,residue_idx1,residue_idx2,"
            "fgw_score,fgw_structure_term,tm_score_norm1\n"
        )
        rng = np.random.default_rng(80)
        for pair_idx in range(self.num_pairs):
            tm = rng.random()
            for _ in range(self.rows_per_pair):
                handle.write(
                    f"{pair_idx},A{pair_idx:03d},B{pair_idx:03d},"
                    f"{rng.integers(0, self.NUM_RESIDUES)},"
                    f"{rng.integers(0, self.NUM_RESIDUES)},"
                    f"{rng.random():.4f},{rng.random():.4f},{tm:.4f}\n"
                )
        handle.close()
        self.csv = handle.name
        self.addCleanup(os.unlink, self.csv)

    def groups(self, chunk_size):
        return list(
            pair_data.iter_pair_groups(self.csv, split=None, chunk_size=chunk_size)
        )

    def test_groups_survive_chunk_boundaries(self):
        """A pair's rows are contiguous but can straddle a read chunk."""
        for chunk_size in (3, 5, 7, 8, 1000):
            with self.subTest(chunk_size=chunk_size):
                groups = self.groups(chunk_size)
                self.assertEqual(len(groups), self.num_pairs)
                self.assertTrue(all(len(g) == self.rows_per_pair for g in groups))
                self.assertTrue(all(g["id1"].nunique() == 1 for g in groups))

    def test_every_row_is_emitted_exactly_once(self):
        groups = self.groups(4)
        keys = sorted(int(g["tm_data_row"].iloc[0]) for g in groups)
        self.assertEqual(keys, list(range(self.num_pairs)))

    def make_cache(self):
        outer = self

        class FakeCache:
            def get(self, protein_id):
                rng = np.random.default_rng(abs(hash(protein_id)) % 2**31)
                return (
                    random_backbone(outer.NUM_RESIDUES, seed=3),
                    rng.normal(size=(outer.NUM_RESIDUES, outer.FEATURE_DIM)).astype(
                        np.float32
                    ),
                )

        return FakeCache()

    def test_collate_pads_and_masks_variable_residue_counts(self):
        groups = self.groups(1000)
        trimmed = [groups[0].iloc[:3], groups[1]]  # 3 residues vs 7
        dataset = pair_data.ProteinPairDataset(trimmed, self.make_cache())
        batch = pair_data.collate_pairs([dataset[0], dataset[1]])

        k = patches.K_NEIGHBORS
        self.assertEqual(batch["pair_mask"].shape, (2, 7))
        self.assertEqual(batch["pair_mask"][0].sum().item(), 3)
        self.assertEqual(batch["pair_mask"][1].sum().item(), 7)
        self.assertEqual(
            batch["patch_features1"].shape, (2, 7, k, self.FEATURE_DIM)
        )
        self.assertEqual(batch["fgw"].shape, (2, 7))
        self.assertEqual(batch["tm"].shape, (2,))
        # padded slots are zeroed
        self.assertEqual(batch["patch_features1"][0, 3:].abs().sum().item(), 0.0)
        self.assertFalse(batch["patch_mask1"][0, 3:].any())

    def test_sequence_mode_adds_sequence_tensors(self):
        groups = self.groups(1000)[:2]
        dataset = pair_data.ProteinPairDataset(
            groups, self.make_cache(), include_sequence=True
        )
        batch = pair_data.collate_pairs([dataset[0], dataset[1]])
        for key in ("features1", "seq_mask1", "residue_idx1"):
            self.assertIn(key, batch)
        self.assertEqual(batch["features1"].shape[-1], self.FEATURE_DIM)
        self.assertEqual(batch["seq_mask1"].sum().item(), 2 * self.NUM_RESIDUES)

    def test_structure_target_is_the_flipped_structure_term(self):
        """The structure-only target must be a similarity, like fgw_score."""
        groups = self.groups(1000)[:1]
        dataset = pair_data.ProteinPairDataset(groups, self.make_cache())
        item = dataset[0]
        expected = 1.0 - groups[0]["fgw_structure_term"].to_numpy()
        np.testing.assert_allclose(item["fgw_structure"], expected, atol=1e-6)

    def test_select_fgw_target_picks_the_right_column(self):
        groups = self.groups(1000)[:2]
        dataset = pair_data.ProteinPairDataset(groups, self.make_cache())
        batch = pair_data.collate_pairs([dataset[0], dataset[1]])

        np.testing.assert_allclose(
            pair_data.select_fgw_target(batch, "structure").numpy(),
            batch["fgw_structure"].numpy(),
        )
        np.testing.assert_allclose(
            pair_data.select_fgw_target(batch, "composite").numpy(),
            batch["fgw"].numpy(),
        )
        # the two targets must actually differ, or the fix is a no-op
        self.assertGreater(
            float(np.abs(batch["fgw_structure"] - batch["fgw"]).max()), 1e-3
        )
        with self.assertRaises(ValueError):
            pair_data.select_fgw_target(batch, "nonsense")

    def test_esm_baseline_uses_the_centre_residue(self):
        """knn puts the centre at patch index 0; the baseline must read it."""
        groups = self.groups(1000)[:2]
        dataset = pair_data.ProteinPairDataset(groups, self.make_cache())
        batch = pair_data.collate_pairs([dataset[0], dataset[1]])

        baseline = pair_data.esm_baseline_similarity(batch)
        self.assertEqual(baseline.shape, batch["fgw"].shape)
        self.assertTrue(torch.all(baseline >= -1.001))
        self.assertTrue(torch.all(baseline <= 1.001))

        centre1 = torch.nn.functional.normalize(
            batch["patch_features1"][0, 0, 0], dim=-1
        )
        centre2 = torch.nn.functional.normalize(
            batch["patch_features2"][0, 0, 0], dim=-1
        )
        self.assertAlmostEqual(
            baseline[0, 0].item(), float((centre1 * centre2).sum()), places=5
        )

    def test_extra_distill_residues_are_sampled_and_masked(self):
        """Unlabelled residues exist only as distillation targets."""
        groups = self.groups(1000)[:2]
        dataset = pair_data.ProteinPairDataset(
            groups, self.make_cache(), include_sequence=True,
            extra_distill_residues=11,
        )
        batch = pair_data.collate_pairs([dataset[0], dataset[1]])

        k = patches.K_NEIGHBORS
        for side in ("1", "2"):
            self.assertEqual(batch[f"extra_mask{side}"].shape, (2, 11))
            self.assertTrue(batch[f"extra_mask{side}"].all())
            self.assertEqual(
                batch[f"extra_patch_features{side}"].shape,
                (2, 11, k, self.FEATURE_DIM),
            )
            indices = batch[f"extra_residue_idx{side}"]
            self.assertTrue(int(indices.max()) < self.NUM_RESIDUES)
            # sampled without replacement
            for row in indices:
                self.assertEqual(len(set(row.tolist())), 11)

    def test_extra_distill_clamps_to_protein_length(self):
        groups = self.groups(1000)[:1]
        dataset = pair_data.ProteinPairDataset(
            groups, self.make_cache(), extra_distill_residues=self.NUM_RESIDUES + 50
        )
        item = dataset[0]
        self.assertEqual(len(item["extra_residue_idx1"]), self.NUM_RESIDUES)

    def test_extra_residues_absent_when_disabled(self):
        groups = self.groups(1000)[:1]
        dataset = pair_data.ProteinPairDataset(groups, self.make_cache())
        self.assertNotIn("extra_residue_idx1", dataset[0])

    def test_max_groups_is_respected_including_the_trailing_group(self):
        for limit in (1, 4, self.num_pairs, self.num_pairs + 5):
            with self.subTest(limit=limit):
                groups = list(
                    pair_data.iter_pair_groups(
                        self.csv, split=None, chunk_size=5, max_groups=limit
                    )
                )
                self.assertEqual(len(groups), min(limit, self.num_pairs))

    def test_unreadable_pair_becomes_None_not_an_exception(self):
        """A dataset failure must not escape the loop and kill the run."""
        class BrokenCache:
            def get(self, protein_id):
                raise ValueError(f"{protein_id}: 70 coords but 140 embeddings")

        groups = self.groups(1000)[:2]
        dataset = pair_data.ProteinPairDataset(groups, BrokenCache())
        self.assertIsNone(dataset[0])
        self.assertEqual(len(dataset.errors), 1)

    def test_skip_errors_False_still_raises(self):
        class BrokenCache:
            def get(self, protein_id):
                raise ValueError("boom")

        groups = self.groups(1000)[:1]
        dataset = pair_data.ProteinPairDataset(
            groups, BrokenCache(), skip_errors=False
        )
        with self.assertRaises(ValueError):
            dataset[0]

    def test_collate_drops_failed_items(self):
        groups = self.groups(1000)[:2]
        dataset = pair_data.ProteinPairDataset(groups, self.make_cache())
        good = dataset[0]

        batch = pair_data.collate_pairs([good, None])
        self.assertEqual(batch["pair_mask"].shape[0], 1)

    def test_collate_returns_None_when_every_item_failed(self):
        self.assertIsNone(pair_data.collate_pairs([None, None]))

    def test_masked_mse_ignores_padded_slots(self):
        predictions = torch.tensor([[1.0, 2.0, 99.0]])
        targets = torch.tensor([[1.0, 4.0, 0.0]])
        mask = torch.tensor([[True, True, False]])
        # only the second slot contributes: (2-4)^2 / 2 valid slots = 2.0
        self.assertAlmostEqual(
            pair_data.masked_mse(predictions, targets, mask).item(), 2.0, places=5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
