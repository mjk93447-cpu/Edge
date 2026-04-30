import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from edge_labeling import (
    compute_edge_label_score,
    load_edge_labels,
    meta_zones_to_mask,
    rasterize_polylines,
    save_edge_label,
    topology_aware_postprocess,
    validate_edge_label_quality,
    zhang_suen_thinning,
)


class EdgeLabelingTest(unittest.TestCase):
    def test_vector_polyline_rasterizes_to_1px_mask(self):
        mask = rasterize_polylines([{"points": [(1, 2), (8, 2)]}], (10, 10), width=1)
        self.assertEqual(int(mask.sum()), 8)
        self.assertTrue(mask[2, 1])
        self.assertTrue(mask[2, 8])

    def test_save_load_label_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "img.png")
            Image.fromarray(np.zeros((20, 20), dtype=np.uint8)).save(image_path)
            rec = save_edge_label(
                image_path,
                (2, 3, 14, 15),
                [{"points": [(0, 0), (10, 10)]}],
                root=os.path.join(tmpdir, "edge_labels"),
            )
            loaded = load_edge_labels([image_path], root=os.path.join(tmpdir, "edge_labels"))
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].roi, rec.roi)
            self.assertTrue(os.path.exists(rec.mask_path))
            self.assertTrue(os.path.exists(rec.skeleton_path))

    def test_perfect_prediction_scores_near_one(self):
        gt = np.zeros((24, 24), dtype=bool)
        gt[12, 3:21] = True
        pred = gt.astype(np.float32)
        score = compute_edge_label_score(pred, gt, tolerance_radius=1)
        self.assertGreaterEqual(score["best_bf1"], 0.99)
        self.assertGreaterEqual(score["cldice"], 0.99)

    def test_shifted_prediction_drops_exact_but_tolerance_recovers(self):
        gt = np.zeros((24, 24), dtype=bool)
        gt[12, 3:21] = True
        pred = np.zeros((24, 24), dtype=np.float32)
        pred[13, 3:21] = 1.0
        strict = compute_edge_label_score(pred, gt, tolerance_radius=0)
        tolerant = compute_edge_label_score(pred, gt, tolerance_radius=1)
        self.assertLess(strict["exact_f1"], 0.5)
        self.assertGreater(tolerant["best_bf1"], strict["best_bf1"])

    def test_meta_zone_ignore_excludes_scoring_region(self):
        gt = np.zeros((20, 20), dtype=bool)
        gt[10, 2:12] = True
        pred = gt.astype(np.float32)
        pred[2:18, 18] = 1.0
        meta_zones = {"ignore": [{"rect": [16, 0, 20, 20]}], "merge-risk": [{"rect": [0, 0, 2, 2]}]}
        ignore_mask = meta_zones_to_mask(meta_zones, gt.shape, zone_types=("ignore",))
        self.assertEqual(int(ignore_mask[:, 18].sum()), 20)
        unignored = compute_edge_label_score(pred, gt, tolerance_radius=0)
        ignored = compute_edge_label_score(pred, gt, meta_zones=meta_zones, tolerance_radius=0)
        self.assertGreater(ignored["precision"], unignored["precision"])
        self.assertEqual(ignored["meta_zone_pixels"]["merge-risk"], 4)

    def test_topology_safe_threshold_penalizes_dense_foreground(self):
        gt = np.zeros((32, 32), dtype=bool)
        gt[16, 4:28] = True
        pred = np.full((32, 32), 0.6, dtype=np.float32)
        score = compute_edge_label_score(pred, gt, topology_safe=True, tolerance_radius=1)
        self.assertLess(score["topology_safe"]["density_ratio"], 8.0)
        self.assertGreater(score["merge_split_risk"], 0.1)

    def test_topology_postprocess_removes_spurs_and_small_components(self):
        pred = np.zeros((32, 32), dtype=np.float32)
        pred[16, 4:28] = 1.0
        pred[15, 10] = 1.0
        pred[2, 2] = 1.0
        processed = topology_aware_postprocess(pred, threshold=0.5, min_component_pixels=4, spur_prune_iters=1)
        self.assertFalse(processed[2, 2])
        self.assertLess(int(processed.sum()), int((pred >= 0.5).sum()))

    def test_label_quality_blocks_dense_or_thick_gt(self):
        dense = np.ones((16, 16), dtype=bool)
        qc = validate_edge_label_quality(dense, max_density=0.25)
        self.assertFalse(qc["valid"])
        self.assertIn("label_density_too_high", qc["warnings"])

    def test_thinning_keeps_single_pixel_skeleton(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[6:9, 2:14] = True
        skel = zhang_suen_thinning(mask)
        self.assertLess(int(skel.sum()), int(mask.sum()))
        self.assertGreater(int(skel.sum()), 0)


if __name__ == "__main__":
    unittest.main()
