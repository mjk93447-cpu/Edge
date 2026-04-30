import inspect
import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

import edge_labeling
import edge_training
from edge_labeling import compute_edge_label_score, load_edge_labels, load_record_masks, save_edge_label
from sobel_edge_detection import EdgeBatchGUI


class DeepEdgeGridAuditTest(unittest.TestCase):
    def test_a_label_storage_grid_vector_mask_skeleton_meta_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "sample.png")
            Image.fromarray(np.zeros((24, 32), dtype=np.uint8)).save(image_path)
            root = os.path.join(tmpdir, "edge_labels")

            rec = save_edge_label(
                image_path,
                (2, 3, 22, 19),
                [{"points": [(1, 2), (16, 2)], "width": 3}],
                meta_zones={"ignore": [{"rect": [1, 1, 4, 4]}], "unsupported": [{"rect": [0, 0, 1, 1]}]},
                root=root,
            )

            self.assertEqual(rec.roi_shape, (16, 20))
            for path in (rec.vector_path, rec.mask_path, rec.skeleton_path, rec.meta_zone_path):
                self.assertTrue(os.path.exists(path), path)

            with open(rec.vector_path, "r", encoding="utf-8") as handle:
                vector_payload = json.load(handle)
            self.assertEqual(vector_payload["roi"], [2, 3, 22, 19])
            self.assertEqual(len(vector_payload["polylines"]), 1)

            with open(rec.meta_zone_path, "r", encoding="utf-8") as handle:
                meta_payload = json.load(handle)
            self.assertIn("ignore", meta_payload)
            self.assertNotIn("unsupported", meta_payload)

            mask, skeleton = load_record_masks(rec)
            self.assertEqual(mask.shape, (16, 20))
            self.assertEqual(skeleton.shape, (16, 20))
            self.assertGreater(mask.sum(), skeleton.sum())
            self.assertEqual(len(load_edge_labels([image_path], root=root)), 1)

    def test_b_scoring_grid_threshold_tolerance_ignore_and_shift_behavior(self):
        gt = np.zeros((1000, 1000), dtype=bool)
        gt[500, 100:900] = True
        pred = gt.astype(np.float32)

        score = compute_edge_label_score(pred, gt)
        self.assertEqual(score["tolerance_radius"], 3)
        self.assertEqual(len(score["curve"]), 19)
        self.assertAlmostEqual(score["best_bf1"], 1.0, places=6)
        self.assertAlmostEqual(score["exact_f1"], 1.0, places=6)

        shifted = np.zeros_like(pred)
        shifted[501, 100:900] = 1.0
        strict = compute_edge_label_score(shifted, gt, tolerance_radius=0)
        tolerant = compute_edge_label_score(shifted, gt, tolerance_radius=1)
        self.assertLess(strict["exact_f1"], 0.01)
        self.assertGreater(tolerant["best_bf1"], 0.99)

        ignore = np.zeros_like(gt)
        ignore[:, 890:] = True
        noisy = pred.copy()
        noisy[100:400, 950] = 1.0
        unignored = compute_edge_label_score(noisy, gt, tolerance_radius=0)
        ignored = compute_edge_label_score(noisy, gt, ignore_mask=ignore, tolerance_radius=0)
        self.assertGreater(ignored["precision"], unignored["precision"])

    def test_c_training_grid_uses_expected_model_loss_optimizer_scheduler(self):
        self.assertIn("teed_1px_default", edge_training.MODEL_PRESETS)
        self.assertIn("dexined_1px_accuracy", edge_training.MODEL_PRESETS)
        cfg = edge_training.TrainConfig()
        self.assertEqual(cfg.model_preset, "teed_1px_default")
        self.assertEqual(cfg.bce_weight, 1.0)
        self.assertEqual(cfg.dice_weight, 1.0)
        self.assertGreater(cfg.cldice_weight, 0.0)

        source = inspect.getsource(edge_training.train_edge_model)
        self.assertIn("torch.optim.AdamW", source)
        self.assertIn("torch.optim.lr_scheduler.OneCycleLR", source)
        self.assertIn("total_neg / max(total_pos, 1.0)", source)

        loss_source = inspect.getsource(edge_training._edge_loss)
        self.assertIn("binary_cross_entropy_with_logits", loss_source)
        self.assertIn("_dice_loss", loss_source)
        self.assertIn("_soft_cldice_loss", loss_source)

    def test_d_gui_routing_grid_default_deep_and_classical_fallback_are_separate(self):
        default_source = inspect.getsource(EdgeBatchGUI._start_auto_optimize)
        self.assertIn("load_edge_labels", default_source)
        self.assertIn("optimize_edge_training", default_source)
        self.assertIn("1px Labels Required", default_source)
        self.assertNotIn("_get_auto_config", default_source)
        self.assertNotIn("_ask_target_score", default_source)

        classical_source = inspect.getsource(EdgeBatchGUI._start_classical_auto_optimize)
        self.assertIn("_get_auto_config", classical_source)
        self.assertIn("_ask_target_score", classical_source)
        self.assertIn("_auto_optimize_worker", classical_source)

        editor_source = inspect.getsource(EdgeBatchGUI._open_edge_label_editor)
        self.assertIn("save_edge_label", editor_source)
        self.assertIn("Pixel snap", editor_source)
        self.assertIn("Finish Line", editor_source)
        self.assertIn("Save 1px GT", editor_source)


if __name__ == "__main__":
    unittest.main()
