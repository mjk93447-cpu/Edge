import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

import edge_training
from edge_labeling import save_edge_label


def _make_records(tmpdir, count=2, size=32):
    records = []
    for idx in range(count):
        arr = np.zeros((size, size), dtype=np.uint8)
        arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 180
        path = os.path.join(tmpdir, f"img_{idx}.png")
        Image.fromarray(arr).save(path)
        rec = save_edge_label(
            path,
            (0, 0, size, size),
            [{"points": [(4, size // 2 + idx % 2), (size - 5, size // 2 + idx % 2)]}],
            root=os.path.join(tmpdir, "edge_labels"),
        )
        records.append(rec)
    return records


class EdgeTrainingTest(unittest.TestCase):
    def test_model_presets_present(self):
        for key in ("teed_1px_default", "dexined_1px_accuracy", "edgenat_1px_adapter", "pidinet_1px_fast"):
            self.assertIn(key, edge_training.MODEL_PRESETS)

    @unittest.skipUnless(edge_training._TORCH_AVAILABLE, "PyTorch is not installed")
    def test_teed_cpu_smoke_train(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records(tmpdir, count=2, size=32)
            cfg = edge_training.TrainConfig(
                output_dir=os.path.join(tmpdir, "train"),
                device="cpu",
                epochs=1,
                batch_size=1,
                crop_size=32,
                lr_max=1e-3,
            )
            result = edge_training.train_edge_model(records, cfg)
            self.assertTrue(os.path.exists(result.model_path))
            self.assertTrue(os.path.exists(result.report_path))
            self.assertTrue(os.path.exists(result.threshold_path))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "train", "pr_curve.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "train", "gt_preview_overlay.png")))
            self.assertGreaterEqual(result.best_score, 0.0)
            with open(result.report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            for key in ("best_bf1", "exact_f1", "cldice", "boundary_iou", "merge_split_risk"):
                self.assertIn(key, report["metrics"])
            for key in ("topology_safe_bf1", "topology_safe_threshold", "topology_safe_merge_split_risk"):
                self.assertIn(key, report["metrics"])
            self.assertIn("promotion_gate", report)
            diagnostics = report["diagnostics"]
            self.assertEqual(diagnostics["optimizer"], "AdamW")
            self.assertEqual(diagnostics["scheduler"], "OneCycleLR")
            self.assertEqual(diagnostics["effective_model_preset"], "teed_1px_default")
            self.assertEqual(len(diagnostics["lr_history"]), diagnostics["n_train_records"])
            self.assertGreaterEqual(diagnostics["pos_weight"], 1.0)
            self.assertIn("label_qc", diagnostics)
            with open(result.threshold_path, "r", encoding="utf-8") as handle:
                threshold_payload = json.load(handle)
            self.assertEqual(threshold_payload["threshold_mode"], "topology_safe")

    @unittest.skipUnless(edge_training._TORCH_AVAILABLE, "PyTorch is not installed")
    def test_loss_weights_change_loss_and_create_gradients(self):
        import torch

        logits = torch.zeros((1, 1, 16, 16), requires_grad=True)
        target = torch.zeros((1, 1, 16, 16))
        target[:, :, 8, 3:13] = 1.0
        outputs = {"logits": logits, "sides": []}
        pos_weight = torch.tensor([5.0])

        base_cfg = edge_training.TrainConfig(bce_weight=1.0, dice_weight=1.0, cldice_weight=0.0)
        cldice_cfg = edge_training.TrainConfig(
            bce_weight=1.0,
            dice_weight=1.0,
            cldice_weight=0.1,
            focal_weight=0.1,
            tversky_weight=0.1,
            background_distance_weight=0.1,
        )
        base_loss = edge_training._edge_loss(outputs, target, base_cfg, pos_weight)
        cldice_loss = edge_training._edge_loss(outputs, target, cldice_cfg, pos_weight)
        self.assertNotAlmostEqual(float(base_loss.detach()), float(cldice_loss.detach()))
        cldice_loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_promotion_gate_uses_topology_safe_metrics_when_present(self):
        result = edge_training.evaluate_promotion_gate(
            {
                "best_bf1": 0.95,
                "exact_f1": 0.8,
                "merge_split_risk": 0.9,
                "topology_safe_bf1": 0.86,
                "topology_safe_exact_f1": 0.7,
                "topology_safe_merge_split_risk": 0.02,
            },
            baseline_sobel_bf1=0.70,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["threshold_mode"], "topology_safe")

    @unittest.skipUnless(edge_training._TORCH_AVAILABLE, "PyTorch is not installed")
    def test_hpo_fallback_records_strategy_and_trial_reset_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records(tmpdir, count=2, size=32)
            cfg = edge_training.HPOConfig(
                output_dir=os.path.join(tmpdir, "hpo"),
                trial_count=2,
                max_epochs_per_trial=1,
                device="cpu",
                batch_size=1,
                crop_size=32,
            )
            summary = edge_training.optimize_edge_training(records, cfg)
            self.assertIn(summary["hpo_strategy"], ("random_fallback_no_optuna", "optuna_available_not_enabled_v1"))
            self.assertEqual(len(summary["trials"]), 2)
            self.assertTrue(summary["trial_reset_verified"])
            hashes = [trial["initial_weight_hash"] for trial in summary["trials"]]
            self.assertEqual(hashes[0], hashes[1])
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "hpo", "hpo_summary.json")))

    @unittest.skipUnless(edge_training._TORCH_AVAILABLE, "PyTorch is not installed")
    def test_adapter_presets_fallback_to_teed_without_crash(self):
        for preset in ("dexined_1px_accuracy", "edgenat_1px_adapter", "pidinet_1px_fast"):
            model = edge_training.create_model(preset)
            self.assertIsInstance(model, edge_training.TEED1pxModel)
            self.assertEqual(edge_training._effective_preset(preset), "teed_1px_default")

    def test_promotion_gate_acceptance_and_failure_reasons(self):
        passed = edge_training.evaluate_promotion_gate(
            {"best_bf1": 0.86, "exact_f1": 0.7, "merge_split_risk": 0.02},
            baseline_sobel_bf1=0.70,
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["reasons"], [])

        failed = edge_training.evaluate_promotion_gate(
            {"best_bf1": 0.74, "exact_f1": 0.0, "merge_split_risk": 0.20},
            baseline_sobel_bf1=0.70,
        )
        self.assertFalse(failed["passed"])
        self.assertIn("bf1_below_absolute_or_delta_gate", failed["reasons"])
        self.assertIn("merge_split_risk_high", failed["reasons"])
        self.assertIn("exact_1px_f1_zero", failed["reasons"])

    @unittest.skipUnless(edge_training._TORCH_AVAILABLE, "PyTorch is not installed")
    def test_cuda_request_falls_back_to_cpu_when_unavailable(self):
        import torch

        device = edge_training._resolve_device("cuda")
        if not torch.cuda.is_available():
            self.assertEqual(str(device), "cpu")


if __name__ == "__main__":
    unittest.main()
