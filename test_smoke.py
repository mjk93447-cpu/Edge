import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from sobel_edge_detection import (
    AUTO_DEFAULTS,
    PARAM_DEFAULTS,
    SobelEdgeDetector,
    compute_auto_score,
    evaluate_one_candidate_mp,
    get_auto_profile_overrides,
    load_json_config,
    save_json_config,
)


class SobelSmokeTest(unittest.TestCase):
    def test_detect_edges_smoke(self):
        image = np.zeros((12, 12), dtype=np.uint8)
        image[3:9, 3:9] = 255

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample.png")
            Image.fromarray(image).save(path)

            detector = SobelEdgeDetector()
            results = detector.detect_edges(path, use_nms=True, use_hysteresis=True)

        self.assertEqual(results["edges"].shape, image.shape)
        self.assertEqual(results["edges"].dtype, np.uint8)

    def test_param_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "params.json")
            payload = {"params": dict(PARAM_DEFAULTS)}
            save_json_config(path, payload)
            loaded = load_json_config(path)
        self.assertEqual(loaded["params"], payload["params"])

    def test_auto_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "auto.json")
            payload = {"auto": dict(AUTO_DEFAULTS)}
            save_json_config(path, payload)
            loaded = load_json_config(path)
        self.assertEqual(loaded["auto"], payload["auto"])

    def test_compute_auto_score_positive(self):
        metrics = {
            "coverage": 0.6,
            "gap": 0.2,
            "continuity": 0.1,
            "intrusion": 0.05,
            "outside": 0.03,
            "thickness": 0.1,
        }
        score = compute_auto_score(metrics, AUTO_DEFAULTS)
        self.assertGreater(score, 0.0)

    def test_get_auto_profile_overrides(self):
        self.assertEqual(get_auto_profile_overrides(None, None), {})
        o = get_auto_profile_overrides(True, None)
        self.assertIn("auto_contrast_ref_min", o)
        self.assertIn("auto_contrast_ref_max", o)
        o2 = get_auto_profile_overrides(None, True)
        self.assertIn("auto_blur_sigma_min", o2)
        self.assertIn("auto_nms_min", o2)
        o3 = get_auto_profile_overrides(False, False)
        self.assertIn("auto_contrast_ref_min", o3)
        self.assertIn("auto_blur_sigma_step", o3)

    def test_evaluate_one_candidate_mp(self):
        image = np.zeros((24, 24), dtype=np.float32)
        image[6:18, 6:18] = 200.0
        mask = np.ones((24, 24), dtype=bool)
        boundary = np.zeros((24, 24), dtype=bool)
        boundary[6, 6:18] = True
        boundary[17, 6:18] = True
        boundary[6:18, 6] = True
        boundary[6:18, 17] = True
        band_radius = 2
        band = np.zeros((24, 24), dtype=bool)
        band[4:20, 4:20] = True
        data = [
            {
                "image": image,
                "mask": mask,
                "boundary": boundary,
                "bands": {band_radius: band, 0: boundary},
                "band_pixels": {band_radius: int(band.sum()), 0: int(boundary.sum())},
                "weight": 1.0,
            }
        ]
        settings = dict(PARAM_DEFAULTS)
        settings.update({
            "nms_relax": 0.95,
            "high_ratio": 0.1,
            "low_ratio": 0.04,
            "boundary_band_radius": band_radius,
            "polarity_drop_margin": 0.2,
            "use_boundary_band_filter": True,
            "blur_sigma": 1.0,
            "blur_kernel_size": 5,
            "thinning_max_iter": 10,
            "use_thinning": False,
        })
        score, summary, qualities = evaluate_one_candidate_mp(data, settings, AUTO_DEFAULTS)
        self.assertIsInstance(score, (int, float))
        self.assertIn("coverage", summary)
        self.assertIn("q_cont", qualities)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
