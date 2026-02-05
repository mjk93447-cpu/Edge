import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from sobel_edge_detection import (
    AUTO_DEFAULTS,
    PARAM_DEFAULTS,
    SobelEdgeDetector,
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


if __name__ == "__main__":
    unittest.main()
