import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from sobel_edge_detection import SobelEdgeDetector


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


if __name__ == "__main__":
    unittest.main()
