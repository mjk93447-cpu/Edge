"""
GPU acceleration benchmark: synthetic images + mini auto-optimization.
Compares CPU vs GPU (CuPy) execution time for detect_edges_array and evaluate_one_candidate_mp.
"""
import os
import sys
import time
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sobel_edge_detection import (
    SobelEdgeDetector,
    evaluate_one_candidate_mp,
    PARAM_DEFAULTS,
    AUTO_DEFAULTS,
    is_gpu_available,
    _CUPY_AVAILABLE,
)

OUT_DIR = "gpu_benchmark_out"


def make_synthetic_images(count=20, sizes=None):
    """Generate synthetic test images: rectangles, circles, mixed."""
    if sizes is None:
        sizes = [(64, 64), (128, 128), (256, 256), (128, 256), (256, 128)]
    images = []
    rng = np.random.default_rng(42)
    for i in range(count):
        h, w = sizes[i % len(sizes)]
        img = np.zeros((h, w), dtype=np.float32)
        # Rectangle or ellipse
        mh, mw = h // 4, w // 4
        if i % 2 == 0:
            img[mh : h - mh, mw : w - mw] = 180.0 + rng.uniform(0, 40, (h - 2 * mh, w - 2 * mw))
        else:
            yy, xx = np.ogrid[:h, :w]
            cy, cx = h // 2, w // 2
            ry, rx = h // 3, w // 3
            mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            img[mask] = 180.0 + rng.uniform(0, 40, mask.sum())
        images.append(img)
    return images


def build_eval_data(images, band_radius=2):
    """Build data items for evaluate_one_candidate_mp."""
    detector = SobelEdgeDetector()
    data = []
    for img in images:
        h, w = img.shape
        boundary = np.zeros((h, w), dtype=bool)
        margin = max(2, min(h, w) // 8)
        boundary[margin, margin : w - margin] = True
        boundary[h - 1 - margin, margin : w - margin] = True
        boundary[margin : h - margin, margin] = True
        boundary[margin : h - margin, w - 1 - margin] = True
        band = detector.dilate_binary(boundary, band_radius)
        band_pixels = int(band.sum())
        data.append({
            "image": img,
            "mask": np.ones((h, w), dtype=bool),
            "boundary": boundary,
            "bands": {band_radius: band, 0: boundary},
            "band_pixels": {band_radius: band_pixels, 0: int(boundary.sum())},
            "weight": 1.0,
        })
    return data


def run_detect_benchmark(detector, images, settings, use_gpu, warmup=2, runs=20):
    """Measure detect_edges_array time."""
    for _ in range(warmup):
        detector.detect_edges_array(images[0], **settings)
    t0 = time.perf_counter()
    for _ in range(runs):
        for img in images:
            detector.detect_edges_array(img, **settings)
    t1 = time.perf_counter()
    return t1 - t0


def run_evaluate_benchmark(data, settings, use_gpu, warmup=1, runs=10):
    """Measure evaluate_one_candidate_mp time for one candidate."""
    settings = dict(settings)
    settings["use_gpu"] = use_gpu
    settings["use_boundary_band_filter"] = False
    detector = SobelEdgeDetector()
    for _ in range(warmup):
        evaluate_one_candidate_mp(data, settings, AUTO_DEFAULTS)
    t0 = time.perf_counter()
    for _ in range(runs):
        evaluate_one_candidate_mp(data, settings, AUTO_DEFAULTS)
    t1 = time.perf_counter()
    return t1 - t0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== GPU Acceleration Benchmark ===\n")
    print(f"CuPy available: {_CUPY_AVAILABLE}")
    print(f"GPU available:  {is_gpu_available()}")

    settings = dict(PARAM_DEFAULTS)
    settings["use_nms"] = True
    settings["use_hysteresis"] = True
    settings["boundary_band_radius"] = 2

    # 1) detect_edges_array benchmark
    images = make_synthetic_images(count=20)
    detector = SobelEdgeDetector()

    t_cpu = run_detect_benchmark(detector, images, {**settings, "use_gpu": False}, use_gpu=False)
    print(f"\n[detect_edges_array] CPU: {t_cpu:.3f}s for {len(images) * 20} runs")

    t_gpu = None
    if is_gpu_available():
        t_gpu = run_detect_benchmark(detector, images, {**settings, "use_gpu": True}, use_gpu=True)
        speedup_detect = t_cpu / t_gpu if t_gpu > 0 else 0
        print(f"[detect_edges_array] GPU: {t_gpu:.3f}s  Speedup: {speedup_detect:.2f}x")
    else:
        print("[detect_edges_array] GPU: skipped (CuPy/CUDA not available)")

    # 2) evaluate_one_candidate_mp (Auto optimization unit)
    data = build_eval_data(images)
    t_eval_cpu = run_evaluate_benchmark(data, settings, False)
    print(f"\n[evaluate_one_candidate] CPU: {t_eval_cpu:.3f}s for 10 runs")

    t_eval_gpu = None
    if is_gpu_available():
        t_eval_gpu = run_evaluate_benchmark(data, settings, True)
        speedup_eval = t_eval_cpu / t_eval_gpu if t_eval_gpu > 0 else 0
        print(f"[evaluate_one_candidate] GPU: {t_eval_gpu:.3f}s  Speedup: {speedup_eval:.2f}x")
    else:
        print("[evaluate_one_candidate] GPU: skipped")

    # Report
    report = {
        "cupy_available": _CUPY_AVAILABLE,
        "gpu_available": is_gpu_available(),
        "detect_cpu_sec": t_cpu,
        "detect_gpu_sec": t_gpu,
        "detect_speedup": float(t_cpu / t_gpu) if t_gpu and t_gpu > 0 else None,
        "evaluate_cpu_sec": t_eval_cpu,
        "evaluate_gpu_sec": t_eval_gpu,
        "evaluate_speedup": float(t_eval_cpu / t_eval_gpu) if t_eval_gpu and t_eval_gpu > 0 else None,
    }
    path = os.path.join(OUT_DIR, "gpu_benchmark_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {path}")

    if t_gpu and t_gpu > 0:
        print(f"\n[RESULT] GPU acceleration provides ~{t_cpu/t_gpu:.1f}x speedup for edge detection.")
    else:
        print("\n[RESULT] Install CuPy (e.g. pip install cupy-cuda12x) for GPU acceleration.")

    return report


if __name__ == "__main__":
    main()
