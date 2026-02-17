"""
영점삼(0.3) 이상 스코어 빠른 도달 테스트 및 3회 평가-재개발 루프.
GUI 없이 데이터 준비 + 최소 평가 루프로 time-to-0.3 및 성능 측정.
"""
import os
import sys
import time
import json
import multiprocessing

import numpy as np
from PIL import Image

from sobel_edge_detection import (
    SobelEdgeDetector,
    AUTO_DEFAULTS,
    PARAM_DEFAULTS,
    compute_auto_score,
    evaluate_one_candidate_mp,
    _eval_candidate_wrapper_mp,
)
from concurrent.futures import ProcessPoolExecutor

TARGET_SCORE = 0.3
TEST_IMAGE_DIR = "test_images_auto"
MAX_FILES_FAST = 6
EVAL_BUDGET_FOR_TARGET = 200
ROUND_SIZE = 25
NUM_LOOPS = 3

# Smoke test: set to True to run only 30 evals per loop (quick check)
SMOKE_TEST = False
if SMOKE_TEST:
    EVAL_BUDGET_FOR_TARGET = 30
    ROUND_SIZE = 10
    NUM_LOOPS = 2


def _compute_boundary(mask):
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if mask[y, x]:
                if not (
                    mask[y - 1, x]
                    and mask[y + 1, x]
                    and mask[y, x - 1]
                    and mask[y, x + 1]
                ):
                    boundary[y, x] = True
    return boundary


def prepare_data(detector, files, base_settings, auto_config, max_files=None):
    if max_files:
        files = files[:max_files]
    band_min = min(auto_config["auto_band_min"], auto_config["auto_band_max"])
    band_max = max(auto_config["auto_band_min"], auto_config["auto_band_max"])
    band_radii = list(range(band_min, band_max + 1))
    if not band_radii:
        band_radii = [base_settings["boundary_band_radius"]]

    data = []
    for path in files:
        image = detector.load_image(path)
        mask_source = image
        if base_settings["use_mask_blur"]:
            mask_source = detector.apply_gaussian_blur(
                image,
                base_settings["mask_blur_kernel_size"],
                base_settings["mask_blur_sigma"],
            )
        mask = detector.estimate_object_mask(
            mask_source, base_settings["object_is_dark"]
        )
        if base_settings["mask_close_radius"] > 0:
            mask = detector.erode_binary(
                detector.dilate_binary(mask, base_settings["mask_close_radius"]),
                base_settings["mask_close_radius"],
            )
        boundary = _compute_boundary(mask)
        bands = {}
        band_pixels = {}
        for radius in band_radii:
            if radius <= 0:
                band = boundary.copy()
            else:
                band = detector.dilate_binary(boundary, radius)
            bands[radius] = band
            band_pixels[radius] = int(band.sum())
        data.append({
            "path": path,
            "image": image,
            "mask": mask,
            "boundary": boundary,
            "bands": bands,
            "band_pixels": band_pixels,
            "weight": 1.0,
        })
    return data


def generate_candidates(rng, base_settings, auto_config, count=50):
    candidates = []
    nms_min = min(auto_config["auto_nms_min"], auto_config["auto_nms_max"])
    nms_max = max(auto_config["auto_nms_min"], auto_config["auto_nms_max"])
    high_min = min(auto_config["auto_high_min"], auto_config["auto_high_max"])
    high_max = max(auto_config["auto_high_min"], auto_config["auto_high_max"])
    low_min = min(auto_config["auto_low_factor_min"], auto_config["auto_low_factor_max"])
    low_max = max(auto_config["auto_low_factor_min"], auto_config["auto_low_factor_max"])
    margin_min = min(auto_config["auto_margin_min"], auto_config["auto_margin_max"])
    margin_max = max(auto_config["auto_margin_min"], auto_config["auto_margin_max"])
    band_min = min(auto_config["auto_band_min"], auto_config["auto_band_max"])
    band_max = max(auto_config["auto_band_min"], auto_config["auto_band_max"])
    blur_sigma_min = min(auto_config["auto_blur_sigma_min"], auto_config["auto_blur_sigma_max"])
    blur_sigma_max = max(auto_config["auto_blur_sigma_min"], auto_config["auto_blur_sigma_max"])
    for _ in range(count):
        s = dict(base_settings)
        s["nms_relax"] = round(float(rng.uniform(nms_min, nms_max)), 3)
        s["high_ratio"] = float(rng.uniform(high_min, high_max))
        low_factor = float(rng.uniform(low_min, low_max))
        s["low_ratio"] = max(0.02, s["high_ratio"] * low_factor)
        s["boundary_band_radius"] = int(rng.randint(band_min, band_max + 1))
        s["polarity_drop_margin"] = float(rng.uniform(margin_min, margin_max))
        s["blur_sigma"] = float(rng.uniform(blur_sigma_min, blur_sigma_max))
        s["blur_kernel_size"] = 5
        s["thinning_max_iter"] = 15
        s["contrast_ref"] = float(rng.uniform(70, 110))
        s["min_threshold_scale"] = float(rng.uniform(0.5, 0.75))
        s["use_boundary_band_filter"] = s["boundary_band_radius"] > 0
        candidates.append(s)
    return candidates


def run_fast_target_loop(
    data_full,
    base_settings,
    auto_config,
    candidate_workers=0,
    eval_budget=EVAL_BUDGET_FOR_TARGET,
    target_score=TARGET_SCORE,
    rng=None,
):
    """목표 스코어(target_score) 도달 시점까지 평가; 시간·평가 횟수 반환."""
    rng = rng or np.random.RandomState(42)
    best_score = 0.0
    best_settings = None
    processed = 0
    start_time = time.perf_counter()
    time_to_target = None

    while processed < eval_budget:
        pool = generate_candidates(rng, base_settings, auto_config, count=ROUND_SIZE)
        if candidate_workers and candidate_workers >= 1:
            batch_size = min(candidate_workers, 8, len(pool))
            batch = pool[:batch_size]
            try:
                with ProcessPoolExecutor(max_workers=len(batch)) as ex:
                    args_list = [(data_full, s, auto_config) for s in batch]
                    results = list(ex.map(_eval_candidate_wrapper_mp, args_list, chunksize=1))
            except Exception as e:
                results = [evaluate_one_candidate_mp(data_full, s, auto_config) for s in batch]
        else:
            batch = pool
            results = [evaluate_one_candidate_mp(data_full, s, auto_config) for s in batch]

        for s, (score, summary, qualities) in zip(batch, results):
            processed += 1
            if best_settings is None or score > best_score:
                best_score = score
                best_settings = s
                if time_to_target is None and score >= target_score:
                    time_to_target = time.perf_counter() - start_time

        if best_score >= target_score:
            break
        if processed >= eval_budget:
            break

    elapsed = time.perf_counter() - start_time
    return {
        "best_score": float(best_score),
        "best_settings": best_settings,
        "processed": processed,
        "elapsed_sec": elapsed,
        "time_to_target_sec": time_to_target,
        "target_reached": best_score >= target_score,
    }


def ensure_test_images(test_dir=TEST_IMAGE_DIR, count=10, size=(400, 300)):
    os.makedirs(test_dir, exist_ok=True)
    detector = SobelEdgeDetector()
    files = []
    for i in range(count):
        path = os.path.join(test_dir, f"test_{i:03d}.png")
        if os.path.isfile(path):
            files.append(path)
            continue
        if i % 3 == 0:
            h, w = size
            img = np.zeros((h, w), dtype=np.uint8)
            cy, cx = h // 2, w // 2
            r = min(h, w) // 4
            y, x = np.ogrid[:h, :w]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
            img[mask] = 200
        elif i % 3 == 1:
            h, w = size
            img = np.zeros((h, w), dtype=np.uint8)
            m = min(h, w) // 5
            img[m : h - m, m : w - m] = 180
        else:
            h, w = size
            img = np.zeros((h, w), dtype=np.uint8)
            for j in range(3):
                cy = h // 4 + (j * h // 4)
                cx = w // 2
                r = min(h, w) // 8
                y, x = np.ogrid[:h, :w]
                mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
                img[mask] = 160 + j * 20
        noise = np.random.randint(-25, 25, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)
        files.append(path)
    return files


def main():
    if sys.platform == "win32":
        multiprocessing.freeze_support()

    detector = SobelEdgeDetector()
    base_settings = dict(PARAM_DEFAULTS)
    auto_config = dict(AUTO_DEFAULTS)
    auto_config["auto_target_score"] = TARGET_SCORE
    auto_config["auto_target_score_rounds_after"] = 1
    candidate_workers = max(0, int(auto_config.get("auto_candidate_workers", 0)))

    files = ensure_test_images(TEST_IMAGE_DIR, count=10)
    files = files[:MAX_FILES_FAST]
    print(f"[SETUP] Using {len(files)} images, target_score={TARGET_SCORE}, candidate_workers={candidate_workers}")

    prep_start = time.perf_counter()
    data_full = prepare_data(detector, files, base_settings, auto_config)
    prep_time = time.perf_counter() - prep_start
    print(f"[SETUP] Data prepared in {prep_time:.3f}s")

    # 3회 테스트-평가-재개발 루프: 매 루프마다 설정을 개선(redevelop) 후 재측정
    redevelop_configs = [
        {"name": "Loop1_baseline", "candidate_workers": 0},
        {"name": "Loop2_parallel_2", "candidate_workers": min(2, (os.cpu_count() or 4) - 1)},
        {"name": "Loop3_parallel_4", "candidate_workers": min(4, max(1, (os.cpu_count() or 4) - 1))},
    ]

    all_loop_results = []
    for loop in range(1, NUM_LOOPS + 1):
        cfg = redevelop_configs[loop - 1]
        workers = cfg["candidate_workers"]
        print(f"\n--- {cfg['name']} (테스트-평가-재개발 루프 {loop}/{NUM_LOOPS}) ---")
        print(f"  Redevelop: candidate_workers={workers}")

        r = run_fast_target_loop(
            data_full,
            base_settings,
            auto_config,
            candidate_workers=workers,
            eval_budget=EVAL_BUDGET_FOR_TARGET,
            target_score=TARGET_SCORE,
            rng=np.random.RandomState(42 + loop),
        )
        rec = {
            "loop": loop,
            "config_name": cfg["name"],
            "candidate_workers": workers,
            "best_score": r["best_score"],
            "processed": r["processed"],
            "elapsed_sec": r["elapsed_sec"],
            "time_to_target_sec": r["time_to_target_sec"],
            "target_reached": r["target_reached"],
        }
        all_loop_results.append(rec)
        print(
            f"  [평가] best_score={r['best_score']:.6f} target_reached={r['target_reached']} "
            f"processed={r['processed']} elapsed={r['elapsed_sec']:.2f}s "
            f"time_to_target={r['time_to_target_sec']}"
        )

    out_path = "fast_target_score_loop_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_score": TARGET_SCORE,
                "prep_time_sec": prep_time,
                "num_loops": NUM_LOOPS,
                "redevelop_configs": [c["name"] for c in redevelop_configs],
                "loop_results": all_loop_results,
            },
            f,
            indent=2,
        )
    print(f"\n[SAVE] {out_path}")

    reached = sum(1 for r in all_loop_results if r["target_reached"])
    avg_time = np.mean([r["elapsed_sec"] for r in all_loop_results])
    avg_evals = np.mean([r["processed"] for r in all_loop_results])
    times_to_target = [r["time_to_target_sec"] for r in all_loop_results if r["time_to_target_sec"] is not None]
    avg_time_to_target = np.mean(times_to_target) if times_to_target else None
    print(f"\n[SUMMARY] target_reached {reached}/{NUM_LOOPS}, avg elapsed={avg_time:.2f}s, avg evals={avg_evals:.0f}")
    if avg_time_to_target is not None:
        print(f"[SUMMARY] avg time_to_target={avg_time_to_target:.2f}s")
    return all_loop_results


if __name__ == "__main__":
    main()
