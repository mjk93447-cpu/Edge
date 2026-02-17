"""
사용자 목표 점수(0.3/0.4/0.5/0.6 등) 입력 → 해당 점수 달성을 위한 전략 자동 튜닝.
영점사(0.4), 영점오(0.5), 영점육(0.6) 각각 다른 전략으로 학습 후,
최소 3회의 [테스트 → 평가·현황 파악 → 개선 전략 도출 → 코드/설정 수정 → 테스트] 루프 수행.
"""
import os
import sys
import time
import json
import multiprocessing
from datetime import datetime

import numpy as np
from PIL import Image

from sobel_edge_detection import (
    SobelEdgeDetector,
    AUTO_DEFAULTS,
    PARAM_DEFAULTS,
    get_strategy_for_target_score,
    evaluate_one_candidate_mp,
    _eval_candidate_wrapper_mp,
)
from concurrent.futures import ProcessPoolExecutor

TEST_IMAGE_DIR = "test_images_auto"
MAX_FILES = 6
TARGETS_TO_TEST = [0.4, 0.5, 0.6]
NUM_IMPROVEMENT_LOOPS = 3
STRATEGY_CONFIG_PATH = "target_score_strategy_config.json"

# 빠른 검증: --quick 시 루프당 예산 축소
QUICK_EVAL_BUDGET = 60
QUICK_ROUND_SIZE = 15


def _compute_boundary(mask):
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if mask[y, x]:
                if not (
                    mask[y - 1, x] and mask[y + 1, x] and mask[y, x - 1] and mask[y, x + 1]
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
        mask_source = detector.apply_gaussian_blur(
            image,
            base_settings["mask_blur_kernel_size"],
            base_settings["mask_blur_sigma"],
        ) if base_settings["use_mask_blur"] else image
        mask = detector.estimate_object_mask(mask_source, base_settings["object_is_dark"])
        if base_settings["mask_close_radius"] > 0:
            mask = detector.erode_binary(
                detector.dilate_binary(mask, base_settings["mask_close_radius"]),
                base_settings["mask_close_radius"],
            )
        boundary = _compute_boundary(mask)
        bands = {}
        band_pixels = {}
        for radius in band_radii:
            band = boundary if radius <= 0 else detector.dilate_binary(boundary, radius)
            bands[radius] = band
            band_pixels[radius] = int(band.sum())
        data.append({
            "path": path, "image": image, "mask": mask, "boundary": boundary,
            "bands": bands, "band_pixels": band_pixels, "weight": 1.0,
        })
    return data


def generate_candidates(rng, base_settings, auto_config, count):
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
    candidates = []
    for _ in range(count):
        s = dict(base_settings)
        s["nms_relax"] = round(float(rng.uniform(nms_min, nms_max)), 3)
        s["high_ratio"] = float(rng.uniform(high_min, high_max))
        s["low_ratio"] = max(0.02, s["high_ratio"] * float(rng.uniform(low_min, low_max)))
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


def run_target_loop(
    data_full,
    base_settings,
    auto_config,
    target_score,
    eval_budget,
    round_size,
    candidate_workers,
    rng,
):
    best_score = 0.0
    best_settings = None
    processed = 0
    start_time = time.perf_counter()
    time_to_target = None

    while processed < eval_budget:
        pool = generate_candidates(rng, base_settings, auto_config, count=round_size)
        batch_size = min(max(1, candidate_workers), 8, len(pool)) if candidate_workers else len(pool)
        batch = pool[:batch_size]
        if candidate_workers and candidate_workers >= 1:
            try:
                with ProcessPoolExecutor(max_workers=len(batch)) as ex:
                    args_list = [(data_full, s, auto_config) for s in batch]
                    results = list(ex.map(_eval_candidate_wrapper_mp, args_list, chunksize=1))
            except Exception:
                results = [evaluate_one_candidate_mp(data_full, s, auto_config) for s in batch]
        else:
            results = [evaluate_one_candidate_mp(data_full, s, auto_config) for s in batch]

        for s, (score, summary, qualities) in zip(batch, results):
            processed += 1
            if best_settings is None or score > best_score:
                best_score = score
                best_settings = s
                if time_to_target is None and score >= target_score:
                    time_to_target = time.perf_counter() - start_time
        if best_score >= target_score or processed >= eval_budget:
            break

    return {
        "best_score": float(best_score),
        "best_settings": best_settings,
        "processed": processed,
        "elapsed_sec": time.perf_counter() - start_time,
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
        h, w = size
        img = np.zeros((h, w), dtype=np.uint8)
        if i % 3 == 0:
            cy, cx = h // 2, w // 2
            r = min(h, w) // 4
            y, x = np.ogrid[:h, :w]
            img[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = 200
        elif i % 3 == 1:
            m = min(h, w) // 5
            img[m : h - m, m : w - m] = 180
        else:
            for j in range(3):
                cy, cx = h // 4 + (j * h // 4), w // 2
                r = min(h, w) // 8
                y, x = np.ogrid[:h, :w]
                img[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = 160 + j * 20
        noise = np.random.randint(-25, 25, img.shape, dtype=np.int16)
        Image.fromarray(np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)).save(path)
        files.append(path)
    return files


def load_strategy_overrides():
    if os.path.isfile(STRATEGY_CONFIG_PATH):
        try:
            with open(STRATEGY_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_strategy_overrides(overrides):
    with open(STRATEGY_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)


def ask_target_score():
    print("\n목표 점수를 입력하세요 (0.3 / 0.4 / 0.5 / 0.6 또는 다른 소수, Enter=0.4): ", end="")
    try:
        line = sys.stdin.readline().strip() or "0.4"
        return float(line)
    except ValueError:
        return 0.4


def run_single_target_test(data_full, base_settings, auto_config, target_score, strategy_overrides, candidate_workers, run_id, quick=False):
    strategy = get_strategy_for_target_score(target_score)
    for k, v in strategy.items():
        if not k.startswith("_"):
            auto_config[k] = v
    eval_budget = strategy.get("_cli_eval_budget", 400)
    round_size = strategy.get("_cli_round_size", 35)
    if quick:
        eval_budget = QUICK_EVAL_BUDGET
        round_size = QUICK_ROUND_SIZE
    overrides = strategy_overrides.get(str(target_score), {})
    eval_budget = overrides.get("eval_budget", eval_budget)
    round_size = overrides.get("round_size", round_size)

    rng = np.random.RandomState(run_id + int(target_score * 1000))
    r = run_target_loop(
        data_full, base_settings, auto_config,
        target_score=target_score,
        eval_budget=eval_budget,
        round_size=round_size,
        candidate_workers=candidate_workers,
        rng=rng,
    )
    return {
        "target_score": target_score,
        "strategy_name": strategy.get("strategy_name", ""),
        "eval_budget": eval_budget,
        "round_size": round_size,
        "best_score": r["best_score"],
        "processed": r["processed"],
        "elapsed_sec": r["elapsed_sec"],
        "time_to_target_sec": r["time_to_target_sec"],
        "target_reached": r["target_reached"],
    }


def evaluate_and_derive_improvement(iteration_results):
    """현황 파악 및 개선 전략 도출: 도달 실패/느린 목표에 예산·라운드 보정."""
    by_target = {}
    for rec in iteration_results:
        t = rec["target_score"]
        if t not in by_target:
            by_target[t] = []
        by_target[t].append(rec)

    improvements = {}
    for target, list_rec in by_target.items():
        reached = sum(1 for r in list_rec if r["target_reached"])
        avg_time = np.mean([r["elapsed_sec"] for r in list_rec])
        avg_evals = np.mean([r["processed"] for r in list_rec])
        times_to_target = [r["time_to_target_sec"] for r in list_rec if r.get("time_to_target_sec") is not None]
        avg_tt = np.mean(times_to_target) if times_to_target else None

        current_budget = list_rec[0].get("eval_budget", 400)
        current_round = list_rec[0].get("round_size", 35)

        if reached < len(list_rec):
            eval_budget = min(1200, int(current_budget * 1.4))
            round_size = min(50, current_round + 5)
            improvements[str(target)] = {"eval_budget": eval_budget, "round_size": round_size}
        elif avg_tt and avg_tt > 120:
            eval_budget = max(100, int(current_budget * 0.9))
            round_size = max(20, current_round - 2)
            improvements[str(target)] = {"eval_budget": eval_budget, "round_size": round_size}
        else:
            improvements[str(target)] = {"eval_budget": current_budget, "round_size": current_round}

    return improvements


def main():
    if sys.platform == "win32":
        multiprocessing.freeze_support()

    interactive = "--batch" not in sys.argv
    quick = "--quick" in sys.argv
    if quick:
        print("[QUICK] 예산 축소 모드로 실행합니다.")
    if interactive:
        user_target = ask_target_score()
        targets_to_run = [user_target]
        print(f"목표 점수 {user_target} 에 맞춘 전략으로 1회 테스트를 실행합니다.")
    else:
        targets_to_run = TARGETS_TO_TEST
        print(f"배치 모드: 목표 점수 {targets_to_run} 각각 다른 전략으로 테스트 후, {NUM_IMPROVEMENT_LOOPS}회 개선 루프 진행.")

    detector = SobelEdgeDetector()
    base_settings = dict(PARAM_DEFAULTS)
    files = ensure_test_images(TEST_IMAGE_DIR, count=10)
    files = files[:MAX_FILES]
    candidate_workers = min(4, max(0, (os.cpu_count() or 4) - 1))

    prep_start = time.perf_counter()
    auto_config = dict(AUTO_DEFAULTS)
    data_full = prepare_data(detector, files, base_settings, auto_config)
    prep_time = time.perf_counter() - prep_start
    print(f"[SETUP] 이미지 {len(files)}장, 데이터 준비 {prep_time:.3f}s")

    if interactive:
        strategy = get_strategy_for_target_score(user_target)
        for k, v in strategy.items():
            if not k.startswith("_"):
                auto_config[k] = v
        eval_budget = strategy.get("_cli_eval_budget", 400)
        round_size = strategy.get("_cli_round_size", 35)
        r = run_target_loop(
            data_full, base_settings, auto_config,
            target_score=user_target,
            eval_budget=eval_budget,
            round_size=round_size,
            candidate_workers=candidate_workers,
            rng=np.random.RandomState(42),
        )
        print(f"[결과] best_score={r['best_score']:.6f} target_reached={r['target_reached']} "
              f"processed={r['processed']} elapsed={r['elapsed_sec']:.2f}s time_to_target={r['time_to_target_sec']}")
        return

    strategy_overrides = load_strategy_overrides()
    all_iterations = []

    for iteration in range(1, NUM_IMPROVEMENT_LOOPS + 1):
        print(f"\n========== 개선 루프 {iteration}/{NUM_IMPROVEMENT_LOOPS} ==========")
        iteration_results = []
        for target in targets_to_run:
            print(f"  목표 {target} ... ", end="", flush=True)
            rec = run_single_target_test(
                data_full, base_settings, dict(auto_config),
                target, strategy_overrides, candidate_workers,
                run_id=iteration * 100 + int(target * 100),
                quick=quick,
            )
            iteration_results.append(rec)
            print(f"reached={rec['target_reached']} score={rec['best_score']:.4f} "
                  f"evals={rec['processed']} time={rec['elapsed_sec']:.1f}s")
        all_iterations.append({"iteration": iteration, "results": iteration_results})

        improvements = evaluate_and_derive_improvement(iteration_results)
        for t, params in improvements.items():
            strategy_overrides[t] = params
        save_strategy_overrides(strategy_overrides)
        print(f"  [개선 전략 반영] {improvements}")

    out_path = f"target_tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "targets": targets_to_run,
            "num_loops": NUM_IMPROVEMENT_LOOPS,
            "prep_time_sec": prep_time,
            "strategy_overrides_final": strategy_overrides,
            "iterations": all_iterations,
        }, f, indent=2)
    print(f"\n[SAVE] {out_path}")

    for target in targets_to_run:
        recs = [r for it in all_iterations for r in it["results"] if r["target_score"] == target]
        reached = sum(1 for r in recs if r["target_reached"])
        times = [r["time_to_target_sec"] for r in recs if r.get("time_to_target_sec") is not None]
        print(f"목표 {target}: 도달 {reached}/{len(recs)}회, 평균 time_to_target={np.mean(times):.2f}s" if times else f"목표 {target}: 도달 {reached}/{len(recs)}회")
    return all_iterations


if __name__ == "__main__":
    main()
