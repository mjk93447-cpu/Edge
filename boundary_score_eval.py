"""
합성 테스트 이미지의 실제 윤곽선(GT)과 검출 테두리를 비교하여
테두리 감지에 최적화된 score 함수를 테스트/평가/전략도출/코드업데이트 4회 이상 루프로 완성.
목표: 끊어지지 않은 하나의 얇은 연결된 선이 경계를 정확히 따라가는 것.
"""
import os
import sys
import json
import math
import multiprocessing
from datetime import datetime

import numpy as np
from PIL import Image

from sobel_edge_detection import (
    SobelEdgeDetector,
    PARAM_DEFAULTS,
    compute_auto_score,
    compute_boundary_optimized_score,
    _count_components_mask,
)
from edge_performance_eval import evaluate_edges, dilate, compute_boundary

# 루프별로 조정할 boundary score 가중치 (전략 도출 후 반영)
BOUNDARY_SCORE_WEIGHTS = {
    "w_align": 0.45,
    "w_thin": 0.25,
    "w_conn": 0.20,
    "w_line": 0.10,
    "connectivity_penalty_per_component": 0.25,
    "endpoint_penalty": 0.4,
    "branch_penalty": 0.25,
}
NUM_IMPROVEMENT_LOOPS = 4
OUTPUT_DIR = "boundary_score_eval_out"
TOLERANCE = 1


def make_gt_boundary_from_mask(mask):
    """마스크에서 1픽셀 얇은 경계선(GT) 계산."""
    return compute_boundary(mask.astype(np.uint8))


def generate_synthetic_with_gt(size=(400, 300), seed=42):
    """합성 이미지 + GT 경계 생성. (image, mask, gt_boundary) 반환."""
    rng = np.random.RandomState(seed)
    h, w = size
    # 패턴별 마스크 생성
    idx = seed % 3
    mask = np.zeros((h, w), dtype=bool)
    if idx == 0:
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = True
    elif idx == 1:
        m = min(h, w) // 5
        mask[m : h - m, m : w - m] = True
    else:
        for j in range(3):
            cy, cx = h // 4 + (j * h // 4), w // 2
            r = min(h, w) // 8
            y, x = np.ogrid[:h, :w]
            mask[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = True

    img = np.zeros((h, w), dtype=np.float32)
    img[mask] = 180 if idx == 1 else 200
    img[~mask] = 30
    noise = rng.randn(h, w).astype(np.float32) * 12
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    gt_boundary = make_gt_boundary_from_mask(mask)
    return img, mask, gt_boundary


def compute_gt_metrics(pred_edges, gt_boundary, detector):
    """GT 대비 정렬·얇음·연결성 메트릭."""
    pred = pred_edges.astype(bool)
    gt = gt_boundary.astype(bool)
    gt_pixels = int(gt.sum())
    pred_pixels = int(pred.sum())

    ev = evaluate_edges(pred, gt, tolerance=TOLERANCE)
    recall_gt = ev["recall"]
    precision_gt = ev["precision"]
    f1_gt = ev["f1"]

    thinness = min(1.0, gt_pixels / max(pred_pixels, 1))

    pred_in_band = pred
    components = _count_components_mask(pred_in_band)
    n_components = components

    neighbor_counts = detector._neighbor_count(pred)
    endpoint_count = int((pred & (neighbor_counts <= 1)).sum())
    branch_count = int((pred & (neighbor_counts >= 3)).sum())
    endpoint_ratio = endpoint_count / max(pred_pixels, 1)
    branch_ratio = branch_count / max(pred_pixels, 1)

    return {
        "recall_gt": recall_gt,
        "precision_gt": precision_gt,
        "f1_gt": f1_gt,
        "thinness": thinness,
        "n_components": n_components,
        "endpoint_ratio": endpoint_ratio,
        "branch_ratio": branch_ratio,
        "pred_pixels": pred_pixels,
        "gt_pixels": gt_pixels,
        "tp": ev["tp"],
        "fp": ev["fp"],
        "fn": ev["fn"],
    }


def boundary_optimized_score_from_weights(metrics_gt, weights):
    """가중치를 인자로 받아 boundary 최적화 점수 계산 (루프별 튜닝용)."""
    f1_gt = float(metrics_gt.get("f1_gt", 0.0))
    thinness = float(metrics_gt.get("thinness", 0.0))
    n_components = int(metrics_gt.get("n_components", 1))
    endpoint_ratio = float(metrics_gt.get("endpoint_ratio", 0.0))
    branch_ratio = float(metrics_gt.get("branch_ratio", 0.0))

    alignment = f1_gt
    c_pen = float(weights.get("connectivity_penalty_per_component", 0.25))
    connectivity = 1.0 if n_components <= 1 else max(0.0, 1.0 - c_pen * (n_components - 1))
    e_pen = float(weights.get("endpoint_penalty", 0.4))
    b_pen = float(weights.get("branch_penalty", 0.25))
    single_line = max(0.0, 1.0 - e_pen * endpoint_ratio - b_pen * branch_ratio)

    w_align = float(weights.get("w_align", 0.45))
    w_thin = float(weights.get("w_thin", 0.25))
    w_conn = float(weights.get("w_conn", 0.20))
    w_line = float(weights.get("w_line", 0.10))
    score = w_align * alignment + w_thin * thinness + w_conn * connectivity + w_line * single_line
    return max(0.0, min(1.0, score))


def run_one_candidate(detector, image, gt_boundary, settings, weights=None):
    """한 설정으로 검출 후 GT 메트릭 + boundary score 반환."""
    w = weights if weights is not None else BOUNDARY_SCORE_WEIGHTS
    kwargs = dict(PARAM_DEFAULTS)
    kwargs.update({k: v for k, v in settings.items() if k in PARAM_DEFAULTS})
    kwargs["use_boundary_band_filter"] = False
    res = detector.detect_edges_array(image, use_nms=True, use_hysteresis=True, **kwargs)
    pred = (res["edges"] > 0).astype(bool)
    metrics_gt = compute_gt_metrics(pred, gt_boundary, detector)
    boundary_score = boundary_optimized_score_from_weights(metrics_gt, w)
    return boundary_score, metrics_gt, pred


def save_overlay(image, pred, gt_boundary, path):
    """합성 이미지 + 예측(녹색) + GT(빨강) 오버레이 저장."""
    h, w = image.shape
    out = np.stack([image, image, image], axis=-1).astype(np.uint8)
    out = np.clip(out, 0, 255)
    out[pred] = [0, 255, 0]
    out[gt_boundary] = [255, 0, 0]
    overlap = pred & gt_boundary
    out[overlap] = [255, 255, 0]
    Image.fromarray(out).save(path)


def evaluate_and_derive_strategy(loop_results):
    """현황 파악 후 가중치 조정 제안 (얇은 연결된 하나의 선 = alignment + thinness + connectivity + single_line)."""
    f1_list = [r["metrics_gt"]["f1_gt"] for r in loop_results]
    thin_list = [r["metrics_gt"]["thinness"] for r in loop_results]
    n_comp_list = [r["metrics_gt"]["n_components"] for r in loop_results]
    ep_list = [r["metrics_gt"]["endpoint_ratio"] for r in loop_results]
    br_list = [r["metrics_gt"]["branch_ratio"] for r in loop_results]

    avg_f1 = np.mean(f1_list)
    avg_thin = np.mean(thin_list)
    avg_comp = np.mean(n_comp_list)
    avg_ep = np.mean(ep_list)
    avg_br = np.mean(br_list)
    improvements = dict(BOUNDARY_SCORE_WEIGHTS)

    if avg_f1 < 0.25:
        improvements["w_align"] = min(0.58, improvements["w_align"] + 0.06)
        improvements["w_thin"] = max(0.12, improvements["w_thin"] - 0.04)
    elif avg_f1 > 0.5:
        improvements["w_align"] = max(0.35, improvements["w_align"] - 0.03)
    if avg_thin < 0.35:
        improvements["w_thin"] = min(0.38, improvements["w_thin"] + 0.06)
    elif avg_thin > 0.7:
        improvements["w_thin"] = max(0.18, improvements["w_thin"] - 0.04)
    if avg_comp > 2.5:
        improvements["connectivity_penalty_per_component"] = min(0.35, improvements["connectivity_penalty_per_component"] + 0.02)
        improvements["w_conn"] = min(0.28, improvements["w_conn"] + 0.02)
    if avg_comp <= 2 and np.mean([r["boundary_score"] for r in loop_results]) > 0.75:
        improvements["connectivity_penalty_per_component"] = max(0.15, improvements["connectivity_penalty_per_component"] - 0.02)
    if avg_ep > 0.15:
        improvements["endpoint_penalty"] = min(0.5, improvements["endpoint_penalty"] + 0.04)
    if avg_br > 0.12:
        improvements["branch_penalty"] = min(0.35, improvements["branch_penalty"] + 0.04)
    total_w = improvements["w_align"] + improvements["w_thin"] + improvements["w_conn"] + improvements["w_line"]
    if abs(total_w - 1.0) > 0.01:
        for k in ["w_align", "w_thin", "w_conn", "w_line"]:
            improvements[k] /= total_w
    return improvements


def main():
    if sys.platform == "win32":
        multiprocessing.freeze_support()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detector = SobelEdgeDetector()
    base = dict(PARAM_DEFAULTS)

    # 합성 이미지 3종 (원, 사각, 3원)
    cases = [
        generate_synthetic_with_gt(seed=0),
        generate_synthetic_with_gt(seed=1),
        generate_synthetic_with_gt(seed=2),
    ]
    # 후보 설정 소수 (빠른 루프)
    candidates = []
    for nms in [0.92, 0.95]:
        for high in [0.09, 0.12]:
            s = dict(base, nms_relax=nms, high_ratio=high, low_ratio=high * 0.35,
                     boundary_band_radius=2, polarity_drop_margin=0.3, blur_sigma=1.2)
            candidates.append(s)

    all_loop_history = []
    weights = dict(BOUNDARY_SCORE_WEIGHTS)

    for loop in range(1, NUM_IMPROVEMENT_LOOPS + 1):
        print(f"\n========== 루프 {loop}/{NUM_IMPROVEMENT_LOOPS} ==========")
        loop_results = []
        for case_idx, (image, mask, gt_boundary) in enumerate(cases):
            best_boundary = -1.0
            best_metrics = None
            best_pred = None
            for s in candidates:
                b_score, m_gt, pred = run_one_candidate(detector, image, gt_boundary, s, weights=weights)
                if b_score > best_boundary:
                    best_boundary = b_score
                    best_metrics = m_gt
                    best_pred = pred
            loop_results.append({
                "case_idx": case_idx,
                "boundary_score": best_boundary,
                "metrics_gt": best_metrics,
                "pred": best_pred,
            })
            overlay_path = os.path.join(OUTPUT_DIR, f"loop{loop}_case{case_idx}_overlay.png")
            save_overlay(image, best_pred, gt_boundary, overlay_path)
            print(f"  case{case_idx} boundary_score={best_boundary:.4f} f1_gt={best_metrics['f1_gt']:.4f} "
                  f"thin={best_metrics['thinness']:.4f} n_comp={best_metrics['n_components']} -> {overlay_path}")

        all_loop_history.append({"loop": loop, "results": [{"boundary_score": r["boundary_score"], "metrics_gt": r["metrics_gt"]} for r in loop_results]})
        improvements = evaluate_and_derive_strategy(loop_results)
        for k, v in improvements.items():
            weights[k] = v
            BOUNDARY_SCORE_WEIGHTS[k] = v
        print(f"  [전략 도출] weights -> {weights}")

    out_path = os.path.join(OUTPUT_DIR, f"boundary_score_loops_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_loops": NUM_IMPROVEMENT_LOOPS,
            "final_weights": BOUNDARY_SCORE_WEIGHTS,
            "loop_history": all_loop_history,
        }, f, indent=2)
    print(f"\n[SAVE] {out_path}")

    weights_path = "boundary_score_weights.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(BOUNDARY_SCORE_WEIGHTS, f, indent=2)
    print(f"[SAVE] 기본 가중치 사용 시 로드: {weights_path}")

    avg_final = np.mean([r["boundary_score"] for r in loop_results])
    print(f"\n[SUMMARY] 최종 평균 boundary_score={avg_final:.4f}, 가중치={BOUNDARY_SCORE_WEIGHTS}")
    return all_loop_history


if __name__ == "__main__":
    main()
