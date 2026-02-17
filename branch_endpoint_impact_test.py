"""
Branch & Endpoint impact test for accurate boundary detection.
- Builds (good boundary) vs (bad boundary) image pairs: good=closed thin contour, bad=open or branched.
- Evaluates with multiple param sets; measures whether branch/endpoint metrics help rank good > bad.
- If without these terms the ranking is unchanged or only slightly worse, recommend reduce/disable.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sobel_edge_detection import (
    SobelEdgeDetector,
    compute_auto_score,
    evaluate_one_candidate_mp,
    PARAM_DEFAULTS,
    AUTO_DEFAULTS,
)


def make_band_and_boundary(h, w, band_radius, margin=2):
    boundary = np.zeros((h, w), dtype=bool)
    boundary[margin, margin:w - margin] = True
    boundary[h - 1 - margin, margin:w - margin] = True
    boundary[margin:h - margin, margin] = True
    boundary[margin:h - margin, w - 1 - margin] = True
    det = SobelEdgeDetector()
    band = det.dilate_binary(boundary, band_radius) if band_radius > 0 else boundary.copy()
    return boundary, band


def build_data_item(image, boundary, band, band_radius):
    mask = np.ones_like(boundary, dtype=bool)
    band_pixels = int(band.sum())
    return {
        "image": np.asarray(image, dtype=np.float32),
        "mask": mask,
        "boundary": boundary,
        "bands": {band_radius: band, 0: boundary},
        "band_pixels": {band_radius: band_pixels, 0: int(boundary.sum())},
        "weight": 1.0,
    }


def get_param_variants():
    """Several param sets to see if branch/endpoint help across different detections."""
    base = dict(PARAM_DEFAULTS)
    base["boundary_band_radius"] = 3
    base["use_boundary_band_filter"] = False
    variants = []
    for nms in [0.92, 0.95, 0.98]:
        for high in [0.08, 0.10, 0.12]:
            s = dict(base)
            s["nms_relax"] = nms
            s["high_ratio"] = high
            s["low_ratio"] = max(0.02, high * 0.35)
            variants.append(s)
    return variants[:9]  # 9 variants


def run_branch_endpoint_impact():
    h, w = 64, 64
    band_radius = 3
    det = SobelEdgeDetector()

    # Good: closed rectangle (ideal boundary)
    boundary, band = make_band_and_boundary(h, w, band_radius)
    img_good = np.zeros((h, w), dtype=np.float32)
    img_good[8:56, 8:56] = 200.0
    data_good = [build_data_item(img_good, boundary, band, band_radius)]

    # Bad endpoint: open boundary (two parallel lines -> many endpoints)
    img_bad_end = np.zeros((h, w), dtype=np.float32)
    img_bad_end[10:54, 12:52] = 200.0
    b_open = np.zeros((h, w), dtype=bool)
    b_open[10, 12:52] = True
    b_open[53, 12:52] = True
    band_open = det.dilate_binary(b_open, band_radius)
    data_bad_end = [build_data_item(img_bad_end, b_open, band_open, band_radius)]

    # Bad branch: cross (T-junctions)
    img_bad_br = np.zeros((h, w), dtype=np.float32)
    img_bad_br[8:56, 8:56] = 200.0
    img_bad_br[28:36, :] = 200.0
    img_bad_br[:, 28:36] = 200.0
    data_bad_br = [build_data_item(img_bad_br, boundary, band, band_radius)]

    config_full = dict(AUTO_DEFAULTS)
    config_no_end_br = dict(AUTO_DEFAULTS)
    config_no_end_br["weight_endpoints"] = 0.0
    config_no_end_br["weight_branch"] = 0.0
    # exp_penalty in compute_auto_score uses endpoint_ratio + branch_ratio; we can't pass a flag,
    # so we measure score with full config and compare ranking. To simulate "no exp penalty" we
    # would need to change the code. Instead we measure: (1) do endpoint/branch values actually
    # differ between good and bad? (2) With full weights, how often does good beat bad?

    variants = get_param_variants()
    results_full = []
    results_no_end_br = []

    for settings in variants:
        # Full weights
        sg, sum_g, _ = evaluate_one_candidate_mp(data_good, settings, config_full)
        se, sum_e, _ = evaluate_one_candidate_mp(data_bad_end, settings, config_full)
        sb, sum_b, _ = evaluate_one_candidate_mp(data_bad_br, settings, config_full)
        results_full.append({
            "good_score": sg,
            "bad_endpoint_score": se,
            "bad_branch_score": sb,
            "good_end": sum_g["endpoints"],
            "good_br": sum_g["branch"],
            "bad_end_end": sum_e["endpoints"],
            "bad_end_br": sum_e["branch"],
            "bad_br_end": sum_b["endpoints"],
            "bad_br_br": sum_b["branch"],
        })

        # Zero endpoint/branch weight (exp_penalty still applied in code)
        sg2, sum_g2, _ = evaluate_one_candidate_mp(data_good, settings, config_no_end_br)
        se2, _, _ = evaluate_one_candidate_mp(data_bad_end, settings, config_no_end_br)
        sb2, _, _ = evaluate_one_candidate_mp(data_bad_br, settings, config_no_end_br)
        results_no_end_br.append({
            "good_score": sg2,
            "bad_endpoint_score": se2,
            "bad_branch_score": sb2,
        })

    # Analysis
    n = len(variants)
    full_good_beats_bad_end = sum(1 for r in results_full if r["good_score"] > r["bad_endpoint_score"])
    full_good_beats_bad_br = sum(1 for r in results_full if r["good_score"] > r["bad_branch_score"])
    no_w_good_beats_bad_end = sum(1 for r in results_no_end_br if r["good_score"] > r["bad_endpoint_score"])
    no_w_good_beats_bad_br = sum(1 for r in results_no_end_br if r["good_score"] > r["bad_branch_score"])

    avg_good_end = np.mean([r["good_end"] for r in results_full])
    avg_good_br = np.mean([r["good_br"] for r in results_full])
    avg_bad_end_end = np.mean([r["bad_end_end"] for r in results_full])
    avg_bad_br_br = np.mean([r["bad_br_br"] for r in results_full])
    diff_end = avg_bad_end_end - avg_good_end
    diff_br = avg_bad_br_br - avg_good_br

    report = {
        "n_param_variants": n,
        "with_full_weights": {
            "good_beat_bad_endpoint_count": full_good_beats_bad_end,
            "good_beat_bad_branch_count": full_good_beats_bad_br,
        },
        "with_zero_end_branch_weight": {
            "good_beat_bad_endpoint_count": no_w_good_beats_bad_end,
            "good_beat_bad_branch_count": no_w_good_beats_bad_br,
        },
        "metric_means": {
            "good_avg_endpoints": avg_good_end,
            "good_avg_branch": avg_good_br,
            "bad_endpoint_avg_endpoints": avg_bad_end_end,
            "bad_branch_avg_branch": avg_bad_br_br,
            "diff_endpoints_good_vs_bad_end": diff_end,
            "diff_branch_good_vs_bad_br": diff_br,
        },
        "recommendation": None,
    }

    # Recommend: if (1) metric difference is tiny OR (2) without weights ranking is same or 1 less
    if diff_end < 0.02 and diff_br < 0.05:
        report["recommendation"] = "disable"
        report["reason"] = "Branch/endpoint metrics show very small difference between good and bad boundaries (discriminative power weak)."
    elif no_w_good_beats_bad_end >= full_good_beats_bad_end - 1 and no_w_good_beats_bad_br >= full_good_beats_bad_br - 1:
        report["recommendation"] = "reduce_or_disable"
        report["reason"] = "Ranking (good > bad) is largely preserved without endpoint/branch weights; impact on accurate boundary detection is limited."
    else:
        report["recommendation"] = "keep"
        report["reason"] = "Endpoint/branch help rank good vs bad; keep current weights or reduce slightly."

    return report, results_full, results_no_end_br


def main():
    out_dir = "score_penalty_analysis_out"
    os.makedirs(out_dir, exist_ok=True)

    print("=== Branch & Endpoint impact on accurate boundary detection ===\n")
    report, results_full, results_no = run_branch_endpoint_impact()

    n = report["n_param_variants"]
    print(f"Param variants: {n}\n")
    print("With full weights (good vs bad_endpoint): good wins", report["with_full_weights"]["good_beat_bad_endpoint_count"], f"/ {n}")
    print("With full weights (good vs bad_branch):   good wins", report["with_full_weights"]["good_beat_bad_branch_count"], f"/ {n}")
    print("With weight_endpoints=0, weight_branch=0:")
    print("  good vs bad_endpoint: good wins", report["with_zero_end_branch_weight"]["good_beat_bad_endpoint_count"], f"/ {n}")
    print("  good vs bad_branch:   good wins", report["with_zero_end_branch_weight"]["good_beat_bad_branch_count"], f"/ {n}")
    m = report["metric_means"]
    print("\nMetric means (good vs bad):")
    print(f"  good avg endpoints: {m['good_avg_endpoints']:.4f}  (bad_endpoint avg: {m['bad_endpoint_avg_endpoints']:.4f})  diff={m['diff_endpoints_good_vs_bad_end']:.4f}")
    print(f"  good avg branch:    {m['good_avg_branch']:.4f}  (bad_branch avg:   {m['bad_branch_avg_branch']:.4f})  diff={m['diff_branch_good_vs_bad_br']:.4f}")
    print(f"\nRecommendation: {report['recommendation'].upper()}")
    print(f"Reason: {report['reason']}")

    path = os.path.join(out_dir, "branch_endpoint_impact_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {path}")
    return report


if __name__ == "__main__":
    main()
