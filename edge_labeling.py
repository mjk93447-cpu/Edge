"""1px edge ground-truth labels and BSDS-style scoring utilities."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

EDGE_LABEL_DIR = "edge_labels"
EDGE_LABEL_VERSION = 1
META_ZONE_TYPES = ("ignore", "gap", "merge-risk", "split-risk", "branch", "loop")


@dataclass
class EdgeLabelRecord:
    image_path: str
    image_hash: str
    roi: Tuple[int, int, int, int]
    roi_shape: Tuple[int, int]
    vector_path: str
    mask_path: str
    skeleton_path: str
    meta_zone_path: str
    updated_at: str
    label_version: int = EDGE_LABEL_VERSION

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EdgeLabelRecord":
        data = dict(payload)
        data["roi"] = tuple(int(v) for v in data["roi"])
        data["roi_shape"] = tuple(int(v) for v in data["roi_shape"])
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def image_hash(path: str) -> str:
    """Return a stable short hash for path contents and basename."""
    h = hashlib.sha1()
    h.update(os.path.basename(path).encode("utf-8", errors="ignore"))
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        h.update(os.path.abspath(path).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def ensure_label_dir(root: Optional[str] = None) -> str:
    base = os.path.abspath(root or EDGE_LABEL_DIR)
    os.makedirs(base, exist_ok=True)
    return base


def _index_path(root: Optional[str] = None) -> str:
    return os.path.join(ensure_label_dir(root), "index.json")


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def load_edge_label_index(root: Optional[str] = None) -> Dict[str, Any]:
    payload = _read_json(_index_path(root), {"records": {}})
    if not isinstance(payload, dict):
        return {"records": {}}
    records = payload.get("records")
    if not isinstance(records, dict):
        payload["records"] = {}
    return payload


def save_edge_label_index(index: Dict[str, Any], root: Optional[str] = None) -> None:
    _write_json(_index_path(root), index)


def load_edge_labels(image_paths: Iterable[str], root: Optional[str] = None) -> List[EdgeLabelRecord]:
    index = load_edge_label_index(root)
    records = index.get("records", {})
    out: List[EdgeLabelRecord] = []
    for path in image_paths:
        rec = records.get(os.path.abspath(path)) or records.get(path)
        if not rec:
            # Hash fallback supports moved files with the same contents.
            h = image_hash(path)
            rec = next((v for v in records.values() if v.get("image_hash") == h), None)
        if rec:
            try:
                record = EdgeLabelRecord.from_dict(rec)
            except (KeyError, TypeError, ValueError):
                continue
            if os.path.exists(record.mask_path):
                out.append(record)
    return out


def _bresenham_line(p0: Tuple[int, int], p1: Tuple[int, int]) -> List[Tuple[int, int]]:
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    points = []
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    radius = int(radius)
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            y0 = radius + dy
            x0 = radius + dx
            out |= padded[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]
    return out


def rasterize_polylines(
    polylines: Sequence[Dict[str, Any]],
    shape: Tuple[int, int],
    width: int = 1,
    snap: Optional[str] = None,
) -> np.ndarray:
    """Rasterize vector lines to a binary mask. Coordinates are x,y in ROI pixels."""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=bool)
    for item in polylines:
        if isinstance(item, dict):
            points = item.get("points", [])
            item_width = int(item.get("width", width))
            closed = bool(item.get("closed"))
        else:
            points = item if isinstance(item, (list, tuple)) else []
            item_width = int(width)
            closed = False
        if len(points) < 2:
            continue
        line_mask = np.zeros((h, w), dtype=bool)
        pts = [(int(round(p[0])), int(round(p[1]))) for p in points]
        if closed and pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        for p0, p1 in zip(pts, pts[1:]):
            for x, y in _bresenham_line(p0, p1):
                if 0 <= x < w and 0 <= y < h:
                    line_mask[y, x] = True
        mask |= dilate_mask(line_mask, max(0, item_width // 2))
    return mask


def zhang_suen_thinning(mask: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Thin a binary mask to a 1px skeleton using Zhang-Suen thinning."""
    img = np.asarray(mask).astype(bool).copy()
    if not np.any(img):
        return img
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            to_remove = []
            ys, xs = np.where(img)
            for y, x in zip(ys, xs):
                if y == 0 or x == 0 or y >= img.shape[0] - 1 or x >= img.shape[1] - 1:
                    continue
                p2 = img[y - 1, x]
                p3 = img[y - 1, x + 1]
                p4 = img[y, x + 1]
                p5 = img[y + 1, x + 1]
                p6 = img[y + 1, x]
                p7 = img[y + 1, x - 1]
                p8 = img[y, x - 1]
                p9 = img[y - 1, x - 1]
                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                n_count = int(sum(neighbors))
                if n_count < 2 or n_count > 6:
                    continue
                transitions = sum((not neighbors[i]) and neighbors[(i + 1) % 8] for i in range(8))
                if transitions != 1:
                    continue
                if step == 0:
                    cond = (not (p2 and p4 and p6)) and (not (p4 and p6 and p8))
                else:
                    cond = (not (p2 and p4 and p8)) and (not (p2 and p6 and p8))
                if cond:
                    to_remove.append((y, x))
            if to_remove:
                yy, xx = zip(*to_remove)
                img[np.asarray(yy), np.asarray(xx)] = False
                changed = True
        if not changed:
            break
    return img


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).astype(np.uint8)
    padded = np.pad(m, 1, mode="constant")
    out = np.zeros_like(m, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out += padded[1 + dy : 1 + dy + m.shape[0], 1 + dx : 1 + dx + m.shape[1]]
    return out


def count_components(mask: np.ndarray) -> int:
    mask = np.asarray(mask).astype(bool)
    if not np.any(mask):
        return 0
    visited = np.zeros(mask.shape, dtype=bool)
    components = 0
    for y, x in np.argwhere(mask):
        if visited[y, x]:
            continue
        components += 1
        stack = [(int(y), int(x))]
        visited[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return components


def remove_small_components(mask: np.ndarray, min_pixels: int = 4) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if min_pixels <= 1 or not np.any(mask):
        return mask.copy()
    visited = np.zeros(mask.shape, dtype=bool)
    out = np.zeros(mask.shape, dtype=bool)
    for y, x in np.argwhere(mask):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        pixels = []
        visited[y, x] = True
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(pixels) >= int(min_pixels):
            yy, xx = zip(*pixels)
            out[np.asarray(yy), np.asarray(xx)] = True
    return out


def prune_spurs(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask).astype(bool).copy()
    for _ in range(max(0, int(iterations))):
        if not np.any(out):
            break
        neighbors = _neighbor_count(out)
        endpoints = out & (neighbors <= 1)
        if not np.any(endpoints):
            break
        out[endpoints] = False
    return out


def bridge_short_gaps(mask: np.ndarray, max_gap: int = 0, protect_topology: bool = True) -> np.ndarray:
    out = np.asarray(mask).astype(bool).copy()
    max_gap = int(max_gap)
    if max_gap <= 0 or not np.any(out):
        return out
    before_components = count_components(out)
    before_branch = int((out & (_neighbor_count(out) >= 3)).sum())
    endpoints = np.argwhere(out & (_neighbor_count(out) <= 1))
    for i, (y0, x0) in enumerate(endpoints):
        for y1, x1 in endpoints[i + 1 :]:
            dy = int(y1) - int(y0)
            dx = int(x1) - int(x0)
            cheb = max(abs(dy), abs(dx))
            if cheb <= 1 or cheb > max_gap:
                continue
            if not (dy == 0 or dx == 0 or abs(dy) == abs(dx)):
                continue
            candidate = out.copy()
            for x, y in _bresenham_line((int(x0), int(y0)), (int(x1), int(y1))):
                if 0 <= y < candidate.shape[0] and 0 <= x < candidate.shape[1]:
                    candidate[y, x] = True
            if protect_topology:
                after_branch = int((candidate & (_neighbor_count(candidate) >= 3)).sum())
                after_components = count_components(candidate)
                if after_branch > before_branch or after_components > before_components:
                    continue
                before_components = after_components
                before_branch = after_branch
            out = candidate
    return out


def topology_aware_postprocess(
    pred_prob: np.ndarray,
    threshold: float,
    min_component_pixels: int = 4,
    spur_prune_iters: int = 1,
    max_bridge_gap: int = 0,
    protect_topology: bool = True,
    final_thinning: bool = True,
) -> np.ndarray:
    pred = np.asarray(pred_prob, dtype=np.float32)
    binary = pred >= float(threshold)
    thin = zhang_suen_thinning(binary)
    filtered = remove_small_components(thin, min_pixels=min_component_pixels)
    pruned = prune_spurs(filtered, iterations=spur_prune_iters)
    bridged = bridge_short_gaps(pruned, max_gap=max_bridge_gap, protect_topology=protect_topology)
    if final_thinning:
        bridged = zhang_suen_thinning(bridged)
    return remove_small_components(bridged, min_pixels=min_component_pixels)


def validate_edge_label_quality(
    gt_mask: np.ndarray,
    min_label_pixels: int = 2,
    max_density: float = 0.25,
    max_branch_ratio: float = 0.20,
    max_thickness_ratio: float = 3.0,
) -> Dict[str, Any]:
    mask = np.asarray(gt_mask).astype(bool)
    skeleton = zhang_suen_thinning(mask)
    pixels = int(mask.sum())
    skeleton_pixels = int(skeleton.sum())
    density = float(pixels / max(1, mask.size))
    branch_count = int((skeleton & (_neighbor_count(skeleton) >= 3)).sum())
    branch_ratio = float(branch_count / max(1, skeleton_pixels))
    thickness_ratio = float(pixels / max(1, skeleton_pixels))
    component_count = count_components(skeleton)
    warnings = []
    if pixels < int(min_label_pixels):
        warnings.append("label_too_short")
    if density > float(max_density):
        warnings.append("label_density_too_high")
    if branch_ratio > float(max_branch_ratio):
        warnings.append("branch_ratio_too_high")
    if thickness_ratio > float(max_thickness_ratio):
        warnings.append("label_not_1px_thin")
    return {
        "valid": not warnings,
        "warnings": warnings,
        "label_pixels": pixels,
        "skeleton_pixels": skeleton_pixels,
        "density": density,
        "component_count": component_count,
        "branch_count": branch_count,
        "branch_ratio": branch_ratio,
        "thickness_ratio": thickness_ratio,
    }


def _threshold_curve(
    pred_prob: np.ndarray,
    gt_mask: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: np.ndarray,
    tolerance_radius: int,
) -> List[Dict[str, float]]:
    gt = gt_mask & valid_mask
    gt_tol = dilate_mask(gt, tolerance_radius) & valid_mask
    out = []
    for th in thresholds:
        pred = (pred_prob >= float(th)) & valid_mask
        pred_tol = dilate_mask(pred, tolerance_radius) & valid_mask
        tp_p = int((pred & gt_tol).sum())
        tp_r = int((gt & pred_tol).sum())
        pred_n = int(pred.sum())
        gt_n = int(gt.sum())
        precision = tp_p / pred_n if pred_n else (1.0 if gt_n == 0 else 0.0)
        recall = tp_r / gt_n if gt_n else (1.0 if pred_n == 0 else 0.0)
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.append({"threshold": float(th), "precision": precision, "recall": recall, "f1": f1})
    return out


def _average_precision(curve: List[Dict[str, float]]) -> float:
    pts = sorted(((p["recall"], p["precision"]) for p in curve), key=lambda x: x[0])
    if len(pts) < 2:
        return pts[0][1] if pts else 0.0
    ap = 0.0
    for (r0, p0), (r1, p1) in zip(pts, pts[1:]):
        ap += max(0.0, r1 - r0) * max(p0, p1)
    return float(max(0.0, min(1.0, ap)))


def meta_zones_to_mask(
    meta_zones: Optional[Dict[str, Any]],
    shape: Tuple[int, int],
    zone_types: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Rasterize simple meta-zone payloads to a binary ROI mask."""
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), dtype=bool)
    if not meta_zones:
        return out
    allowed = set(zone_types or META_ZONE_TYPES)
    for zone_type, entries in meta_zones.items():
        if zone_type not in allowed:
            continue
        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if "rect" in entry:
                x1, y1, x2, y2 = [int(round(v)) for v in entry["rect"]]
                x1, x2 = sorted((max(0, x1), min(w, x2)))
                y1, y2 = sorted((max(0, y1), min(h, y2)))
                out[y1:y2, x1:x2] = True
            elif "points" in entry:
                width = int(entry.get("width", 1))
                out |= rasterize_polylines([entry], shape, width=width)
    return out


def compute_edge_label_score(
    pred_prob: np.ndarray,
    gt_mask: np.ndarray,
    ignore_mask: Optional[np.ndarray] = None,
    meta_zones: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Sequence[float]] = None,
    tolerance_radius: Optional[int] = None,
    topology_safe: bool = True,
    topology_penalty: float = 0.25,
    topology_failure_penalty: float = 2.0,
    topology_risk_tolerance: float = 0.10,
    density_penalty: float = 0.20,
    min_precision: float = 0.05,
    max_density_ratio: float = 8.0,
    postprocess_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute BSDS-style threshold sweep plus 1px topology metrics."""
    pred = np.asarray(pred_prob, dtype=np.float32)
    gt = np.asarray(gt_mask).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match GT shape {gt.shape}.")
    if pred.max(initial=0.0) > 1.0 or pred.min(initial=0.0) < 0.0:
        pmin = float(pred.min(initial=0.0))
        pmax = float(pred.max(initial=1.0))
        pred = (pred - pmin) / max(pmax - pmin, 1e-6)
    valid = np.ones(gt.shape, dtype=bool)
    meta_ignore = meta_zones_to_mask(meta_zones, gt.shape, zone_types=("ignore",))
    if ignore_mask is not None:
        valid &= ~np.asarray(ignore_mask).astype(bool)
    if np.any(meta_ignore):
        valid &= ~meta_ignore
    gt &= valid
    if thresholds is None:
        thresholds_arr = np.linspace(0.05, 0.95, 19)
    else:
        thresholds_arr = np.asarray(list(thresholds), dtype=np.float32)
    if tolerance_radius is None:
        tolerance_radius = max(2, int(round(0.003 * max(gt.shape))))
    curve = _threshold_curve(pred, gt, valid, thresholds_arr, int(tolerance_radius))
    best = max(curve, key=lambda row: (row["f1"], row["precision"], row["recall"])) if curve else {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    pred_bin = (pred >= best["threshold"]) & valid
    exact_curve = _threshold_curve(pred, gt, valid, [best["threshold"]], 0)
    exact = exact_curve[0]
    pred_skel = zhang_suen_thinning(pred_bin)
    gt_skel = zhang_suen_thinning(gt)
    smooth = 1e-6
    tprec = float((pred_skel & dilate_mask(gt_skel, int(tolerance_radius))).sum()) / max(float(pred_skel.sum()), smooth)
    tsens = float((gt_skel & dilate_mask(pred_skel, int(tolerance_radius))).sum()) / max(float(gt_skel.sum()), smooth)
    cldice = 2.0 * tprec * tsens / max(tprec + tsens, smooth)
    pred_band = dilate_mask(pred_bin, int(tolerance_radius))
    gt_band = dilate_mask(gt, int(tolerance_radius))
    union = int((pred_band | gt_band).sum())
    boundary_iou = int((pred_band & gt_band).sum()) / union if union else 1.0
    pred_comp = count_components(pred_skel)
    gt_comp = count_components(gt_skel)
    split_risk = max(0, pred_comp - gt_comp) / max(1, gt_comp)
    merge_risk = max(0, gt_comp - pred_comp) / max(1, gt_comp)
    neighbor = _neighbor_count(pred_skel)
    endpoint_count = int((pred_skel & (neighbor <= 1)).sum())
    branch_count = int((pred_skel & (neighbor >= 3)).sum())
    branch_risk = float(branch_count / max(1, int(gt_skel.sum())))
    loop_count = max(0, int(pred_skel.sum()) - endpoint_count - branch_count - max(0, pred_comp - 1))
    meta_zone_pixels = {}
    if meta_zones:
        for zone_type in META_ZONE_TYPES:
            zone_mask = meta_zones_to_mask(meta_zones, gt.shape, zone_types=(zone_type,))
            meta_zone_pixels[zone_type] = int(zone_mask.sum())
    result = {
        "best_bf1": float(best["f1"]),
        "best_threshold": float(best["threshold"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "ap": _average_precision(curve),
        "exact_f1": float(exact["f1"]),
        "exact_precision": float(exact["precision"]),
        "exact_recall": float(exact["recall"]),
        "cldice": float(max(0.0, min(1.0, cldice))),
        "boundary_iou": float(boundary_iou),
        "merge_risk": float(merge_risk),
        "split_risk": float(split_risk),
        "branch_risk": float(branch_risk),
        "merge_split_risk": float(max(merge_risk, split_risk, branch_risk)),
        "endpoint_count": endpoint_count,
        "branch_count": branch_count,
        "loop_count": loop_count,
        "tolerance_radius": int(tolerance_radius),
        "curve": curve,
        "topology_risk_high": bool(max(merge_risk, split_risk, branch_risk) >= 0.10),
        "meta_zone_pixels": meta_zone_pixels,
    }
    if topology_safe:
        pp_cfg = dict(postprocess_config or {})
        safe_candidates = []
        gt_density = float(gt.sum()) / max(1, int(valid.sum()))
        for row in curve:
            threshold = float(row["threshold"])
            safe_pred = topology_aware_postprocess(
                pred,
                threshold=threshold,
                min_component_pixels=int(pp_cfg.get("min_component_pixels", 4)),
                spur_prune_iters=int(pp_cfg.get("spur_prune_iters", 1)),
                max_bridge_gap=int(pp_cfg.get("max_bridge_gap", 0)),
                protect_topology=bool(pp_cfg.get("protect_topology", True)),
                final_thinning=bool(pp_cfg.get("final_thinning", True)),
            ) & valid
            safe_curve = _threshold_curve(safe_pred.astype(np.float32), gt, valid, [0.5], int(tolerance_radius))
            safe_exact = _threshold_curve(safe_pred.astype(np.float32), gt, valid, [0.5], 0)[0]
            safe_skel = zhang_suen_thinning(safe_pred)
            safe_comp = count_components(safe_skel)
            safe_pred_n = int(safe_pred.sum())
            safe_density = float(safe_pred_n) / max(1, int(valid.sum()))
            density_ratio = safe_density / max(gt_density, 1e-6)
            safe_gt_skel = zhang_suen_thinning(gt)
            safe_gt_comp = count_components(safe_gt_skel)
            safe_split = max(0, safe_comp - safe_gt_comp) / max(1, safe_gt_comp)
            safe_merge = max(0, safe_gt_comp - safe_comp) / max(1, safe_gt_comp)
            safe_branch_count = int((safe_skel & (_neighbor_count(safe_skel) >= 3)).sum())
            safe_branch_risk = float(safe_branch_count / max(1, int(safe_gt_skel.sum())))
            safe_risk = float(max(safe_merge, safe_split, safe_branch_risk))
            safe_precision = float(safe_curve[0]["precision"])
            density_over = max(0.0, density_ratio - 1.0)
            topology_excess = max(0.0, safe_risk - float(topology_risk_tolerance))
            precision_fail = safe_precision < float(min_precision)
            density_fail = density_ratio > float(max_density_ratio)
            topology_fail = safe_risk >= float(topology_risk_tolerance)
            objective = (
                float(safe_curve[0]["f1"])
                - float(topology_penalty) * safe_risk
                - float(topology_failure_penalty) * topology_excess * topology_excess
                - float(density_penalty) * density_over
                - (0.25 if precision_fail else 0.0)
                - (0.25 if density_fail else 0.0)
                - (0.50 if topology_fail else 0.0)
            )
            safe_candidates.append(
                {
                    "threshold": threshold,
                    "objective": float(objective),
                    "bf1": float(safe_curve[0]["f1"]),
                    "precision": safe_precision,
                    "recall": float(safe_curve[0]["recall"]),
                    "exact_f1": float(safe_exact["f1"]),
                    "merge_split_risk": safe_risk,
                    "branch_risk": safe_branch_risk,
                    "density_ratio": float(density_ratio),
                    "pred_pixels": safe_pred_n,
                    "component_count": int(safe_comp),
                    "passes_precision_floor": not precision_fail,
                    "passes_density_ratio": not density_fail,
                    "passes_topology_gate": not topology_fail,
                }
            )
        safe_best = max(
            safe_candidates,
            key=lambda row: (
                row["passes_topology_gate"],
                row["objective"],
                row["passes_precision_floor"],
                row["passes_density_ratio"],
                row["bf1"],
                -row["merge_split_risk"],
            ),
        ) if safe_candidates else None
        if safe_best:
            result["topology_safe"] = {
                "best_threshold": float(safe_best["threshold"]),
                "objective": float(safe_best["objective"]),
                "bf1": float(safe_best["bf1"]),
                "precision": float(safe_best["precision"]),
                "recall": float(safe_best["recall"]),
                "exact_f1": float(safe_best["exact_f1"]),
                "merge_split_risk": float(safe_best["merge_split_risk"]),
                "branch_risk": float(safe_best["branch_risk"]),
                "density_ratio": float(safe_best["density_ratio"]),
                "pred_pixels": int(safe_best["pred_pixels"]),
                "component_count": int(safe_best["component_count"]),
                "candidates": safe_candidates,
                "postprocess_config": {
                    "min_component_pixels": int(pp_cfg.get("min_component_pixels", 4)),
                    "spur_prune_iters": int(pp_cfg.get("spur_prune_iters", 1)),
                    "max_bridge_gap": int(pp_cfg.get("max_bridge_gap", 0)),
                    "protect_topology": bool(pp_cfg.get("protect_topology", True)),
                    "final_thinning": bool(pp_cfg.get("final_thinning", True)),
                    "topology_penalty": float(topology_penalty),
                    "topology_failure_penalty": float(topology_failure_penalty),
                    "topology_risk_tolerance": float(topology_risk_tolerance),
                    "density_penalty": float(density_penalty),
                    "min_precision": float(min_precision),
                    "max_density_ratio": float(max_density_ratio),
                },
            }
    return result


def save_edge_label(
    image_path: str,
    roi: Tuple[int, int, int, int],
    polylines: Sequence[Dict[str, Any]],
    brush_mask: Optional[np.ndarray] = None,
    meta_zones: Optional[Dict[str, Any]] = None,
    root: Optional[str] = None,
) -> EdgeLabelRecord:
    label_dir = ensure_label_dir(root)
    x1, y1, x2, y2 = [int(v) for v in roi]
    shape = (max(1, y2 - y1), max(1, x2 - x1))
    h = image_hash(image_path)
    vector_path = os.path.join(label_dir, f"{h}_vector_line.json")
    mask_path = os.path.join(label_dir, f"{h}_mask.png")
    skeleton_path = os.path.join(label_dir, f"{h}_skeleton.png")
    meta_zone_path = os.path.join(label_dir, f"{h}_meta_zone.json")
    vector_mask = rasterize_polylines(polylines, shape, width=1)
    merged = vector_mask.copy()
    if brush_mask is not None:
        bm = np.asarray(brush_mask).astype(bool)
        if bm.shape == merged.shape:
            merged |= bm
    skeleton = zhang_suen_thinning(merged)
    _write_json(vector_path, {"image_path": image_path, "roi": list(roi), "polylines": list(polylines)})
    clean_meta = {k: v for k, v in (meta_zones or {}).items() if k in META_ZONE_TYPES}
    _write_json(meta_zone_path, clean_meta)
    Image.fromarray((merged.astype(np.uint8) * 255)).save(mask_path)
    Image.fromarray((skeleton.astype(np.uint8) * 255)).save(skeleton_path)
    rec = EdgeLabelRecord(
        image_path=image_path,
        image_hash=h,
        roi=(x1, y1, x2, y2),
        roi_shape=shape,
        vector_path=os.path.abspath(vector_path),
        mask_path=os.path.abspath(mask_path),
        skeleton_path=os.path.abspath(skeleton_path),
        meta_zone_path=os.path.abspath(meta_zone_path),
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    index = load_edge_label_index(root)
    index.setdefault("records", {})[os.path.abspath(image_path)] = rec.to_dict()
    save_edge_label_index(index, root)
    return rec


def load_mask(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def load_record_masks(record: EdgeLabelRecord) -> Tuple[np.ndarray, np.ndarray]:
    return load_mask(record.mask_path), load_mask(record.skeleton_path)
