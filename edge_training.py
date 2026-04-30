"""Lightweight 1px edge training and HPO entry points.

The v1 production path is a compact TEED-style side-output model. Heavy
research backbones are represented as presets/adapters so they can be added
without blocking the local supervised workflow.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from edge_labeling import EdgeLabelRecord, compute_edge_label_score, load_record_masks, validate_edge_label_quality

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on machines without torch.
    torch = None
    nn = None
    F = None
    DataLoader = object
    Dataset = object
    _TORCH_AVAILABLE = False


MODEL_PRESETS = {
    "teed_1px_default": {
        "family": "compact_teed",
        "channels": 24,
        "deep_supervision": True,
        "description": "Small TEED-style v1 production model.",
    },
    "dexined_1px_accuracy": {
        "family": "dexined_adapter",
        "description": "DexiNed adapter placeholder; use compact core until adapter dependency is installed.",
    },
    "edgenat_1px_adapter": {
        "family": "edgenat_adapter",
        "description": "EdgeNAT adapter placeholder for accuracy ceiling.",
    },
    "pidinet_1px_fast": {
        "family": "pidinet_adapter",
        "description": "PiDiNet/PIDINet-MC ultra-fast adapter placeholder.",
    },
}

_OPTUNA_AVAILABLE = importlib.util.find_spec("optuna") is not None


@dataclass
class TrainConfig:
    output_dir: str = "outputs/deep_edge_training"
    model_preset: str = "teed_1px_default"
    device: str = "auto"
    seed: int = 42
    epochs: int = 3
    batch_size: int = 2
    lr_max: float = 1e-3
    weight_decay: float = 1e-4
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 0.0
    focal_gamma: float = 2.0
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    cldice_weight: float = 0.05
    cldice_ramp_epochs: int = 3
    cbdice_weight: float = 0.0
    negative_centerline_weight: float = 0.0
    background_distance_weight: float = 0.0
    simplified_topology_weight: float = 0.0
    topology_failure_weight: float = 2.0
    topology_risk_tolerance: float = 0.10
    false_bridge_weight: float = 1.0
    split_gap_weight: float = 1.0
    topology_loss_warmup_epochs: int = 1
    crop_size: int = 256
    min_label_pixels: int = 2
    max_label_density: float = 0.25
    max_label_branch_ratio: float = 0.20
    max_label_thickness_ratio: float = 3.0
    tolerance_radius: Optional[int] = None
    topology_penalty: float = 0.25
    topology_failure_penalty: float = 2.0
    density_penalty: float = 0.20
    min_precision: float = 0.05
    max_density_ratio: float = 8.0
    min_component_pixels: int = 4
    spur_prune_iters: int = 1
    max_bridge_gap: int = 0
    protect_topology_postprocess: bool = True
    final_thinning_postprocess: bool = True
    final_train: bool = False
    amp_enabled: bool = True


@dataclass
class TrainingResult:
    best_score: float
    best_threshold: float
    model_path: str
    threshold_path: str
    report_path: str
    metrics: Dict[str, Any]
    config: Dict[str, Any]


@dataclass
class HPOConfig:
    output_dir: str = "outputs/deep_edge_hpo"
    model_presets: Sequence[str] = field(default_factory=lambda: ("teed_1px_default",))
    trial_count: int = 3
    max_epochs_per_trial: int = 2
    final_epochs: int = 5
    seed: int = 42
    device: str = "auto"
    batch_size: int = 2
    crop_size: int = 256


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for deep edge training. Install torch or use Classical fallback.")


def _resolve_device(device: str) -> "torch.device":
    _require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _effective_preset(preset: str) -> str:
    spec = MODEL_PRESETS.get(preset, MODEL_PRESETS["teed_1px_default"])
    if spec["family"] != "compact_teed":
        return "teed_1px_default"
    return preset if preset in MODEL_PRESETS else "teed_1px_default"


def _model_state_hash(model: "nn.Module") -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class EdgeLabelDataset(Dataset):
    def __init__(self, records: Sequence[EdgeLabelRecord], crop_size: int = 256):
        _require_torch()
        self.records = list(records)
        self.crop_size = int(crop_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert("L")
        x1, y1, x2, y2 = rec.roi
        roi_img = image.crop((x1, y1, x2, y2))
        gt_mask, gt_skel = load_record_masks(rec)
        target_size = (self.crop_size, self.crop_size)
        roi_img = roi_img.resize(target_size, resample=Image.BILINEAR)
        mask_img = Image.fromarray((gt_skel.astype(np.uint8) * 255)).resize(target_size, resample=Image.NEAREST)
        img_arr = np.asarray(roi_img, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask_img, dtype=np.uint8) > 0).astype(np.float32)
        image_tensor = torch.from_numpy(img_arr[None, :, :])
        target_tensor = torch.from_numpy(mask_arr[None, :, :])
        return {"image": image_tensor, "target": target_tensor}


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TEED1pxModel(nn.Module):
    """Compact TEED-style model with side outputs for 1px edge maps."""

    def __init__(self, channels: int = 24):
        super().__init__()
        c = int(channels)
        self.stem = _ConvBlock(1, c)
        self.down1 = _ConvBlock(c, c * 2, dilation=1)
        self.down2 = _ConvBlock(c * 2, c * 3, dilation=2)
        self.down3 = _ConvBlock(c * 3, c * 4, dilation=3)
        self.side1 = nn.Conv2d(c, 1, 1)
        self.side2 = nn.Conv2d(c * 2, 1, 1)
        self.side3 = nn.Conv2d(c * 3, 1, 1)
        self.side4 = nn.Conv2d(c * 4, 1, 1)
        self.fuse = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        f1 = self.stem(x)
        f2 = self.down1(F.max_pool2d(f1, 2))
        f3 = self.down2(F.max_pool2d(f2, 2))
        f4 = self.down3(F.max_pool2d(f3, 2))
        sides = [
            self.side1(f1),
            F.interpolate(self.side2(f2), size=(h, w), mode="bilinear", align_corners=False),
            F.interpolate(self.side3(f3), size=(h, w), mode="bilinear", align_corners=False),
            F.interpolate(self.side4(f4), size=(h, w), mode="bilinear", align_corners=False),
        ]
        fused = self.fuse(torch.cat(sides, dim=1))
        return {"logits": fused, "sides": sides}


def create_model(preset: str = "teed_1px_default") -> "nn.Module":
    _require_torch()
    effective = _effective_preset(preset)
    spec = MODEL_PRESETS[effective]
    return TEED1pxModel(channels=int(spec.get("channels", 24)))


def _dice_loss(prob, target, eps: float = 1e-6):
    inter = (prob * target).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def _tversky_loss(prob, target, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
    tp = (prob * target).sum(dim=(2, 3))
    fp = (prob * (1.0 - target)).sum(dim=(2, 3))
    fn = ((1.0 - prob) * target).sum(dim=(2, 3))
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - score).mean()


def _focal_loss(logits, target, gamma: float = 2.0, eps: float = 1e-6):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    return (((1.0 - pt).clamp(min=eps) ** gamma) * bce).mean()


def _soft_skeleton(x, iterations: int = 8):
    p = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    contour = F.relu(x - F.max_pool2d(p, kernel_size=3, stride=1, padding=1))
    skel = contour
    for _ in range(iterations):
        x = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        p = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        contour = F.relu(x - F.max_pool2d(p, kernel_size=3, stride=1, padding=1))
        skel = skel + F.relu(contour * (1.0 - skel))
    return torch.clamp(skel, 0.0, 1.0)


def _soft_cldice_loss(prob, target, eps: float = 1e-6):
    skel_pred = _soft_skeleton(prob)
    skel_true = _soft_skeleton(target)
    tprec = (skel_pred * target).sum(dim=(2, 3)) / (skel_pred.sum(dim=(2, 3)) + eps)
    tsens = (skel_true * prob).sum(dim=(2, 3)) / (skel_true.sum(dim=(2, 3)) + eps)
    cl = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return (1.0 - cl).mean()


def _background_distance_loss(prob, target, radius: int = 2, eps: float = 1e-6):
    band = F.max_pool2d(target, kernel_size=2 * radius + 1, stride=1, padding=radius)
    outside = (1.0 - band).clamp(0.0, 1.0)
    denom = outside.sum(dim=(2, 3)).clamp(min=eps)
    return ((prob * prob * outside).sum(dim=(2, 3)) / denom).mean()


def _topology_failure_loss(prob, target, cfg: TrainConfig, eps: float = 1e-6):
    """Large penalty for likely merge/split failures without rewarding topology itself.

    False bridges are high probabilities outside the GT support band. Splits are
    low probabilities on the 1px centerline. Both terms are gated by a small
    tolerance so easy frames still optimize primarily for BF1/BCE/Dice.
    """
    band = F.max_pool2d(target, kernel_size=5, stride=1, padding=2)
    outside = (1.0 - band).clamp(0.0, 1.0)
    false_bridge = ((prob * prob * outside).sum(dim=(2, 3)) / outside.sum(dim=(2, 3)).clamp(min=eps))
    centerline_miss = (((1.0 - prob) * target).sum(dim=(2, 3)) / target.sum(dim=(2, 3)).clamp(min=eps))
    violation = (
        float(cfg.false_bridge_weight) * false_bridge
        + float(cfg.split_gap_weight) * centerline_miss
        - float(cfg.topology_risk_tolerance)
    )
    return F.relu(violation).pow(2).mean()


def _edge_loss(outputs: Dict[str, Any], target, cfg: TrainConfig, pos_weight, epoch: int = 0):
    logits = outputs["logits"]
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    prob = torch.sigmoid(logits)
    loss = cfg.bce_weight * bce + cfg.dice_weight * _dice_loss(prob, target)
    if cfg.focal_weight > 0:
        loss = loss + cfg.focal_weight * _focal_loss(logits, target, gamma=cfg.focal_gamma)
    if cfg.tversky_weight > 0:
        loss = loss + cfg.tversky_weight * _tversky_loss(prob, target, alpha=cfg.tversky_alpha, beta=cfg.tversky_beta)
    if cfg.cldice_weight > 0:
        ramp = 1.0
        if cfg.cldice_ramp_epochs > 0:
            ramp = min(1.0, float(epoch + 1) / float(cfg.cldice_ramp_epochs))
        loss = loss + cfg.cldice_weight * ramp * _soft_cldice_loss(prob, target)
    if cfg.background_distance_weight > 0 or cfg.negative_centerline_weight > 0:
        bg_w = cfg.background_distance_weight + cfg.negative_centerline_weight
        loss = loss + bg_w * _background_distance_loss(prob, target)
    if cfg.topology_failure_weight > 0:
        topo_ramp = 1.0
        if cfg.topology_loss_warmup_epochs > 0:
            topo_ramp = min(1.0, float(epoch + 1) / float(cfg.topology_loss_warmup_epochs))
        loss = loss + cfg.topology_failure_weight * topo_ramp * _topology_failure_loss(prob, target, cfg)
    # Feature-flagged topology losses are represented by the same differentiable
    # centerline support in v1; separate adapters can replace these terms later.
    topo_w = cfg.cbdice_weight + cfg.simplified_topology_weight
    if topo_w > 0:
        loss = loss + topo_w * _soft_cldice_loss(prob, target)
    side_loss = 0.0
    for side in outputs.get("sides", []):
        side_loss = side_loss + F.binary_cross_entropy_with_logits(side, target, pos_weight=pos_weight)
    if outputs.get("sides"):
        loss = loss + 0.2 * side_loss / len(outputs["sides"])
    return loss


def _split_records(records: Sequence[EdgeLabelRecord]) -> tuple[List[EdgeLabelRecord], List[EdgeLabelRecord]]:
    recs = list(records)
    if len(recs) <= 1:
        return recs, recs
    n_val = max(1, int(round(len(recs) * 0.25)))
    return recs[:-n_val], recs[-n_val:]


def _evaluate_model(model, records: Sequence[EdgeLabelRecord], cfg: TrainConfig, device) -> Dict[str, Any]:
    model.eval()
    scores = []
    with torch.no_grad():
        for rec in records:
            image = Image.open(rec.image_path).convert("L")
            x1, y1, x2, y2 = rec.roi
            roi_img = image.crop((x1, y1, x2, y2))
            original_shape = (max(1, y2 - y1), max(1, x2 - x1))
            resized = roi_img.resize((cfg.crop_size, cfg.crop_size), resample=Image.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr[None, None, :, :]).to(device)
            logits = model(tensor)["logits"]
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            prob_img = Image.fromarray(np.clip(prob * 255, 0, 255).astype(np.uint8)).resize(
                (original_shape[1], original_shape[0]),
                resample=Image.BILINEAR,
            )
            pred_prob = np.asarray(prob_img, dtype=np.float32) / 255.0
            mask, gt_skel = load_record_masks(rec)
            scores.append(
                compute_edge_label_score(
                    pred_prob,
                    mask,
                    tolerance_radius=cfg.tolerance_radius,
                    topology_safe=True,
                    topology_penalty=cfg.topology_penalty,
                    topology_failure_penalty=cfg.topology_failure_penalty,
                    topology_risk_tolerance=cfg.topology_risk_tolerance,
                    density_penalty=cfg.density_penalty,
                    min_precision=cfg.min_precision,
                    max_density_ratio=cfg.max_density_ratio,
                    postprocess_config={
                        "min_component_pixels": cfg.min_component_pixels,
                        "spur_prune_iters": cfg.spur_prune_iters,
                        "max_bridge_gap": cfg.max_bridge_gap,
                        "protect_topology": cfg.protect_topology_postprocess,
                        "final_thinning": cfg.final_thinning_postprocess,
                    },
                )
            )
    if not scores:
        return {"best_bf1": 0.0, "best_threshold": 0.5, "scores": []}
    avg = {}
    for key in ("best_bf1", "ap", "exact_f1", "cldice", "boundary_iou", "merge_split_risk"):
        avg[key] = float(np.mean([s.get(key, 0.0) for s in scores]))
    avg["best_threshold"] = float(np.median([s.get("best_threshold", 0.5) for s in scores]))
    safe_scores = [s.get("topology_safe", {}) for s in scores if s.get("topology_safe")]
    if safe_scores:
        avg["topology_safe_bf1"] = float(np.mean([s.get("bf1", 0.0) for s in safe_scores]))
        avg["topology_safe_threshold"] = float(np.median([s.get("best_threshold", 0.5) for s in safe_scores]))
        avg["topology_safe_objective"] = float(np.mean([s.get("objective", 0.0) for s in safe_scores]))
        avg["topology_safe_exact_f1"] = float(np.mean([s.get("exact_f1", 0.0) for s in safe_scores]))
        avg["topology_safe_merge_split_risk"] = float(np.mean([s.get("merge_split_risk", 1.0) for s in safe_scores]))
        avg["topology_safe_density_ratio"] = float(np.mean([s.get("density_ratio", 0.0) for s in safe_scores]))
        avg["topology_safe_objective_min"] = float(np.min([s.get("objective", 0.0) for s in safe_scores]))
        avg["topology_safe_merge_split_risk_max"] = float(np.max([s.get("merge_split_risk", 1.0) for s in safe_scores]))
    avg["topology_risk_high"] = bool(avg["merge_split_risk"] >= 0.10)
    avg["scores"] = scores
    return avg


def _save_pr_curve(metrics: Dict[str, Any], output_dir: str) -> str:
    path = os.path.join(output_dir, "pr_curve.csv")
    curve = []
    for score in metrics.get("scores", []):
        curve.extend(score.get("curve", []))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "precision", "recall", "f1"])
        writer.writeheader()
        for row in curve:
            writer.writerow(row)
    return path


def _save_overlay(records: Sequence[EdgeLabelRecord], metrics: Dict[str, Any], output_dir: str) -> Optional[str]:
    if not records:
        return None
    rec = records[0]
    _, gt = load_record_masks(rec)
    overlay = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    overlay[:, :, :] = 30
    overlay[gt] = [255, 0, 0]
    path = os.path.join(output_dir, "gt_preview_overlay.png")
    Image.fromarray(overlay).save(path)
    return path


def evaluate_promotion_gate(
    metrics: Dict[str, Any],
    baseline_sobel_bf1: float = 0.0,
    min_absolute_bf1: float = 0.85,
    min_delta_bf1: float = 0.10,
    max_merge_split_risk: float = 0.10,
) -> Dict[str, Any]:
    """Evaluate the validation promotion rule for a trained 1px edge model."""
    bf1 = float(metrics.get("topology_safe_bf1", metrics.get("best_bf1", 0.0)))
    exact_f1 = float(metrics.get("topology_safe_exact_f1", metrics.get("exact_f1", 0.0)))
    merge_split_risk = float(metrics.get("topology_safe_merge_split_risk", metrics.get("merge_split_risk", 1.0)))
    max_frame_risk = float(metrics.get("topology_safe_merge_split_risk_max", merge_split_risk))
    baseline = float(baseline_sobel_bf1)
    bf1_gate = bf1 >= min_absolute_bf1 or bf1 >= baseline + min_delta_bf1
    topology_gate = merge_split_risk < max_merge_split_risk and max_frame_risk < max_merge_split_risk
    exact_gate = exact_f1 > 0.0
    passed = bool(bf1_gate and topology_gate and exact_gate)
    reasons = []
    if not bf1_gate:
        reasons.append("bf1_below_absolute_or_delta_gate")
    if not topology_gate:
        reasons.append("merge_split_risk_high")
    if not exact_gate:
        reasons.append("exact_1px_f1_zero")
    return {
        "passed": passed,
        "bf1": bf1,
        "baseline_sobel_bf1": baseline,
        "bf1_delta": bf1 - baseline,
        "merge_split_risk": merge_split_risk,
        "max_frame_merge_split_risk": max_frame_risk,
        "exact_1px_f1": exact_f1,
        "threshold_mode": "topology_safe" if "topology_safe_bf1" in metrics else "bf1",
        "reasons": reasons,
    }


def train_edge_model(
    records: Sequence[EdgeLabelRecord],
    train_config: Optional[TrainConfig | Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> TrainingResult:
    _require_torch()
    cfg = train_config if isinstance(train_config, TrainConfig) else TrainConfig(**(train_config or {}))
    usable = []
    qc_reports = []
    for rec in records:
        mask, skel = load_record_masks(rec)
        qc = validate_edge_label_quality(
            mask,
            min_label_pixels=cfg.min_label_pixels,
            max_density=cfg.max_label_density,
            max_branch_ratio=cfg.max_label_branch_ratio,
            max_thickness_ratio=cfg.max_label_thickness_ratio,
        )
        qc["image_path"] = rec.image_path
        qc_reports.append(qc)
        if qc["valid"] and int(skel.sum()) >= cfg.min_label_pixels:
            usable.append(rec)
    if not usable:
        raise ValueError(f"No usable 1px edge labels were found for training. QC={qc_reports}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    train_records, val_records = _split_records(usable)
    device = _resolve_device(cfg.device)
    model = create_model(cfg.model_preset).to(device)
    initial_weight_hash = _model_state_hash(model)
    effective_preset = _effective_preset(cfg.model_preset)
    dataset = EdgeLabelDataset(train_records, cfg.crop_size)
    loader = DataLoader(dataset, batch_size=max(1, int(cfg.batch_size)), shuffle=True)
    total_pos = 0.0
    total_neg = 0.0
    for rec in train_records:
        _, skel = load_record_masks(rec)
        total_pos += float(skel.sum())
        total_neg += float(skel.size - skel.sum())
    pos_weight = torch.tensor([min(100.0, max(1.0, total_neg / max(total_pos, 1.0)))], device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr_max, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=cfg.lr_max,
        epochs=max(1, cfg.epochs),
        steps_per_epoch=max(1, len(loader)),
    )
    best_metrics = {"best_bf1": -1.0, "best_threshold": 0.5, "topology_safe_objective": -1.0e9}
    best_state = None
    lr_history = []
    start = time.time()
    for epoch in range(max(1, cfg.epochs)):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            optim.zero_grad(set_to_none=True)
            outputs = model(image)
            loss = _edge_loss(outputs, target, cfg, pos_weight, epoch=epoch)
            loss.backward()
            optim.step()
            scheduler.step()
            lr_history.append(float(optim.param_groups[0]["lr"]))
            epoch_loss += float(loss.detach().cpu())
        metrics = _evaluate_model(model, val_records, cfg, device)
        metric_value = metrics.get("topology_safe_objective", metrics.get("best_bf1", -1.0))
        best_value = best_metrics.get("topology_safe_objective", best_metrics.get("best_bf1", -1.0))
        if metric_value > best_value:
            best_metrics = metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if progress_cb:
            progress_cb({
                "epoch": epoch + 1,
                "epochs": cfg.epochs,
                "loss": epoch_loss / max(1, len(loader)),
                "best_bf1": best_metrics.get("topology_safe_bf1", best_metrics["best_bf1"]),
                "merge_split_risk": best_metrics.get("topology_safe_merge_split_risk", best_metrics.get("merge_split_risk", 1.0)),
                "elapsed": time.time() - start,
            })
    if best_state is not None:
        model.load_state_dict(best_state)
    model_path = os.path.join(cfg.output_dir, "best_model.pt")
    threshold_path = os.path.join(cfg.output_dir, "best_threshold.json")
    report_path = os.path.join(cfg.output_dir, "edge_training_report.json")
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg), "preset": cfg.model_preset}, model_path)
    with open(threshold_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "threshold": best_metrics.get("topology_safe_threshold", best_metrics.get("best_threshold", 0.5)),
                "threshold_mode": "topology_safe" if "topology_safe_threshold" in best_metrics else "bf1",
                "bf1_threshold": best_metrics.get("best_threshold", 0.5),
                "topology_safe_threshold": best_metrics.get("topology_safe_threshold"),
            },
            handle,
            indent=2,
        )
    _save_pr_curve(best_metrics, cfg.output_dir)
    _save_overlay(val_records, best_metrics, cfg.output_dir)
    diagnostics = {
        "requested_model_preset": cfg.model_preset,
        "effective_model_preset": effective_preset,
        "adapter_fallback": effective_preset != cfg.model_preset,
        "initial_weight_hash": initial_weight_hash,
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "lr_history": lr_history,
        "pos_weight": float(pos_weight.detach().cpu().item()),
        "device": str(device),
        "n_train_records": len(train_records),
        "n_val_records": len(val_records),
        "label_qc": qc_reports,
    }
    report = {
        "metrics": best_metrics,
        "config": asdict(cfg),
        "diagnostics": diagnostics,
        "promotion_gate": evaluate_promotion_gate(best_metrics),
        "n_records": len(usable),
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
    return TrainingResult(
        best_score=float(best_metrics.get("topology_safe_bf1", best_metrics.get("best_bf1", 0.0))),
        best_threshold=float(best_metrics.get("topology_safe_threshold", best_metrics.get("best_threshold", 0.5))),
        model_path=model_path,
        threshold_path=threshold_path,
        report_path=report_path,
        metrics=best_metrics,
        config={**asdict(cfg), "diagnostics": diagnostics},
    )


def optimize_edge_training(
    records: Sequence[EdgeLabelRecord],
    hpo_config: Optional[HPOConfig | Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    cfg = hpo_config if isinstance(hpo_config, HPOConfig) else HPOConfig(**(hpo_config or {}))
    os.makedirs(cfg.output_dir, exist_ok=True)
    rng = random.Random(cfg.seed)
    trials = []
    best = None
    hpo_strategy = "optuna_available_not_enabled_v1" if _OPTUNA_AVAILABLE else "random_fallback_no_optuna"
    for trial_idx in range(max(1, int(cfg.trial_count))):
        preset = list(cfg.model_presets)[trial_idx % len(cfg.model_presets)]
        train_cfg = TrainConfig(
            output_dir=os.path.join(cfg.output_dir, f"trial_{trial_idx:03d}"),
            model_preset=preset,
            device=cfg.device,
            seed=cfg.seed,
            epochs=max(1, int(cfg.max_epochs_per_trial)),
            batch_size=max(1, int(cfg.batch_size)),
            crop_size=max(8, int(cfg.crop_size)),
            lr_max=10 ** rng.uniform(-4.5, -2.7),
            weight_decay=10 ** rng.uniform(-6.0, -3.0),
            focal_weight=rng.uniform(0.0, 0.20),
            focal_gamma=rng.uniform(1.5, 3.0),
            tversky_weight=rng.uniform(0.0, 0.20),
            cldice_weight=rng.uniform(0.01, 0.10),
            cbdice_weight=rng.uniform(0.0, 0.10),
            negative_centerline_weight=rng.uniform(0.0, 0.10),
            background_distance_weight=rng.uniform(0.05, 0.50),
            simplified_topology_weight=rng.uniform(0.0, 0.10),
            topology_failure_weight=rng.uniform(1.0, 4.0),
            topology_risk_tolerance=rng.uniform(0.04, 0.10),
            false_bridge_weight=rng.uniform(0.75, 2.0),
            split_gap_weight=rng.uniform(0.75, 2.0),
            topology_penalty=rng.uniform(0.20, 0.60),
            topology_failure_penalty=rng.uniform(1.5, 5.0),
            density_penalty=rng.uniform(0.10, 0.40),
            min_component_pixels=rng.randint(2, 8),
            spur_prune_iters=rng.randint(0, 3),
            max_bridge_gap=rng.randint(0, 3),
            protect_topology_postprocess=True,
            final_thinning_postprocess=True,
        )
        status = "completed"
        try:
            result = train_edge_model(records, train_cfg, progress_cb=progress_cb)
            value = float(result.metrics.get("topology_safe_objective", result.best_score))
        except Exception as exc:
            result = None
            value = 0.0
            status = "failed"
            if progress_cb:
                progress_cb({"trial": trial_idx, "status": status, "error": str(exc)})
        record = {
            "trial": trial_idx,
            "status": status,
            "score": float(value),
            "config": asdict(train_cfg),
            "result": asdict(result) if result is not None else None,
            "initial_weight_hash": result.config.get("diagnostics", {}).get("initial_weight_hash") if result is not None else None,
        }
        trials.append(record)
        if result is not None and (best is None or value > best["score"]):
            best = record
    summary = {
        "best": best,
        "trials": trials,
        "hpo_strategy": hpo_strategy,
        "optuna_available": _OPTUNA_AVAILABLE,
        "trial_reset_verified": len({t.get("initial_weight_hash") for t in trials if t.get("initial_weight_hash")}) <= 1,
    }
    with open(os.path.join(cfg.output_dir, "hpo_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)
    return summary
