import os
import time
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

from sobel_edge_detection import SobelEdgeDetector, _make_overlay_image


def create_bending_loop_mask(width=640, height=480, line_width=42):
    """Create a single connected bending-loop mask."""
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    path = [
        (60, 200),
        (360, 200),
        (470, 260),
        (470, 360),
        (360, 420),
        (60, 420),
    ]
    draw.line(path, fill=255, width=line_width, joint="curve")

    # Connector pads to suggest the FCB ends.
    draw.rectangle([20, 180, 90, 440], fill=255)
    draw.rectangle([430, 240, 540, 340], fill=255)

    return np.array(mask_img, dtype=np.uint8)


def render_image_from_mask(mask, background=230, foreground=45, noise_sigma=4, seed=7):
    """Render a grayscale image from a mask with mild noise."""
    image = np.full(mask.shape, background, dtype=np.float32)
    image[mask > 0] = foreground

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_sigma, size=mask.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return image


def compute_boundary(mask):
    """Compute boundary pixels for a binary mask."""
    mask_bool = mask > 0
    padded = np.pad(mask_bool, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    boundary = np.zeros_like(center, dtype=bool)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = padded[1 + dy : 1 + dy + center.shape[0], 1 + dx : 1 + dx + center.shape[1]]
            boundary |= neighbor != center

    return boundary & center


def dilate(binary, radius=1):
    """Simple binary dilation with a square structuring element."""
    if radius <= 0:
        return binary.copy()

    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    out = np.zeros_like(binary, dtype=bool)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out |= padded[dy : dy + binary.shape[0], dx : dx + binary.shape[1]]
    return out


def evaluate_edges(pred_edges, gt_edges, tolerance=1):
    """Evaluate precision/recall/F1 with a pixel tolerance."""
    pred = pred_edges.astype(bool)
    gt = gt_edges.astype(bool)

    gt_tol = dilate(gt, tolerance)
    pred_tol = dilate(pred, tolerance)

    tp = int(np.logical_and(pred, gt_tol).sum())
    fp = int(np.logical_and(pred, ~gt_tol).sum())
    fn = int(np.logical_and(gt, ~pred_tol).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_edge_pixels": int(pred.sum()),
        "gt_edge_pixels": int(gt.sum()),
    }


def ensure_output_dir(root="outputs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(root, f"perf_eval_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as handle:
        for key, value in metrics.items():
            if isinstance(value, float):
                handle.write(f"{key}: {value:.4f}\n")
            else:
                handle.write(f"{key}: {value}\n")


def _save_debug_overlay(original, pred_edges, missing_edges, path):
    overlay = np.stack([original] * 3, axis=-1)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay[pred_edges] = [0, 255, 0]
    overlay[missing_edges] = [0, 0, 255]
    Image.fromarray(overlay).save(path)


def _run_case(
    detector,
    output_dir,
    name,
    line_width,
    noise_sigma,
    background,
    foreground,
    settings,
):
    mask = create_bending_loop_mask(line_width=line_width)
    image = render_image_from_mask(
        mask,
        background=background,
        foreground=foreground,
        noise_sigma=noise_sigma,
        seed=7,
    )

    input_path = os.path.join(output_dir, f"{name}_input.png")
    mask_path = os.path.join(output_dir, f"{name}_mask.png")
    Image.fromarray(image).save(input_path)
    Image.fromarray(mask).save(mask_path)

    start = time.perf_counter()
    results = detector.detect_edges(
        input_path,
        use_nms=True,
        use_hysteresis=True,
        use_blur=settings["use_blur"],
        blur_kernel_size=settings["blur_kernel_size"],
        blur_sigma=settings["blur_sigma"],
        use_contrast_stretch=settings["use_contrast_stretch"],
        contrast_low_pct=settings["contrast_low_pct"],
        contrast_high_pct=settings["contrast_high_pct"],
        magnitude_gamma=settings["magnitude_gamma"],
        low_ratio=settings["low_ratio"],
        high_ratio=settings["high_ratio"],
        threshold_method=settings["threshold_method"],
        low_percentile=settings["low_percentile"],
        high_percentile=settings["high_percentile"],
        min_threshold=settings["min_threshold"],
    )
    elapsed = time.perf_counter() - start

    pred_edges = results["edges"] > 0
    gt_edges = compute_boundary(mask)

    overlay, _ = _make_overlay_image(results["original"], results["edges"])
    overlay_path = os.path.join(output_dir, f"{name}_edges_green.png")
    Image.fromarray(overlay).save(overlay_path)

    pred_edge_path = os.path.join(output_dir, f"{name}_edges_binary.png")
    gt_edge_path = os.path.join(output_dir, f"{name}_edges_gt.png")
    Image.fromarray((pred_edges * 255).astype(np.uint8)).save(pred_edge_path)
    Image.fromarray((gt_edges * 255).astype(np.uint8)).save(gt_edge_path)

    missing_edges = gt_edges & ~dilate(pred_edges, radius=1)
    missing_overlay_path = os.path.join(output_dir, f"{name}_edges_missing.png")
    _save_debug_overlay(results["original"], pred_edges, missing_edges, missing_overlay_path)

    metrics = evaluate_edges(pred_edges, gt_edges, tolerance=1)
    metrics["elapsed_sec"] = elapsed
    metrics["line_width"] = line_width
    metrics["noise_sigma"] = noise_sigma
    metrics["background"] = background
    metrics["foreground"] = foreground
    for key, value in settings.items():
        metrics[f"setting_{key}"] = value

    metrics_path = os.path.join(output_dir, f"edge_metrics_{name}.txt")
    save_metrics(metrics, metrics_path)

    return metrics, overlay_path, missing_overlay_path


def main():
    output_dir = ensure_output_dir()

    detector = SobelEdgeDetector()
    strategies = [
        {
            "name": "ratio_blur5",
            "settings": {
                "use_blur": True,
                "blur_kernel_size": 5,
                "blur_sigma": 1.2,
                "use_contrast_stretch": False,
                "contrast_low_pct": 2.0,
                "contrast_high_pct": 98.0,
                "magnitude_gamma": 1.0,
                "low_ratio": 0.06,
                "high_ratio": 0.18,
                "threshold_method": "ratio",
                "low_percentile": 35.0,
                "high_percentile": 80.0,
                "min_threshold": 1.0,
            },
        },
        {
            "name": "percentile_contrast",
            "settings": {
                "use_blur": True,
                "blur_kernel_size": 3,
                "blur_sigma": 0.9,
                "use_contrast_stretch": True,
                "contrast_low_pct": 2.0,
                "contrast_high_pct": 98.0,
                "magnitude_gamma": 1.0,
                "low_ratio": 0.06,
                "high_ratio": 0.18,
                "threshold_method": "percentile",
                "low_percentile": 35.0,
                "high_percentile": 80.0,
                "min_threshold": 1.0,
            },
        },
        {
            "name": "percentile_gamma",
            "settings": {
                "use_blur": True,
                "blur_kernel_size": 3,
                "blur_sigma": 0.9,
                "use_contrast_stretch": True,
                "contrast_low_pct": 1.5,
                "contrast_high_pct": 98.5,
                "magnitude_gamma": 0.85,
                "low_ratio": 0.06,
                "high_ratio": 0.18,
                "threshold_method": "percentile",
                "low_percentile": 30.0,
                "high_percentile": 75.0,
                "min_threshold": 1.0,
            },
        },
    ]

    cases = [
        {"name": "bending_loop", "line_width": 42, "noise_sigma": 4, "background": 230, "foreground": 45},
        {"name": "bending_loop_thin", "line_width": 24, "noise_sigma": 5, "background": 230, "foreground": 45},
        {"name": "bending_loop_faint_thin", "line_width": 18, "noise_sigma": 6, "background": 200, "foreground": 150},
    ]

    print("Performance evaluation complete.")
    print(f"Output directory: {output_dir}")

    for strategy in strategies:
        strategy_name = strategy["name"]
        settings = strategy["settings"]
        print(f"\n=== Strategy: {strategy_name} ===")
        for case in cases:
            case_tag = f"{strategy_name}_{case['name']}"
            metrics, overlay_path, missing_overlay_path = _run_case(
                detector,
                output_dir,
                case_tag,
                case["line_width"],
                case["noise_sigma"],
                case["background"],
                case["foreground"],
                settings,
            )
            print(f"\nCase: {case_tag}")
            print(f"- overlay: {overlay_path}")
            print(f"- missing: {missing_overlay_path}")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")


if __name__ == "__main__":
    main()
