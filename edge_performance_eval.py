import os
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


def main():
    output_dir = ensure_output_dir()

    mask = create_bending_loop_mask()
    image = render_image_from_mask(mask)

    input_path = os.path.join(output_dir, "bending_loop_input.png")
    mask_path = os.path.join(output_dir, "bending_loop_mask.png")
    Image.fromarray(image).save(input_path)
    Image.fromarray(mask).save(mask_path)

    detector = SobelEdgeDetector()
    settings = {
        "use_blur": True,
        "blur_kernel_size": 5,
        "blur_sigma": 1.4,
        "low_ratio": 0.12,
        "high_ratio": 0.25,
    }
    results = detector.detect_edges(
        input_path,
        use_nms=True,
        use_hysteresis=True,
        use_blur=settings["use_blur"],
        blur_kernel_size=settings["blur_kernel_size"],
        blur_sigma=settings["blur_sigma"],
        low_ratio=settings["low_ratio"],
        high_ratio=settings["high_ratio"],
    )

    pred_edges = results["edges"] > 0
    gt_edges = compute_boundary(mask)

    overlay, _ = _make_overlay_image(results["original"], results["edges"])
    overlay_path = os.path.join(output_dir, "bending_loop_edges_green.png")
    Image.fromarray(overlay).save(overlay_path)

    pred_edge_path = os.path.join(output_dir, "bending_loop_edges_binary.png")
    gt_edge_path = os.path.join(output_dir, "bending_loop_edges_gt.png")
    Image.fromarray((pred_edges * 255).astype(np.uint8)).save(pred_edge_path)
    Image.fromarray((gt_edges * 255).astype(np.uint8)).save(gt_edge_path)

    metrics = evaluate_edges(pred_edges, gt_edges, tolerance=1)
    for key, value in settings.items():
        metrics[f"setting_{key}"] = value
    metrics_path = os.path.join(output_dir, "edge_metrics.txt")
    save_metrics(metrics, metrics_path)

    print("Performance evaluation complete.")
    print(f"Output directory: {output_dir}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
