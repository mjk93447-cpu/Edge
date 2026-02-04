import os
import time
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

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


def render_image_from_mask(
    mask,
    background=230,
    foreground=45,
    noise_sigma=4,
    seed=7,
    gradient_strength=0.0,
    gradient_axis="x",
    blur_radius=0.0,
    downscale_factor=1,
):
    """Render a grayscale image from a mask with mild noise."""
    image = np.full(mask.shape, background, dtype=np.float32)
    image[mask > 0] = foreground

    if gradient_strength:
        if gradient_axis == "y":
            grad = np.linspace(-gradient_strength / 2, gradient_strength / 2, mask.shape[0])
            image += grad[:, None]
        else:
            grad = np.linspace(-gradient_strength / 2, gradient_strength / 2, mask.shape[1])
            image += grad[None, :]

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_sigma, size=mask.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    if blur_radius > 0 or downscale_factor > 1:
        pil_img = Image.fromarray(image)
        if downscale_factor > 1:
            w, h = pil_img.size
            small = pil_img.resize(
                (max(1, w // downscale_factor), max(1, h // downscale_factor)),
                resample=Image.BILINEAR,
            )
            pil_img = small.resize((w, h), resample=Image.BILINEAR)
        if blur_radius > 0:
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_radius))
        image = np.array(pil_img, dtype=np.uint8)
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


def compute_intrusion(pred_edges, mask, boundary, band_radius=1):
    """Measure how much edges intrude into object interior."""
    boundary_band = dilate(boundary, band_radius)
    interior = (mask > 0) & ~boundary_band
    intrusion = pred_edges & interior
    intrusion_pixels = int(intrusion.sum())
    pred_pixels = int(pred_edges.sum())
    intrusion_ratio = intrusion_pixels / pred_pixels if pred_pixels else 0.0
    return intrusion_pixels, intrusion_ratio


def compute_alignment_metrics(pred_edges, mask, boundary, band_radius=1):
    """Measure how well edges align to boundary band."""
    boundary_band = dilate(boundary, band_radius)
    pred_pixels = int(pred_edges.sum())
    if pred_pixels == 0:
        return 0, 0.0, 0, 0.0

    within_band = pred_edges & boundary_band
    band_pixels = int(within_band.sum())
    band_ratio = band_pixels / pred_pixels

    outside = pred_edges & (~mask) & (~boundary_band)
    outside_pixels = int(outside.sum())
    outside_ratio = outside_pixels / pred_pixels
    return band_pixels, band_ratio, outside_pixels, outside_ratio


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
    gradient_strength,
    gradient_axis,
    blur_radius,
    downscale_factor,
    settings,
):
    mask = create_bending_loop_mask(line_width=line_width)
    image = render_image_from_mask(
        mask,
        background=background,
        foreground=foreground,
        noise_sigma=noise_sigma,
        seed=7,
        gradient_strength=gradient_strength,
        gradient_axis=gradient_axis,
        blur_radius=blur_radius,
        downscale_factor=downscale_factor,
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
        use_median_filter=settings["use_median_filter"],
        median_kernel_size=settings["median_kernel_size"],
        use_blur=settings["use_blur"],
        blur_kernel_size=settings["blur_kernel_size"],
        blur_sigma=settings["blur_sigma"],
        use_contrast_stretch=settings["use_contrast_stretch"],
        contrast_low_pct=settings["contrast_low_pct"],
        contrast_high_pct=settings["contrast_high_pct"],
        magnitude_gamma=settings["magnitude_gamma"],
        nms_relax=settings["nms_relax"],
        low_ratio=settings["low_ratio"],
        high_ratio=settings["high_ratio"],
        threshold_method=settings["threshold_method"],
        low_percentile=settings["low_percentile"],
        high_percentile=settings["high_percentile"],
        min_threshold=settings["min_threshold"],
        mad_low_k=settings["mad_low_k"],
        mad_high_k=settings["mad_high_k"],
        use_soft_linking=settings["use_soft_linking"],
        soft_low_ratio=settings["soft_low_ratio"],
        soft_high_ratio=settings["soft_high_ratio"],
        link_radius=settings["link_radius"],
        soft_threshold_method=settings["soft_threshold_method"],
        soft_low_percentile=settings["soft_low_percentile"],
        soft_high_percentile=settings["soft_high_percentile"],
        soft_mad_low_k=settings["soft_mad_low_k"],
        soft_mad_high_k=settings["soft_mad_high_k"],
        use_closing=settings["use_closing"],
        closing_radius=settings["closing_radius"],
        closing_iterations=settings["closing_iterations"],
        use_peak_refine=settings["use_peak_refine"],
        peak_fill_radius=settings["peak_fill_radius"],
        use_polarity_filter=settings["use_polarity_filter"],
        polarity_min_diff=settings["polarity_min_diff"],
        polarity_min_support=settings["polarity_min_support"],
        polarity_drop_margin=settings["polarity_drop_margin"],
        use_boundary_band_filter=settings["use_boundary_band_filter"],
        boundary_band_radius=settings["boundary_band_radius"],
        mask_min_area=settings["mask_min_area"],
        mask_max_area=settings["mask_max_area"],
        object_is_dark=settings["object_is_dark"],
        use_mask_blur=settings["use_mask_blur"],
        mask_blur_kernel_size=settings["mask_blur_kernel_size"],
        mask_blur_sigma=settings["mask_blur_sigma"],
        mask_close_radius=settings["mask_close_radius"],
        use_thinning=settings["use_thinning"],
        thinning_max_iter=settings["thinning_max_iter"],
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
    intrusion_pixels, intrusion_ratio = compute_intrusion(pred_edges, mask, gt_edges, band_radius=1)
    metrics["intrusion_pixels"] = intrusion_pixels
    metrics["intrusion_ratio"] = intrusion_ratio
    band_pixels, band_ratio, outside_pixels, outside_ratio = compute_alignment_metrics(
        pred_edges, mask, gt_edges, band_radius=1
    )
    metrics["band_pixels"] = band_pixels
    metrics["band_ratio"] = band_ratio
    metrics["outside_pixels"] = outside_pixels
    metrics["outside_ratio"] = outside_ratio
    metrics["elapsed_sec"] = elapsed
    metrics["line_width"] = line_width
    metrics["noise_sigma"] = noise_sigma
    metrics["background"] = background
    metrics["foreground"] = foreground
    metrics["gradient_strength"] = gradient_strength
    metrics["gradient_axis"] = gradient_axis
    metrics["blur_radius"] = blur_radius
    metrics["downscale_factor"] = downscale_factor
    for key, value in settings.items():
        metrics[f"setting_{key}"] = value

    metrics_path = os.path.join(output_dir, f"edge_metrics_{name}.txt")
    save_metrics(metrics, metrics_path)

    return metrics, overlay_path, missing_overlay_path


def main():
    output_dir = ensure_output_dir()

    detector = SobelEdgeDetector()
    base_settings = {
        "use_median_filter": False,
        "median_kernel_size": 3,
        "use_blur": True,
        "blur_kernel_size": 5,
        "blur_sigma": 1.2,
        "use_contrast_stretch": False,
        "contrast_low_pct": 2.0,
        "contrast_high_pct": 98.0,
        "magnitude_gamma": 1.0,
        "nms_relax": 1.0,
        "low_ratio": 0.04,
        "high_ratio": 0.12,
        "threshold_method": "ratio",
        "low_percentile": 35.0,
        "high_percentile": 80.0,
        "min_threshold": 1.0,
        "mad_low_k": 1.5,
        "mad_high_k": 3.0,
        "use_soft_linking": False,
        "soft_low_ratio": 0.03,
        "soft_high_ratio": 0.1,
        "link_radius": 2,
        "soft_threshold_method": None,
        "soft_low_percentile": None,
        "soft_high_percentile": None,
        "soft_mad_low_k": None,
        "soft_mad_high_k": None,
        "use_closing": False,
        "closing_radius": 1,
        "closing_iterations": 1,
        "use_peak_refine": False,
        "peak_fill_radius": 1,
        "use_polarity_filter": True,
        "polarity_min_diff": 1.0,
        "polarity_min_support": 50,
        "polarity_drop_margin": 0.5,
        "use_boundary_band_filter": True,
        "boundary_band_radius": 2,
        "mask_min_area": 0.05,
        "mask_max_area": 0.95,
        "object_is_dark": True,
        "use_mask_blur": True,
        "mask_blur_kernel_size": 5,
        "mask_blur_sigma": 1.0,
        "mask_close_radius": 1,
        "use_thinning": True,
        "thinning_max_iter": 15,
    }

    cases = [
        {
            "name": "bending_loop",
            "line_width": 42,
            "noise_sigma": 4,
            "background": 230,
            "foreground": 45,
            "gradient_strength": 0.0,
            "gradient_axis": "x",
            "blur_radius": 0.0,
            "downscale_factor": 1,
        },
        {
            "name": "bending_loop_thin",
            "line_width": 24,
            "noise_sigma": 5,
            "background": 230,
            "foreground": 45,
            "gradient_strength": 0.0,
            "gradient_axis": "x",
            "blur_radius": 0.0,
            "downscale_factor": 1,
        },
        {
            "name": "fcb_side_like",
            "line_width": 18,
            "noise_sigma": 6,
            "background": 190,
            "foreground": 155,
            "gradient_strength": 30.0,
            "gradient_axis": "y",
            "blur_radius": 0.8,
            "downscale_factor": 1,
        },
        {
            "name": "fcb_side_low_quality",
            "line_width": 18,
            "noise_sigma": 8,
            "background": 200,
            "foreground": 175,
            "gradient_strength": 40.0,
            "gradient_axis": "y",
            "blur_radius": 1.5,
            "downscale_factor": 2,
        },
    ]

    candidate_relax = [1.0, 0.98, 0.96, 0.95, 0.94, 0.92, 0.9]
    search_results = []

    print("Auto-searching nms_relax values...")
    for relax in candidate_relax:
        settings = dict(base_settings)
        settings["nms_relax"] = relax

        total_recall = 0.0
        total_intrusion = 0.0
        total_precision = 0.0
        total_band_ratio = 0.0
        total_outside_ratio = 0.0
        for case in cases:
            metrics, _, _ = _run_case(
                detector,
                output_dir,
                f"search_relax_{relax:.2f}_{case['name']}",
                case["line_width"],
                case["noise_sigma"],
                case["background"],
                case["foreground"],
                case["gradient_strength"],
                case["gradient_axis"],
                case["blur_radius"],
                case["downscale_factor"],
                settings,
            )
            total_recall += metrics["recall"]
            total_intrusion += metrics["intrusion_ratio"]
            total_precision += metrics["precision"]
            total_band_ratio += metrics["band_ratio"]
            total_outside_ratio += metrics["outside_ratio"]

        avg_recall = total_recall / len(cases)
        avg_intrusion = total_intrusion / len(cases)
        avg_precision = total_precision / len(cases)
        avg_band_ratio = total_band_ratio / len(cases)
        avg_outside_ratio = total_outside_ratio / len(cases)
        score = avg_recall + 0.3 * avg_band_ratio - 1.2 * avg_intrusion - 0.7 * avg_outside_ratio
        search_results.append((relax, avg_recall, avg_intrusion, avg_precision, score))
        print(
            f"relax={relax:.2f} avg_recall={avg_recall:.4f} "
            f"avg_intrusion={avg_intrusion:.4f} avg_band={avg_band_ratio:.4f} "
            f"avg_outside={avg_outside_ratio:.4f} avg_precision={avg_precision:.4f} score={score:.4f}"
        )

    search_results.sort(key=lambda x: (x[4], x[1]), reverse=True)
    best_relax = search_results[0][0]
    print(f"\nBest nms_relax: {best_relax:.2f}")

    strategies = [
        {
            "name": f"best_relax_{best_relax:.2f}_band2",
            "settings": dict(base_settings, nms_relax=best_relax, boundary_band_radius=2),
        },
        {
            "name": f"best_relax_{best_relax:.2f}_band3",
            "settings": dict(base_settings, nms_relax=best_relax, boundary_band_radius=3),
        },
        {
            "name": f"best_relax_{best_relax:.2f}_no_band",
            "settings": dict(base_settings, nms_relax=best_relax, use_boundary_band_filter=False),
        },
    ]

    print("\nPerformance evaluation complete.")
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
                case["gradient_strength"],
                case["gradient_axis"],
                case["blur_radius"],
                case["downscale_factor"],
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
