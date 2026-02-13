import json
import math
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError:
    tk = None
    filedialog = None
    messagebox = None
    ttk = None

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageTk, ImageDraw

# Display scale for small scores so UI shows readable numbers (learning uses raw score only)
SCORE_DISPLAY_SCALE = 1e15

PARAM_DEFAULTS = {
    "nms_relax": 0.95,
    "low_ratio": 0.04,
    "high_ratio": 0.12,
    "use_median_filter": True,
    "median_kernel_size": 3,
    "use_blur": True,
    "blur_sigma": 1.2,
    "blur_kernel_size": 5,
    "magnitude_gamma": 1.0,
    "use_contrast_stretch": False,
    "contrast_low_pct": 2.0,
    "contrast_high_pct": 98.0,
    "auto_threshold": True,
    "contrast_ref": 80.0,
    "min_threshold_scale": 0.5,
    "use_soft_linking": False,
    "soft_low_ratio": 0.03,
    "soft_high_ratio": 0.1,
    "link_radius": 2,
    "use_closing": False,
    "closing_radius": 1,
    "closing_iterations": 1,
    "use_thinning": True,
    "thinning_max_iter": 15,
    "use_edge_smooth": False,
    "edge_smooth_radius": 1,
    "edge_smooth_iters": 1,
    "spur_prune_iters": 0,
    "use_boundary_band_filter": True,
    "boundary_band_radius": 2,
    "object_is_dark": True,
    "use_mask_blur": True,
    "mask_blur_sigma": 1.0,
    "mask_blur_kernel_size": 5,
    "mask_close_radius": 1,
    "use_polarity_filter": True,
    "polarity_drop_margin": 0.5,
}

# Optimal defaults: high-impact params use finer steps; low-impact use coarser or fixed ranges.
AUTO_DEFAULTS = {
    "auto_nms_min": 0.90,
    "auto_nms_max": 0.97,
    "auto_nms_step": 0.005,
    "auto_high_min": 0.07,
    "auto_high_max": 0.14,
    "auto_high_step": 0.005,
    "auto_low_factor_min": 0.30,
    "auto_low_factor_max": 0.42,
    "auto_low_factor_step": 0.02,
    "auto_band_min": 1,
    "auto_band_max": 3,
    "auto_margin_min": 0.10,
    "auto_margin_max": 0.45,
    "auto_margin_step": 0.02,
    "auto_blur_sigma_min": 1.0,
    "auto_blur_sigma_max": 1.4,
    "auto_blur_sigma_step": 0.1,
    "auto_blur_kernel_min": 3,
    "auto_blur_kernel_max": 7,
    "auto_blur_kernel_step": 2,
    "auto_thinning_min": 10,
    "auto_thinning_max": 18,
    "auto_thinning_step": 2,
    "auto_contrast_ref_min": 70.0,
    "auto_contrast_ref_max": 110.0,
    "auto_contrast_ref_step": 5.0,
    "auto_min_scale_min": 0.50,
    "auto_min_scale_max": 0.75,
    "auto_min_scale_step": 0.05,
    "auto_soft_high_min": 0.08,
    "auto_soft_high_max": 0.16,
    "auto_soft_high_step": 0.02,
    "auto_soft_low_factor_min": 0.28,
    "auto_soft_low_factor_max": 0.42,
    "auto_soft_low_factor_step": 0.05,
    "auto_link_radius_min": 1,
    "auto_link_radius_max": 4,
    "auto_link_radius_step": 1,
    "auto_soft_link_prob": 0.4,
    "auto_edge_smooth_radius_min": 0,
    "auto_edge_smooth_radius_max": 2,
    "auto_edge_smooth_iters_min": 1,
    "auto_edge_smooth_iters_max": 2,
    "auto_spur_prune_min": 0,
    "auto_spur_prune_max": 2,
    "auto_use_edge_smooth_prob": 0.5,
    "auto_use_closing_prob": 0.25,
    "auto_closing_radius_min": 1,
    "auto_closing_radius_max": 2,
    "auto_closing_iter_min": 1,
    "auto_closing_iter_max": 2,
    "auto_magnitude_gamma_min": 0.9,
    "auto_magnitude_gamma_max": 1.2,
    "auto_magnitude_gamma_step": 0.1,
    "auto_median_kernel_min": 3,
    "auto_median_kernel_max": 5,
    "auto_median_kernel_step": 2,
    "weight_continuity": 24.0,
    "weight_band_fit": 12.0,
    "weight_coverage": 1.0,
    "weight_gap": 1.0,
    "weight_outside": 1.0,
    "weight_thickness": 1.2,
    "weight_intrusion": 1.0,
    "weight_endpoints": 1.0,
    "weight_wrinkle": 1.0,
    "weight_branch": 2.0,
    "weight_low_quality": 0.5,
    "early_stop_enabled": True,
    "early_stop_minutes": 10.0,
    "auto_round_early_exit_no_improve_frac": 0.25,
    "auto_no_improve_rounds_stop": 2,
    "auto_phase1_max_thickness": 0.25,
    "auto_phase1_frac": 0.5,
    "auto_phase1_min_evals": 150,
    "auto_parallel": True,
    "auto_workers": max(1, (os.cpu_count() or 4) - 1),
}


def get_auto_profile_overrides(grayscale=None, low_quality=None):
    """Return dict of auto_config overrides based on user answers. None = not sure (no override)."""
    overrides = {}
    if grayscale is True:
        overrides["auto_contrast_ref_min"] = 75.0
        overrides["auto_contrast_ref_max"] = 105.0
        overrides["auto_contrast_ref_step"] = 5.0
        overrides["auto_magnitude_gamma_min"] = 0.95
        overrides["auto_magnitude_gamma_max"] = 1.15
    elif grayscale is False:
        overrides["auto_contrast_ref_min"] = 65.0
        overrides["auto_contrast_ref_max"] = 115.0
        overrides["auto_contrast_ref_step"] = 5.0
    if low_quality is True:
        overrides["auto_blur_sigma_min"] = 1.1
        overrides["auto_blur_sigma_max"] = 1.8
        overrides["auto_blur_sigma_step"] = 0.15
        overrides["auto_nms_min"] = 0.88
        overrides["auto_nms_max"] = 0.98
        overrides["auto_nms_step"] = 0.008
        overrides["auto_spur_prune_max"] = 3
        overrides["auto_use_edge_smooth_prob"] = 0.6
        overrides["weight_low_quality"] = 0.7
    elif low_quality is False:
        overrides["auto_blur_sigma_min"] = 1.0
        overrides["auto_blur_sigma_max"] = 1.35
        overrides["auto_blur_sigma_step"] = 0.08
        overrides["auto_nms_step"] = 0.005
    return overrides

PARAM_KEYS = tuple(PARAM_DEFAULTS.keys())
AUTO_KEYS = tuple(AUTO_DEFAULTS.keys())
ROI_CACHE_PATH = os.path.abspath("roi_cache.json")

# Perfect mode: importance-weighted step multipliers (1/N = N× denser grid). Score function unchanged.
# Based on sensitivity: threshold/NMS/high/low/margin have highest impact on edge quality.
PERFECT_STEP_MULTIPLIERS = {
    "nms_relax": 0.2,
    "high_ratio": 0.2,
    "low_ratio": 0.25,
    "polarity_drop_margin": 0.2,
    "boundary_band_radius": 0.4,
    "blur_sigma": 0.4,
    "blur_kernel_size": 0.5,
    "thinning_max_iter": 0.5,
    "contrast_ref": 0.5,
    "min_threshold_scale": 0.5,
    "magnitude_gamma": 0.5,
    "median_kernel_size": 0.5,
    "soft_high_ratio": 0.5,
    "soft_low_ratio": 0.5,
    "link_radius": 0.5,
    "edge_smooth_radius": 0.7,
    "edge_smooth_iters": 0.7,
    "spur_prune_iters": 0.7,
    "closing_radius": 0.7,
    "closing_iterations": 0.7,
}


def save_json_config(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def load_json_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_auto_score(metrics, weights, return_details=False):
    """
    Priority: continuity (20x+) > band_fit (10x+) > thickness > intrusion > others.
    Uses weighted geometric mean (ML-style) so low continuity/band strongly penalizes score.
    """
    coverage = float(metrics.get("coverage", 0.0))
    gap_ratio = float(metrics.get("gap", 1.0))
    continuity = float(metrics.get("continuity", 1.0))
    intrusion = float(metrics.get("intrusion", 1.0))
    outside = float(metrics.get("outside", 1.0))
    thickness = float(metrics.get("thickness", 1.0))
    band_ratio = float(metrics.get("band_ratio", 0.0))
    endpoint_ratio = float(metrics.get("endpoints", 1.0))
    wrinkle_ratio = float(metrics.get("wrinkle", 1.0))
    branch_ratio = float(metrics.get("branch", 1.0))

    w_cont = float(weights.get("weight_continuity", 24.0))
    w_band = float(weights.get("weight_band_fit", 12.0))
    w_cov = float(weights.get("weight_coverage", 1.0))
    w_gap = float(weights.get("weight_gap", 1.0))
    w_out = float(weights.get("weight_outside", 1.0))
    w_thick = float(weights.get("weight_thickness", 1.2))
    w_intr = float(weights.get("weight_intrusion", 1.0))
    w_end = float(weights.get("weight_endpoints", 1.0))
    w_wrinkle = float(weights.get("weight_wrinkle", 1.0))
    w_branch = float(weights.get("weight_branch", 1.0))

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))

    q_cont = sigmoid((0.12 - continuity) / 0.04)
    q_band = sigmoid((band_ratio - 0.85) / 0.08)
    q_cov = sigmoid((coverage - 0.85) / 0.08)
    q_gap = sigmoid((0.18 - gap_ratio) / 0.06)
    q_out = sigmoid((0.05 - outside) / 0.03)
    q_thick = sigmoid((0.15 - thickness) / 0.08)
    q_intr = sigmoid((0.03 - intrusion) / 0.02)
    q_end = sigmoid((0.05 - endpoint_ratio) / 0.02)
    q_wrinkle = sigmoid((0.20 - wrinkle_ratio) / 0.06)
    q_branch = sigmoid((0.08 - branch_ratio) / 0.04)

    weights_list = [w_cont, w_band, w_cov, w_gap, w_thick, w_intr, w_out, w_end, w_wrinkle, w_branch]
    q_list = [q_cont, q_band, q_cov, q_gap, q_thick, q_intr, q_out, q_end, q_wrinkle, q_branch]
    total_w = sum(weights_list)
    eps = 1e-12
    log_score = 0.0
    for qi, wi in zip(q_list, weights_list):
        log_score += (wi / max(total_w, eps)) * math.log(qi + eps)
    score = math.exp(log_score)
    exp_penalty = math.exp(-2.5 * (endpoint_ratio + wrinkle_ratio + branch_ratio))
    score *= exp_penalty
    score = max(0.0, min(1.0, score))
    if return_details:
        return score, {
            "q_cont": q_cont,
            "q_band": q_band,
            "q_cov": q_cov,
            "q_gap": q_gap,
            "q_thick": q_thick,
            "q_intr": q_intr,
            "q_out": q_out,
            "q_end": q_end,
            "q_wrinkle": q_wrinkle,
            "q_branch": q_branch,
        }
    return score

class SobelEdgeDetector:
    """Sobel 필터 기반 에지 검출 클래스"""
    
    def __init__(self):
        # Sobel 커널 정의
        self.sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        
        self.sobel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
    
    def load_image(self, image_path):
        """이미지 로드 및 그레이스케일 변환"""
        img = Image.open(image_path).convert('L')  # 그레이스케일 변환
        return np.array(img, dtype=np.float32)
    
    def apply_convolution(self, image, kernel):
        """컨볼루션 연산 적용"""
        h, w = image.shape
        kernel = kernel.astype(np.float32, copy=False)
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        # 패딩 추가
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        windows = sliding_window_view(padded, (k_h, k_w))
        result = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))
        return result.astype(np.float32, copy=False)

    def apply_gaussian_blur(self, image, kernel_size=5, sigma=1.2):
        """가우시안 블러로 노이즈를 완화하여 정밀도 개선"""
        if kernel_size < 3:
            return image
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = max(float(sigma), 0.1)

        radius = kernel_size // 2
        ax = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        return self.apply_convolution(image, kernel)

    def apply_median_filter(self, image, kernel_size=3):
        """메디안 필터로 노이즈를 줄이고 경계를 보존"""
        if kernel_size < 3:
            return image
        if kernel_size % 2 == 0:
            kernel_size += 1

        radius = kernel_size // 2
        padded = np.pad(image, radius, mode="edge")
        windows = sliding_window_view(padded, (kernel_size, kernel_size))
        median = np.median(windows, axis=(-2, -1))
        return median.astype(np.float32, copy=False)

    def _neighbor_count(self, mask):
        mask_u8 = mask.astype(np.uint8, copy=False)
        padded = np.pad(mask_u8, 1, mode="constant", constant_values=0)
        windows = sliding_window_view(padded, (3, 3))
        counts = windows.sum(axis=(2, 3)) - mask_u8
        return counts

    def prune_spurs(self, edge_mask, iterations=1):
        edge_mask = edge_mask.astype(bool, copy=True)
        iterations = max(0, int(iterations))
        if iterations <= 0:
            return edge_mask
        for _ in range(iterations):
            counts = self._neighbor_count(edge_mask)
            endpoints = edge_mask & (counts <= 1)
            if not endpoints.any():
                break
            edge_mask[endpoints] = False
        return edge_mask

    def contrast_stretch(self, image, low_pct=2.0, high_pct=98.0):
        """저대비 영역을 늘려 약한 에지를 강화"""
        low_pct = float(low_pct)
        high_pct = float(high_pct)
        if high_pct <= low_pct:
            return image

        low_val, high_val = np.percentile(image, [low_pct, high_pct])
        if high_val <= low_val:
            return image

        stretched = (image - low_val) * (255.0 / (high_val - low_val))
        return np.clip(stretched, 0, 255).astype(np.float32)
    
    def compute_gradient(self, image):
        """Sobel 필터로 그래디언트 계산"""
        # x, y 방향 그래디언트 계산
        grad_x = self.apply_convolution(image, self.sobel_x)
        grad_y = self.apply_convolution(image, self.sobel_y)
        
        # 그래디언트 크기와 방향 계산
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        return magnitude, direction, grad_x, grad_y
    
    def non_maximum_suppression(self, magnitude, direction, relax=1.0):
        """Non-Maximum Suppression으로 얇은 에지 생성"""
        magnitude = magnitude.astype(np.float32, copy=False)
        suppressed = np.zeros_like(magnitude, dtype=np.float32)
        relax = float(relax)
        if relax <= 0:
            relax = 1.0
        if relax > 1.0:
            relax = 1.0

        # 각도를 0, 45, 90, 135도로 양자화
        angle = direction * 180.0 / np.pi
        angle[angle < 0] += 180

        padded = np.pad(magnitude, ((1, 1), (1, 1)), mode="constant")
        center = padded[1:-1, 1:-1]
        right = padded[1:-1, 2:]
        left = padded[1:-1, :-2]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        up_left = padded[:-2, :-2]
        up_right = padded[:-2, 2:]
        down_left = padded[2:, :-2]
        down_right = padded[2:, 2:]

        mask_0 = (angle < 22.5) | (angle >= 157.5)
        mask_45 = (angle >= 22.5) & (angle < 67.5)
        mask_90 = (angle >= 67.5) & (angle < 112.5)
        mask_135 = (angle >= 112.5) & (angle < 157.5)

        keep_0 = mask_0 & (center >= right * relax) & (center >= left * relax)
        keep_45 = mask_45 & (center >= up_right * relax) & (center >= down_left * relax)
        keep_90 = mask_90 & (center >= up * relax) & (center >= down * relax)
        keep_135 = mask_135 & (center >= up_left * relax) & (center >= down_right * relax)

        suppressed[keep_0 | keep_45 | keep_90 | keep_135] = center[
            keep_0 | keep_45 | keep_90 | keep_135
        ]

        suppressed[0, :] = 0
        suppressed[-1, :] = 0
        suppressed[:, 0] = 0
        suppressed[:, -1] = 0
        return suppressed
    
    def double_threshold(
        self,
        image,
        low_ratio=0.06,
        high_ratio=0.18,
        method="ratio",
        low_percentile=35.0,
        high_percentile=80.0,
        min_threshold=1.0,
        mad_low_k=1.5,
        mad_high_k=3.0,
    ):
        """이중 임계값 적용"""
        if method == "percentile":
            sample = image[image > 0]
            if sample.size == 0:
                high_threshold = image.max() * high_ratio
                low_threshold = image.max() * low_ratio
            else:
                high_threshold = np.percentile(sample, high_percentile)
                low_threshold = np.percentile(sample, low_percentile)
        elif method == "mad":
            sample = image[image > 0]
            if sample.size < 10:
                sample = image
            median = float(np.median(sample))
            mad = float(np.median(np.abs(sample - median)))
            sigma = 1.4826 * mad
            high_threshold = median + mad_high_k * sigma
            low_threshold = median + mad_low_k * sigma
        else:
            high_threshold = image.max() * high_ratio
            low_threshold = image.max() * low_ratio

        high_threshold = max(float(high_threshold), min_threshold)
        low_threshold = max(float(low_threshold), min_threshold)
        if low_threshold > high_threshold:
            low_threshold = high_threshold
        
        strong = 255
        weak = 75
        
        result = np.zeros_like(image)
        
        strong_i, strong_j = np.where(image >= high_threshold)
        weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))
        
        result[strong_i, strong_j] = strong
        result[weak_i, weak_j] = weak
        
        return result, weak, strong
    
    def edge_tracking(self, image, weak, strong):
        """에지 추적 (Hysteresis)"""
        h, w = image.shape
        result = image.copy()
        strong_mask = result == strong
        weak_mask = result == weak

        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        stack = deque(zip(*np.where(strong_mask)))
        while stack:
            i, j = stack.pop()
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and weak_mask[ni, nj]:
                    weak_mask[ni, nj] = False
                    result[ni, nj] = strong
                    stack.append((ni, nj))

        result[weak_mask] = 0
        return result

    def dilate_binary(self, binary, radius=1):
        """이진 이미지 팽창 (연결 보강용)"""
        if radius <= 0:
            return binary
        size = radius * 2 + 1
        padded = np.pad(binary, radius, mode="constant", constant_values=False)
        windows = sliding_window_view(padded, (size, size))
        return np.any(windows, axis=(-2, -1))

    def erode_binary(self, binary, radius=1):
        """이진 이미지 침식 (노이즈 정리/클로징용)"""
        if radius <= 0:
            return binary
        size = radius * 2 + 1
        padded = np.pad(binary, radius, mode="constant", constant_values=False)
        windows = sliding_window_view(padded, (size, size))
        return np.all(windows, axis=(-2, -1))

    def thin_edges_zhang_suen(self, binary, max_iter=20):
        """Zhang-Suen 알고리즘으로 1픽셀 두께로 박리"""
        img = binary.astype(np.uint8, copy=True)
        if img.size == 0:
            return binary

        for _ in range(max(int(max_iter), 1)):
            changed = False

            P2 = img[:-2, 1:-1] == 1
            P3 = img[:-2, 2:] == 1
            P4 = img[1:-1, 2:] == 1
            P5 = img[2:, 2:] == 1
            P6 = img[2:, 1:-1] == 1
            P7 = img[2:, :-2] == 1
            P8 = img[1:-1, :-2] == 1
            P9 = img[:-2, :-2] == 1

            B = (
                P2.astype(np.uint8)
                + P3.astype(np.uint8)
                + P4.astype(np.uint8)
                + P5.astype(np.uint8)
                + P6.astype(np.uint8)
                + P7.astype(np.uint8)
                + P8.astype(np.uint8)
                + P9.astype(np.uint8)
            )
            A = (
                (~P2 & P3).astype(np.uint8)
                + (~P3 & P4).astype(np.uint8)
                + (~P4 & P5).astype(np.uint8)
                + (~P5 & P6).astype(np.uint8)
                + (~P6 & P7).astype(np.uint8)
                + (~P7 & P8).astype(np.uint8)
                + (~P8 & P9).astype(np.uint8)
                + (~P9 & P2).astype(np.uint8)
            )

            m1 = (A == 1) & (B >= 2) & (B <= 6) & ~(P2 & P4 & P6) & ~(P4 & P6 & P8)
            if np.any(m1):
                img[1:-1, 1:-1][m1] = 0
                changed = True

            P2 = img[:-2, 1:-1] == 1
            P3 = img[:-2, 2:] == 1
            P4 = img[1:-1, 2:] == 1
            P5 = img[2:, 2:] == 1
            P6 = img[2:, 1:-1] == 1
            P7 = img[2:, :-2] == 1
            P8 = img[1:-1, :-2] == 1
            P9 = img[:-2, :-2] == 1

            B = (
                P2.astype(np.uint8)
                + P3.astype(np.uint8)
                + P4.astype(np.uint8)
                + P5.astype(np.uint8)
                + P6.astype(np.uint8)
                + P7.astype(np.uint8)
                + P8.astype(np.uint8)
                + P9.astype(np.uint8)
            )
            A = (
                (~P2 & P3).astype(np.uint8)
                + (~P3 & P4).astype(np.uint8)
                + (~P4 & P5).astype(np.uint8)
                + (~P5 & P6).astype(np.uint8)
                + (~P6 & P7).astype(np.uint8)
                + (~P7 & P8).astype(np.uint8)
                + (~P8 & P9).astype(np.uint8)
                + (~P9 & P2).astype(np.uint8)
            )

            m2 = (A == 1) & (B >= 2) & (B <= 6) & ~(P2 & P4 & P8) & ~(P2 & P6 & P8)
            if np.any(m2):
                img[1:-1, 1:-1][m2] = 0
                changed = True

            if not changed:
                break

        return img.astype(bool)

    def refine_edge_peaks(self, edge_mask, magnitude, direction, fill_radius=1):
        """완화된 에지를 엄격한 피크에 맞춰 얇게 보정"""
        strict_peaks = self.non_maximum_suppression(magnitude, direction, relax=1.0) > 0
        refined = strict_peaks & edge_mask

        if fill_radius <= 0:
            return refined

        near_refined = self.dilate_binary(refined, fill_radius)
        uncovered = edge_mask & ~near_refined
        if not np.any(uncovered):
            return refined

        padded = np.pad(magnitude, 1, mode="edge")
        windows = sliding_window_view(padded, (3, 3))
        local_max = magnitude >= windows.max(axis=(-2, -1))
        return refined | (uncovered & local_max)

    def filter_edge_polarity(
        self,
        edge_mask,
        image,
        grad_x,
        grad_y,
        min_diff=1.0,
        min_support=50,
        drop_margin=0.0,
    ):
        """그라디언트 방향의 밝기 변화로 내부 에지를 제거"""
        if not np.any(edge_mask):
            return edge_mask

        h, w = image.shape
        ys, xs = np.where(edge_mask)
        step_x = np.sign(grad_x).astype(np.int32)
        step_y = np.sign(grad_y).astype(np.int32)

        y_plus = np.clip(ys + step_y[ys, xs], 0, h - 1)
        x_plus = np.clip(xs + step_x[ys, xs], 0, w - 1)
        y_minus = np.clip(ys - step_y[ys, xs], 0, h - 1)
        x_minus = np.clip(xs - step_x[ys, xs], 0, w - 1)

        diff = image[y_plus, x_plus] - image[y_minus, x_minus]
        strong = np.abs(diff) >= min_diff
        if int(strong.sum()) < int(min_support):
            return edge_mask

        median_diff = float(np.median(diff[strong]))
        sign = 1.0 if median_diff >= 0 else -1.0
        keep = diff * sign >= -float(drop_margin)

        filtered = edge_mask.copy()
        filtered[ys, xs] = keep
        return filtered

    def otsu_threshold(self, image):
        """Otsu 임계값 계산"""
        values = image.astype(np.uint8, copy=False).ravel()
        hist = np.bincount(values, minlength=256).astype(np.float64)
        total = values.size
        if total == 0:
            return 0

        sum_total = float(np.dot(np.arange(256), hist))
        sum_b = 0.0
        w_b = 0.0
        max_var = -1.0
        threshold = 0

        for t in range(256):
            w_b += hist[t]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * (m_b - m_f) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = t
        return int(threshold)

    def estimate_object_mask(self, image, object_is_dark=None):
        """Otsu 기반 마스크 추정"""
        threshold = self.otsu_threshold(image)
        low_mask = image <= threshold
        high_mask = ~low_mask

        if object_is_dark is None:
            low_mean = float(image[low_mask].mean()) if np.any(low_mask) else 0.0
            high_mean = float(image[high_mask].mean()) if np.any(high_mask) else 0.0
            object_is_dark = low_mean < high_mean

        return low_mask if object_is_dark else high_mask

    def boundary_band_filter(
        self,
        edge_mask,
        image,
        band_radius=2,
        mask_min_area=0.05,
        mask_max_area=0.95,
        object_is_dark=True,
        use_mask_blur=True,
        mask_blur_kernel_size=5,
        mask_blur_sigma=1.0,
        mask_close_radius=1,
    ):
        """경계 대역만 남겨 내부 곡선 제거"""
        mask_source = image
        if use_mask_blur:
            mask_source = self.apply_gaussian_blur(image, mask_blur_kernel_size, mask_blur_sigma)

        mask = self.estimate_object_mask(mask_source, object_is_dark)
        area_ratio = float(mask.mean())
        if area_ratio < mask_min_area or area_ratio > mask_max_area:
            return edge_mask

        if mask_close_radius > 0:
            mask = self.erode_binary(self.dilate_binary(mask, mask_close_radius), mask_close_radius)

        boundary = np.zeros_like(mask, dtype=bool)
        padded = np.pad(mask, 1, mode="edge")
        center = padded[1:-1, 1:-1]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = padded[1 + dy : 1 + dy + center.shape[0], 1 + dx : 1 + dx + center.shape[1]]
                boundary |= neighbor != center

        band = self.dilate_binary(boundary, band_radius)
        return edge_mask & band
    
    def detect_edges_array(
        self,
        image,
        use_nms=True,
        use_hysteresis=True,
        use_median_filter=True,
        median_kernel_size=3,
        use_blur=True,
        blur_kernel_size=3,
        blur_sigma=0.7,
        use_contrast_stretch=False,
        contrast_low_pct=2.0,
        contrast_high_pct=98.0,
        magnitude_gamma=1.0,
        nms_relax=0.95,
        low_ratio=0.04,
        high_ratio=0.12,
        auto_threshold=True,
        contrast_ref=80.0,
        min_threshold_scale=0.5,
        threshold_method="ratio",
        low_percentile=35.0,
        high_percentile=80.0,
        min_threshold=1.0,
        mad_low_k=1.5,
        mad_high_k=3.0,
        use_soft_linking=False,
        soft_low_ratio=0.03,
        soft_high_ratio=0.1,
        link_radius=2,
        soft_threshold_method=None,
        soft_low_percentile=None,
        soft_high_percentile=None,
        soft_mad_low_k=None,
        soft_mad_high_k=None,
        use_closing=False,
        closing_radius=1,
        closing_iterations=1,
        use_peak_refine=False,
        peak_fill_radius=1,
        use_polarity_filter=True,
        polarity_min_diff=1.0,
        polarity_min_support=50,
        polarity_drop_margin=0.0,
        use_boundary_band_filter=True,
        boundary_band_radius=2,
        mask_min_area=0.05,
        mask_max_area=0.95,
        object_is_dark=True,
        use_mask_blur=True,
        mask_blur_kernel_size=5,
        mask_blur_sigma=1.0,
        mask_close_radius=1,
        use_edge_smooth=False,
        edge_smooth_radius=1,
        edge_smooth_iters=1,
        use_thinning=True,
        thinning_max_iter=15,
        spur_prune_iters=0,
    ):
        """전체 에지 검출 파이프라인"""
        # 1. 입력 배열 준비
        image = np.array(image, dtype=np.float32, copy=False)
        original = image.copy()

        # 1-1. 메디안 필터로 약한 노이즈 제거
        if use_median_filter:
            image = self.apply_median_filter(image, median_kernel_size)

        # 1-2. 블러로 노이즈 완화
        if use_blur:
            image = self.apply_gaussian_blur(image, blur_kernel_size, blur_sigma)

        # 1-3. 저대비 보정
        if use_contrast_stretch:
            image = self.contrast_stretch(image, contrast_low_pct, contrast_high_pct)
        
        # 2. Sobel 필터 적용
        magnitude, direction, grad_x, grad_y = self.compute_gradient(image)
        if magnitude_gamma != 1.0:
            magnitude = np.power(magnitude, magnitude_gamma)
        
        # 3. Non-Maximum Suppression (얇은 에지)
        if use_nms:
            edges = self.non_maximum_suppression(magnitude, direction, relax=nms_relax)
        else:
            edges = magnitude
        
        # 4. 이중 임계값 및 에지 추적 (연결된 에지)
        low_ratio_adj = low_ratio
        high_ratio_adj = high_ratio
        soft_low_adj = soft_low_ratio
        soft_high_adj = soft_high_ratio
        if auto_threshold:
            p10, p90 = np.percentile(image, [10, 90])
            contrast = max(float(p90 - p10), 1.0)
            scale = min(1.0, contrast / float(contrast_ref))
            scale = max(scale, float(min_threshold_scale))
            low_ratio_adj *= scale
            high_ratio_adj *= scale
            soft_low_adj *= scale
            soft_high_adj *= scale

        if use_hysteresis:
            edges_threshold, weak, strong = self.double_threshold(
                edges,
                low_ratio=low_ratio_adj,
                high_ratio=high_ratio_adj,
                method=threshold_method,
                low_percentile=low_percentile,
                high_percentile=high_percentile,
                min_threshold=min_threshold,
                mad_low_k=mad_low_k,
                mad_high_k=mad_high_k,
            )
            edges_strong = self.edge_tracking(edges_threshold, weak, strong)

            if use_soft_linking:
                soft_method = soft_threshold_method or threshold_method
                soft_low_pct = low_percentile if soft_low_percentile is None else soft_low_percentile
                soft_high_pct = high_percentile if soft_high_percentile is None else soft_high_percentile
                soft_mad_low = mad_low_k if soft_mad_low_k is None else soft_mad_low_k
                soft_mad_high = mad_high_k if soft_mad_high_k is None else soft_mad_high_k
                edges_threshold_soft, weak_soft, strong_soft = self.double_threshold(
                    edges,
                    low_ratio=soft_low_adj,
                    high_ratio=soft_high_adj,
                    method=soft_method,
                    low_percentile=soft_low_pct,
                    high_percentile=soft_high_pct,
                    min_threshold=min_threshold,
                    mad_low_k=soft_mad_low,
                    mad_high_k=soft_mad_high,
                )
                edges_soft = self.edge_tracking(edges_threshold_soft, weak_soft, strong_soft)
                strong_mask = edges_strong > 0
                soft_mask = edges_soft > 0
                if link_radius > 0:
                    near_strong = self.dilate_binary(strong_mask, link_radius)
                    combined = strong_mask | (soft_mask & near_strong)
                else:
                    combined = strong_mask | soft_mask
                edges_final = np.where(combined, 255, 0)
            else:
                edges_final = edges_strong
        else:
            # 단순 임계값
            threshold = edges.max() * 0.15
            edges_final = np.where(edges > threshold, 255, 0)
        
        if use_closing and use_hysteresis:
            edge_mask = edges_final > 0
            for _ in range(max(int(closing_iterations), 1)):
                edge_mask = self.erode_binary(self.dilate_binary(edge_mask, closing_radius), closing_radius)
            edges_final = np.where(edge_mask, 255, 0)

        if use_peak_refine:
            edge_mask = edges_final > 0
            edge_mask = self.refine_edge_peaks(edge_mask, magnitude, direction, peak_fill_radius)
            edges_final = np.where(edge_mask, 255, 0)

        if use_polarity_filter:
            edge_mask = edges_final > 0
            edge_mask = self.filter_edge_polarity(
                edge_mask,
                original,
                grad_x,
                grad_y,
                min_diff=polarity_min_diff,
                min_support=polarity_min_support,
                drop_margin=polarity_drop_margin,
            )
            edges_final = np.where(edge_mask, 255, 0)

        if use_boundary_band_filter:
            edge_mask = edges_final > 0
            edge_mask = self.boundary_band_filter(
                edge_mask,
                original,
                band_radius=boundary_band_radius,
                mask_min_area=mask_min_area,
                mask_max_area=mask_max_area,
                object_is_dark=object_is_dark,
                use_mask_blur=use_mask_blur,
                mask_blur_kernel_size=mask_blur_kernel_size,
                mask_blur_sigma=mask_blur_sigma,
                mask_close_radius=mask_close_radius,
            )
            edges_final = np.where(edge_mask, 255, 0)

        if use_edge_smooth and edge_smooth_radius > 0:
            edge_mask = edges_final > 0
            for _ in range(max(int(edge_smooth_iters), 1)):
                edge_mask = self.erode_binary(self.dilate_binary(edge_mask, edge_smooth_radius), edge_smooth_radius)
            edges_final = np.where(edge_mask, 255, 0)

        if use_thinning:
            edge_mask = edges_final > 0
            edge_mask = self.thin_edges_zhang_suen(edge_mask, thinning_max_iter)
            edges_final = np.where(edge_mask, 255, 0)

        if spur_prune_iters > 0:
            edge_mask = edges_final > 0
            edge_mask = self.prune_spurs(edge_mask, spur_prune_iters)
            edges_final = np.where(edge_mask, 255, 0)

        edges_final = edges_final.astype(np.uint8)
        return {
            'original': original,
            'magnitude': magnitude,
            'grad_x': grad_x,
            'grad_y': grad_y,
            'edges': edges_final
        }

    def detect_edges(
        self,
        image_path,
        **kwargs,
    ):
        """파일 경로를 받아 에지 검출"""
        image = self.load_image(image_path)
        return self.detect_edges_array(image, **kwargs)

def _make_overlay_image(original_gray, edges):
    """그레이스케일 원본 위에 초록색 에지 dot를 표시"""
    overlay = np.stack([original_gray] * 3, axis=-1)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    edge_mask = edges > 0
    overlay[edge_mask] = [0, 255, 0]
    return overlay, edge_mask


def process_image_file(image_path, output_dir, detector, settings=None):
    """단일 이미지 처리 및 결과 저장"""
    settings = settings or {}
    results = detector.detect_edges(
        image_path,
        use_nms=True,
        use_hysteresis=True,
        **settings,
    )
    overlay, edge_mask = _make_overlay_image(results['original'], results['edges'])

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    overlay_path = os.path.join(output_dir, f"{base_name}_edges_green.png")
    coords_path = os.path.join(output_dir, f"{base_name}_edge_coords.txt")

    Image.fromarray(overlay).save(overlay_path)

    coords = np.column_stack(np.where(edge_mask))
    with open(coords_path, "w", encoding="utf-8") as handle:
        handle.write("# x,y\n")
        for y, x in coords:
            handle.write(f"{x},{y}\n")

    return overlay_path, coords_path


class EdgeBatchGUI:
    """오프라인 GUI 배치 처리기"""

    def __init__(self, root):
        self.root = root
        self.root.title("Sobel Edge Batch Processor (ver20)")
        self.root.geometry("1920x1080")

        self.detector = SobelEdgeDetector()
        self.selected_files = []
        self.roi_map = {}
        self.roi_cache = self._load_roi_cache()
        self.max_files = 500
        self.output_root = os.path.abspath("outputs")

        self._message_queue = queue.Queue()
        self._worker_thread = None
        self._auto_thread = None
        self.param_vars = self._init_param_vars()
        self.auto_mode = tk.StringVar(value="Fast")
        self.score_display_mode = tk.StringVar(value="scaled")
        self.log_text = None
        self.score_graph_label = None
        self.best_graph_label = None
        self.metric_graph_label = None
        self.detail_graph_label = None
        self._auto_scores = []
        self._auto_best_scores = []
        self._auto_best_time_series = []
        self._auto_cont_scores = []
        self._auto_band_scores = []
        self._auto_penalty_scores = []
        self._auto_wrinkle_scores = []
        self._auto_endpoint_scores = []
        self._auto_branch_scores = []
        self._auto_image_grayscale = None
        self._auto_image_low_quality = None
        self.score_display_mode.trace_add("write", lambda *_: self._refresh_auto_graphs())
        self._auto_start_time = None
        self._auto_last_best_time = None
        self._last_auto_best_score = None
        self._auto_pause_event = threading.Event()
        self._auto_stop_event = threading.Event()
        self.pause_button = None
        self.stop_button = None

        self._build_ui()

    def _init_param_vars(self):
        vars_map = {}
        for key, value in {**PARAM_DEFAULTS, **AUTO_DEFAULTS}.items():
            if isinstance(value, bool):
                vars_map[key] = tk.BooleanVar(value=value)
            elif isinstance(value, int):
                vars_map[key] = tk.IntVar(value=value)
            else:
                vars_map[key] = tk.DoubleVar(value=value)
        return vars_map

    def _build_ui(self):
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        v_scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        main_frame = ttk.Frame(canvas, padding=12)
        canvas_window = canvas.create_window((0, 0), window=main_frame, anchor="nw")
        self._scroll_canvas = canvas
        self._scroll_window = canvas_window

        def _on_frame_configure(event):
            bbox = canvas.bbox("all")
            if bbox:
                # Extra bottom margin (~graph height) so full scroll reaches the end
                canvas.configure(scrollregion=(bbox[0], bbox[1], bbox[2], bbox[3] + 300))

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        main_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            delta = -1 * (event.delta // 120) if event.delta else 0
            if delta:
                canvas.yview_scroll(delta, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda event: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda event: canvas.yview_scroll(1, "units"))

        header = ttk.Label(
            main_frame,
            text="Select up to 500 images and process sequentially.",
            font=("Arial", 12, "bold"),
        )
        header.pack(anchor="w")

        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=(10, 6))

        ttk.Label(output_frame, text="Output folder:").pack(side=tk.LEFT)
        self.output_label = ttk.Label(output_frame, text=self.output_root)
        self.output_label.pack(side=tk.LEFT, padx=(6, 12))
        ttk.Button(output_frame, text="Change", command=self._choose_output_dir).pack(side=tk.RIGHT)

        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.file_listbox = tk.Listbox(list_frame, height=8)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        param_frame = ttk.LabelFrame(main_frame, text="Parameter Settings")
        param_frame.pack(fill=tk.X, pady=(8, 6))

        ttk.Label(param_frame, text="NMS relax").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0.85,
            to=1.0,
            increment=0.01,
            textvariable=self.param_vars["nms_relax"],
            width=6,
        ).grid(row=0, column=1, sticky="w", padx=(4, 12))

        ttk.Label(param_frame, text="Low ratio").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0.01,
            to=0.2,
            increment=0.01,
            textvariable=self.param_vars["low_ratio"],
            width=6,
        ).grid(row=0, column=3, sticky="w", padx=(4, 12))

        ttk.Label(param_frame, text="High ratio").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0.05,
            to=0.4,
            increment=0.01,
            textvariable=self.param_vars["high_ratio"],
            width=6,
        ).grid(row=0, column=5, sticky="w", padx=(4, 0))

        ttk.Label(param_frame, text="Blur sigma").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0.3,
            to=3.0,
            increment=0.1,
            textvariable=self.param_vars["blur_sigma"],
            width=6,
        ).grid(row=1, column=1, sticky="w", padx=(4, 12))

        ttk.Label(param_frame, text="Blur kernel").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=3,
            to=9,
            increment=2,
            textvariable=self.param_vars["blur_kernel_size"],
            width=6,
        ).grid(row=1, column=3, sticky="w", padx=(4, 12))

        ttk.Checkbutton(
            param_frame,
            text="Auto threshold",
            variable=self.param_vars["auto_threshold"],
        ).grid(row=1, column=4, sticky="w", padx=(4, 0))

        ttk.Checkbutton(
            param_frame,
            text="Use blur",
            variable=self.param_vars["use_blur"],
        ).grid(row=1, column=5, sticky="w", padx=(6, 0))

        ttk.Checkbutton(
            param_frame,
            text="Median filter",
            variable=self.param_vars["use_median_filter"],
        ).grid(row=1, column=6, sticky="w", padx=(6, 0))

        ttk.Label(param_frame, text="Median kernel").grid(row=1, column=7, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=3,
            to=7,
            increment=2,
            textvariable=self.param_vars["median_kernel_size"],
            width=5,
        ).grid(row=1, column=8, sticky="w")

        ttk.Label(param_frame, text="Contrast ref").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=20,
            to=160,
            increment=5,
            textvariable=self.param_vars["contrast_ref"],
            width=6,
        ).grid(row=2, column=1, sticky="w", padx=(4, 12))

        ttk.Label(param_frame, text="Min scale").grid(row=2, column=2, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0.2,
            to=1.0,
            increment=0.1,
            textvariable=self.param_vars["min_threshold_scale"],
            width=6,
        ).grid(row=2, column=3, sticky="w", padx=(4, 12))

        ttk.Checkbutton(
            param_frame,
            text="Thinning (1px)",
            variable=self.param_vars["use_thinning"],
        ).grid(row=2, column=4, sticky="w", padx=(4, 0))

        ttk.Label(param_frame, text="Thinning iters").grid(row=2, column=5, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=5,
            to=30,
            increment=1,
            textvariable=self.param_vars["thinning_max_iter"],
            width=5,
        ).grid(row=2, column=6, sticky="w")

        ttk.Label(param_frame, text="Magnitude gamma").grid(row=3, column=6, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0.6,
            to=2.0,
            increment=0.1,
            textvariable=self.param_vars["magnitude_gamma"],
            width=5,
        ).grid(row=3, column=7, sticky="w")

        ttk.Checkbutton(
            param_frame,
            text="Boundary band",
            variable=self.param_vars["use_boundary_band_filter"],
        ).grid(row=3, column=0, sticky="w")

        ttk.Label(param_frame, text="Band radius").grid(row=3, column=2, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=1,
            to=6,
            increment=1,
            textvariable=self.param_vars["boundary_band_radius"],
            width=6,
        ).grid(row=3, column=3, sticky="w", padx=(4, 12))

        ttk.Checkbutton(
            param_frame,
            text="Object dark",
            variable=self.param_vars["object_is_dark"],
        ).grid(row=3, column=4, sticky="w", padx=(4, 0))

        ttk.Checkbutton(
            param_frame,
            text="Contrast stretch",
            variable=self.param_vars["use_contrast_stretch"],
        ).grid(row=3, column=5, sticky="w", padx=(6, 0))

        ttk.Checkbutton(
            param_frame,
            text="Mask blur",
            variable=self.param_vars["use_mask_blur"],
        ).grid(row=4, column=0, sticky="w")

        ttk.Label(param_frame, text="Mask sigma").grid(row=4, column=2, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0.5,
            to=3.0,
            increment=0.1,
            textvariable=self.param_vars["mask_blur_sigma"],
            width=6,
        ).grid(row=4, column=3, sticky="w", padx=(4, 12))

        ttk.Label(param_frame, text="Mask close").grid(row=4, column=4, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0,
            to=3,
            increment=1,
            textvariable=self.param_vars["mask_close_radius"],
            width=6,
        ).grid(row=4, column=5, sticky="w", padx=(4, 0))

        ttk.Label(param_frame, text="Low pct").grid(row=4, column=6, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0.0,
            to=10.0,
            increment=0.5,
            textvariable=self.param_vars["contrast_low_pct"],
            width=5,
        ).grid(row=4, column=7, sticky="w")

        ttk.Label(param_frame, text="High pct").grid(row=5, column=6, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=90.0,
            to=100.0,
            increment=0.5,
            textvariable=self.param_vars["contrast_high_pct"],
            width=5,
        ).grid(row=5, column=7, sticky="w")

        ttk.Checkbutton(
            param_frame,
            text="Polarity filter",
            variable=self.param_vars["use_polarity_filter"],
        ).grid(row=5, column=0, sticky="w")

        ttk.Label(param_frame, text="Polarity margin").grid(row=5, column=2, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0.0,
            to=1.0,
            increment=0.1,
            textvariable=self.param_vars["polarity_drop_margin"],
            width=6,
        ).grid(row=5, column=3, sticky="w", padx=(4, 12))

        ttk.Checkbutton(
            param_frame,
            text="Soft linking",
            variable=self.param_vars["use_soft_linking"],
        ).grid(row=6, column=0, sticky="w")
        ttk.Label(param_frame, text="Soft low").grid(row=6, column=2, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0.01,
            to=0.2,
            increment=0.01,
            textvariable=self.param_vars["soft_low_ratio"],
            width=6,
        ).grid(row=6, column=3, sticky="w", padx=(4, 12))
        ttk.Label(param_frame, text="Soft high").grid(row=6, column=4, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0.05,
            to=0.4,
            increment=0.01,
            textvariable=self.param_vars["soft_high_ratio"],
            width=6,
        ).grid(row=6, column=5, sticky="w", padx=(4, 12))
        ttk.Label(param_frame, text="Link radius").grid(row=6, column=6, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0,
            to=6,
            increment=1,
            textvariable=self.param_vars["link_radius"],
            width=5,
        ).grid(row=6, column=7, sticky="w")

        ttk.Checkbutton(
            param_frame,
            text="Closing",
            variable=self.param_vars["use_closing"],
        ).grid(row=7, column=0, sticky="w")
        ttk.Label(param_frame, text="Close radius").grid(row=7, column=2, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0,
            to=4,
            increment=1,
            textvariable=self.param_vars["closing_radius"],
            width=6,
        ).grid(row=7, column=3, sticky="w", padx=(4, 12))
        ttk.Label(param_frame, text="Close iters").grid(row=7, column=4, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=1,
            to=6,
            increment=1,
            textvariable=self.param_vars["closing_iterations"],
            width=6,
        ).grid(row=7, column=5, sticky="w", padx=(4, 12))

        ttk.Checkbutton(
            param_frame,
            text="Edge smooth",
            variable=self.param_vars["use_edge_smooth"],
        ).grid(row=8, column=0, sticky="w")
        ttk.Label(param_frame, text="Smooth radius").grid(row=8, column=2, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0,
            to=4,
            increment=1,
            textvariable=self.param_vars["edge_smooth_radius"],
            width=6,
        ).grid(row=8, column=3, sticky="w", padx=(4, 12))
        ttk.Label(param_frame, text="Smooth iters").grid(row=8, column=4, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=1,
            to=6,
            increment=1,
            textvariable=self.param_vars["edge_smooth_iters"],
            width=6,
        ).grid(row=8, column=5, sticky="w", padx=(4, 12))
        ttk.Label(param_frame, text="Spur prune").grid(row=8, column=6, sticky="w")
        ttk.Spinbox(
            param_frame,
            from_=0,
            to=6,
            increment=1,
            textvariable=self.param_vars["spur_prune_iters"],
            width=5,
        ).grid(row=8, column=7, sticky="w")

        auto_frame = ttk.LabelFrame(main_frame, text="Auto Search Range / Scoring")
        auto_frame.pack(fill=tk.X, pady=(6, 6))

        ttk.Label(auto_frame, text="NMS min").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.85,
            to=1.0,
            increment=0.01,
            textvariable=self.param_vars["auto_nms_min"],
            width=6,
        ).grid(row=0, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="NMS max").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.85,
            to=1.0,
            increment=0.01,
            textvariable=self.param_vars["auto_nms_max"],
            width=6,
        ).grid(row=0, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="NMS step").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.01,
            to=0.1,
            increment=0.01,
            textvariable=self.param_vars["auto_nms_step"],
            width=6,
        ).grid(row=0, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="High min").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.05,
            to=0.3,
            increment=0.01,
            textvariable=self.param_vars["auto_high_min"],
            width=6,
        ).grid(row=1, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="High max").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.05,
            to=0.3,
            increment=0.01,
            textvariable=self.param_vars["auto_high_max"],
            width=6,
        ).grid(row=1, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="High step").grid(row=1, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.01,
            to=0.1,
            increment=0.01,
            textvariable=self.param_vars["auto_high_step"],
            width=6,
        ).grid(row=1, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Low factor min").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.2,
            to=0.6,
            increment=0.05,
            textvariable=self.param_vars["auto_low_factor_min"],
            width=6,
        ).grid(row=2, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Low factor max").grid(row=2, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.2,
            to=0.8,
            increment=0.05,
            textvariable=self.param_vars["auto_low_factor_max"],
            width=6,
        ).grid(row=2, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Low factor step").grid(row=2, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.01,
            to=0.2,
            increment=0.01,
            textvariable=self.param_vars["auto_low_factor_step"],
            width=6,
        ).grid(row=2, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Band min").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0,
            to=6,
            increment=1,
            textvariable=self.param_vars["auto_band_min"],
            width=6,
        ).grid(row=3, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Band max").grid(row=3, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0,
            to=6,
            increment=1,
            textvariable=self.param_vars["auto_band_max"],
            width=6,
        ).grid(row=3, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Margin min").grid(row=3, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.param_vars["auto_margin_min"],
            width=6,
        ).grid(row=3, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Margin max").grid(row=4, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.param_vars["auto_margin_max"],
            width=6,
        ).grid(row=4, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Margin step").grid(row=4, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.01,
            to=0.5,
            increment=0.01,
            textvariable=self.param_vars["auto_margin_step"],
            width=6,
        ).grid(row=4, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Blur sigma min").grid(row=4, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.4,
            to=2.5,
            increment=0.1,
            textvariable=self.param_vars["auto_blur_sigma_min"],
            width=6,
        ).grid(row=4, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Blur sigma max").grid(row=5, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.4,
            to=3.0,
            increment=0.1,
            textvariable=self.param_vars["auto_blur_sigma_max"],
            width=6,
        ).grid(row=5, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Blur sigma step").grid(row=5, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.05,
            to=0.5,
            increment=0.05,
            textvariable=self.param_vars["auto_blur_sigma_step"],
            width=6,
        ).grid(row=5, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Blur kernel min").grid(row=5, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=3,
            to=9,
            increment=2,
            textvariable=self.param_vars["auto_blur_kernel_min"],
            width=6,
        ).grid(row=5, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Blur kernel max").grid(row=6, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=3,
            to=11,
            increment=2,
            textvariable=self.param_vars["auto_blur_kernel_max"],
            width=6,
        ).grid(row=6, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Blur kernel step").grid(row=6, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=2,
            to=4,
            increment=2,
            textvariable=self.param_vars["auto_blur_kernel_step"],
            width=6,
        ).grid(row=6, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Thinning min").grid(row=6, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=5,
            to=30,
            increment=1,
            textvariable=self.param_vars["auto_thinning_min"],
            width=6,
        ).grid(row=6, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Thinning max").grid(row=7, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=5,
            to=30,
            increment=1,
            textvariable=self.param_vars["auto_thinning_max"],
            width=6,
        ).grid(row=7, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Thinning step").grid(row=7, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=1,
            to=6,
            increment=1,
            textvariable=self.param_vars["auto_thinning_step"],
            width=6,
        ).grid(row=7, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Contrast ref min").grid(row=7, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=20,
            to=160,
            increment=5,
            textvariable=self.param_vars["auto_contrast_ref_min"],
            width=6,
        ).grid(row=7, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Contrast ref max").grid(row=8, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=20,
            to=180,
            increment=5,
            textvariable=self.param_vars["auto_contrast_ref_max"],
            width=6,
        ).grid(row=8, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Contrast ref step").grid(row=8, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=5,
            to=20,
            increment=5,
            textvariable=self.param_vars["auto_contrast_ref_step"],
            width=6,
        ).grid(row=8, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Min scale min").grid(row=8, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.2,
            to=1.0,
            increment=0.05,
            textvariable=self.param_vars["auto_min_scale_min"],
            width=6,
        ).grid(row=8, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Min scale max").grid(row=9, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.2,
            to=1.0,
            increment=0.05,
            textvariable=self.param_vars["auto_min_scale_max"],
            width=6,
        ).grid(row=9, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Min scale step").grid(row=9, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.05,
            to=0.3,
            increment=0.05,
            textvariable=self.param_vars["auto_min_scale_step"],
            width=6,
        ).grid(row=9, column=3, sticky="w", padx=(4, 12))

        ttk.Checkbutton(
            auto_frame,
            text="Parallel eval",
            variable=self.param_vars["auto_parallel"],
        ).grid(row=9, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=1,
            to=32,
            increment=1,
            textvariable=self.param_vars["auto_workers"],
            width=6,
        ).grid(row=9, column=5, sticky="w", padx=(4, 12))

        ttk.Button(
            auto_frame,
            text="Score Variables Help",
            command=self._show_score_help,
            width=18,
        ).grid(row=10, column=0, columnspan=2, sticky="w", padx=(0, 12))
        ttk.Button(
            auto_frame,
            text="Auto Search Params Help",
            command=self._show_auto_params_help,
            width=20,
        ).grid(row=10, column=2, columnspan=2, sticky="w", padx=(0, 12))
        ttk.Label(auto_frame, text="W continuity").grid(row=10, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=1.0,
            to=5.0,
            increment=0.2,
            textvariable=self.param_vars["weight_continuity"],
            width=6,
        ).grid(row=10, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W band fit").grid(row=11, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=1.0,
            to=4.0,
            increment=0.2,
            textvariable=self.param_vars["weight_band_fit"],
            width=6,
        ).grid(row=11, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W coverage").grid(row=11, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.5,
            to=3.0,
            increment=0.1,
            textvariable=self.param_vars["weight_coverage"],
            width=6,
        ).grid(row=11, column=3, sticky="w", padx=(4, 12))

        ttk.Spinbox(
            auto_frame,
            from_=0.5,
            to=3.0,
            increment=0.1,
            textvariable=self.param_vars["weight_gap"],
            width=6,
        ).grid(row=11, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W outside").grid(row=12, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.5,
            to=3.0,
            increment=0.1,
            textvariable=self.param_vars["weight_outside"],
            width=6,
        ).grid(row=12, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W thickness").grid(row=12, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.0,
            to=2.5,
            increment=0.1,
            textvariable=self.param_vars["weight_thickness"],
            width=6,
        ).grid(row=12, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W intrusion").grid(row=12, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.5,
            to=3.0,
            increment=0.1,
            textvariable=self.param_vars["weight_intrusion"],
            width=6,
        ).grid(row=12, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W low quality").grid(row=13, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.0,
            to=2.0,
            increment=0.1,
            textvariable=self.param_vars["weight_low_quality"],
            width=6,
        ).grid(row=13, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W endpoints").grid(row=13, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.5,
            to=4.0,
            increment=0.1,
            textvariable=self.param_vars["weight_endpoints"],
            width=6,
        ).grid(row=13, column=3, sticky="w", padx=(4, 12))

        ttk.Checkbutton(
            auto_frame,
            text="Early stop on stagnation (min)",
            variable=self.param_vars["early_stop_enabled"],
        ).grid(row=13, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=1,
            to=60,
            increment=1,
            textvariable=self.param_vars["early_stop_minutes"],
            width=6,
        ).grid(row=13, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W wrinkle").grid(row=14, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.5,
            to=4.0,
            increment=0.1,
            textvariable=self.param_vars["weight_wrinkle"],
            width=6,
        ).grid(row=14, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W branch").grid(row=14, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.5,
            to=3.0,
            increment=0.1,
            textvariable=self.param_vars["weight_branch"],
            width=6,
        ).grid(row=14, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Round early exit %").grid(row=14, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.05,
            to=0.5,
            increment=0.05,
            textvariable=self.param_vars["auto_round_early_exit_no_improve_frac"],
            width=6,
            format="%.2f",
        ).grid(row=14, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="No-improve rounds stop").grid(row=15, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=1,
            to=10,
            increment=1,
            textvariable=self.param_vars["auto_no_improve_rounds_stop"],
            width=6,
        ).grid(row=15, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Soft link prob").grid(row=16, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.param_vars["auto_soft_link_prob"],
            width=6,
        ).grid(row=16, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Soft high min").grid(row=16, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.04,
            to=0.3,
            increment=0.01,
            textvariable=self.param_vars["auto_soft_high_min"],
            width=6,
        ).grid(row=16, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Soft high max").grid(row=16, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.04,
            to=0.3,
            increment=0.01,
            textvariable=self.param_vars["auto_soft_high_max"],
            width=6,
        ).grid(row=16, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Soft high step").grid(row=17, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.005,
            to=0.05,
            increment=0.005,
            textvariable=self.param_vars["auto_soft_high_step"],
            width=6,
        ).grid(row=17, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Soft low factor min").grid(row=16, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.1,
            to=0.8,
            increment=0.05,
            textvariable=self.param_vars["auto_soft_low_factor_min"],
            width=6,
        ).grid(row=16, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Soft low factor max").grid(row=16, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.1,
            to=0.8,
            increment=0.05,
            textvariable=self.param_vars["auto_soft_low_factor_max"],
            width=6,
        ).grid(row=16, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Soft low factor step").grid(row=17, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.01,
            to=0.2,
            increment=0.01,
            textvariable=self.param_vars["auto_soft_low_factor_step"],
            width=6,
        ).grid(row=17, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Link radius min").grid(row=17, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=1,
            to=6,
            increment=1,
            textvariable=self.param_vars["auto_link_radius_min"],
            width=6,
        ).grid(row=17, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Link radius max").grid(row=17, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=1,
            to=6,
            increment=1,
            textvariable=self.param_vars["auto_link_radius_max"],
            width=6,
        ).grid(row=17, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Edge smooth prob").grid(row=18, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.param_vars["auto_use_edge_smooth_prob"],
            width=6,
        ).grid(row=17, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Smooth radius min").grid(row=18, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0,
            to=4,
            increment=1,
            textvariable=self.param_vars["auto_edge_smooth_radius_min"],
            width=6,
        ).grid(row=18, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Smooth radius max").grid(row=18, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0,
            to=4,
            increment=1,
            textvariable=self.param_vars["auto_edge_smooth_radius_max"],
            width=6,
        ).grid(row=18, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Smooth iters min").grid(row=18, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=1,
            to=6,
            increment=1,
            textvariable=self.param_vars["auto_edge_smooth_iters_min"],
            width=6,
        ).grid(row=18, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Smooth iters max").grid(row=18, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=1,
            to=6,
            increment=1,
            textvariable=self.param_vars["auto_edge_smooth_iters_max"],
            width=6,
        ).grid(row=18, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Spur prune min").grid(row=18, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0,
            to=6,
            increment=1,
            textvariable=self.param_vars["auto_spur_prune_min"],
            width=6,
        ).grid(row=18, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Spur prune max").grid(row=19, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0,
            to=6,
            increment=1,
            textvariable=self.param_vars["auto_spur_prune_max"],
            width=6,
        ).grid(row=19, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Closing prob").grid(row=19, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.param_vars["auto_use_closing_prob"],
            width=6,
        ).grid(row=19, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Closing radius min").grid(row=20, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0,
            to=4,
            increment=1,
            textvariable=self.param_vars["auto_closing_radius_min"],
            width=6,
        ).grid(row=20, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Closing radius max").grid(row=21, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0,
            to=4,
            increment=1,
            textvariable=self.param_vars["auto_closing_radius_max"],
            width=6,
        ).grid(row=21, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Closing iters min").grid(row=21, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=1,
            to=5,
            increment=1,
            textvariable=self.param_vars["auto_closing_iter_min"],
            width=6,
        ).grid(row=21, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Closing iters max").grid(row=21, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=1,
            to=5,
            increment=1,
            textvariable=self.param_vars["auto_closing_iter_max"],
            width=6,
        ).grid(row=21, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Gamma min").grid(row=22, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.6,
            to=2.0,
            increment=0.1,
            textvariable=self.param_vars["auto_magnitude_gamma_min"],
            width=6,
        ).grid(row=22, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Gamma max").grid(row=22, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.6,
            to=2.0,
            increment=0.1,
            textvariable=self.param_vars["auto_magnitude_gamma_max"],
            width=6,
        ).grid(row=22, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Gamma step").grid(row=22, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.05,
            to=0.3,
            increment=0.05,
            textvariable=self.param_vars["auto_magnitude_gamma_step"],
            width=6,
        ).grid(row=22, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Median kernel min").grid(row=23, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=3,
            to=7,
            increment=2,
            textvariable=self.param_vars["auto_median_kernel_min"],
            width=6,
        ).grid(row=22, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Median kernel max").grid(row=23, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=3,
            to=9,
            increment=2,
            textvariable=self.param_vars["auto_median_kernel_max"],
            width=6,
        ).grid(row=23, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Median kernel step").grid(row=23, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=2,
            to=4,
            increment=2,
            textvariable=self.param_vars["auto_median_kernel_step"],
            width=6,
        ).grid(row=23, column=5, sticky="w", padx=(4, 12))

        auto_btn_frame = ttk.Frame(auto_frame)
        auto_btn_frame.grid(row=24, column=0, columnspan=6, sticky="w", pady=(4, 0))
        ttk.Button(auto_btn_frame, text="Save Auto Config", command=self._save_auto_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(auto_btn_frame, text="Load Auto Config", command=self._load_auto_config).pack(side=tk.LEFT, padx=4)

        param_btn_frame = ttk.Frame(param_frame)
        param_btn_frame.grid(row=9, column=0, columnspan=8, sticky="w", pady=(4, 0))
        ttk.Button(param_btn_frame, text="Save Params", command=self._save_param_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(param_btn_frame, text="Load Params", command=self._load_param_config).pack(side=tk.LEFT, padx=4)
        self.auto_score_label = ttk.Label(param_frame, text="Auto best score: —")
        self.auto_score_label.grid(row=10, column=0, columnspan=8, sticky="w", pady=(2, 0))

        guide_frame = ttk.LabelFrame(main_frame, text="Tuning Guide")
        guide_frame.pack(fill=tk.X, pady=(6, 6))
        guide_text = tk.Text(guide_frame, height=9, wrap="word")
        guide_text.insert(
            "1.0",
            "- ROI: Draw a rectangle to focus auto-optimization on the true boundary area.\n"
            "- Auto Optimize: Fast (subset), Precise (all images, refine), Perfect (denser grid, ~5× time, coordinate descent).\n"
            "- NMS relax: Lower reduces gaps, too low can thicken edges or create inner curl.\n"
            "- Low/High ratio: Lower increases recall but can add intrusion/noise.\n"
            "- Boundary band radius: Smaller forces edges to stay on the outer boundary.\n"
            "- Polarity filter: Removes inner curves by checking edge brightness direction.\n"
            "- Thinning: Keeps a single-pixel contour; adjust NMS relax if gaps appear.\n"
            "- Auto threshold: Adapts to low-contrast images automatically.\n"
            "- Weights: Continuity and band fit are highest priority for stable outer edges.\n"
            "- Early stop ends optimization when best score stalls for N minutes.\n"
            "- Save Params stores detection settings; Save Auto Config stores search ranges/weights.\n",
        )
        guide_text.configure(state="disabled")
        guide_text.pack(fill=tk.X)
        graph_frame = ttk.LabelFrame(main_frame, text="Auto Optimization Progress")
        graph_frame.pack(fill=tk.X, pady=(6, 6))
        self.score_graph_label = ttk.Label(graph_frame)
        self.score_graph_label.pack(side=tk.LEFT, padx=6)
        self.best_graph_label = ttk.Label(graph_frame)
        self.best_graph_label.pack(side=tk.LEFT, padx=6)
        self.metric_graph_label = ttk.Label(graph_frame)
        self.metric_graph_label.pack(side=tk.LEFT, padx=6)
        self.detail_graph_label = ttk.Label(graph_frame)
        self.detail_graph_label.pack(side=tk.LEFT, padx=6)
        self.score_graph_label.bind("<Button-1>", lambda _e: self._open_graph_window("score"))
        self.best_graph_label.bind("<Button-1>", lambda _e: self._open_graph_window("best"))
        self.metric_graph_label.bind("<Button-1>", lambda _e: self._open_graph_window("metric"))
        self.detail_graph_label.bind("<Button-1>", lambda _e: self._open_graph_window("detail"))

        self.log_text = tk.Text(main_frame, height=6, wrap="word")
        self.log_text.pack(fill=tk.BOTH, pady=(6, 6))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 6))

        ttk.Button(button_frame, text="Add Files", command=self._add_files).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Clear List", command=self._clear_files).pack(side=tk.LEFT, padx=6)
        ttk.Button(button_frame, text="Set ROI", command=self._open_roi_editor).pack(side=tk.LEFT, padx=6)
        ttk.Button(button_frame, text="Clear ROI", command=self._clear_roi).pack(side=tk.LEFT, padx=6)
        ttk.Label(button_frame, text="Auto mode").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Combobox(
            button_frame,
            textvariable=self.auto_mode,
            values=["Fast", "Precise", "Perfect"],
            width=8,
            state="readonly",
        ).pack(side=tk.LEFT)
        self.auto_button = ttk.Button(button_frame, text="Auto Optimize", command=self._start_auto_optimize)
        self.auto_button.pack(side=tk.LEFT, padx=6)
        ttk.Label(button_frame, text="Score display").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Combobox(
            button_frame,
            textvariable=self.score_display_mode,
            values=["raw", "log10", "scaled"],
            width=7,
            state="readonly",
        ).pack(side=tk.LEFT)
        self.pause_button = ttk.Button(
            button_frame, text="Pause", command=self._toggle_auto_pause, state=tk.DISABLED
        )
        self.pause_button.pack(side=tk.LEFT, padx=4)
        self.stop_button = ttk.Button(
            button_frame, text="Stop", command=self._stop_auto_optimize, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=4)
        self.start_button = ttk.Button(button_frame, text="Start Processing", command=self._start_processing)
        self.start_button.pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Idle...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(anchor="w", pady=(6, 0))

        self.root.after(200, self._refresh_scroll_region)

    def _log(self, message):
        if not self.log_text:
            return
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def _refresh_scroll_region(self):
        canvas = getattr(self, "_scroll_canvas", None)
        if not canvas:
            return
        bbox = canvas.bbox("all")
        if bbox:
            canvas.configure(scrollregion=(bbox[0], bbox[1], bbox[2], bbox[3] + 300))

    def _load_roi_cache(self):
        if not os.path.exists(ROI_CACHE_PATH):
            return {}
        try:
            data = load_json_config(ROI_CACHE_PATH)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        cache = {}
        for key, value in data.items():
            if not isinstance(value, (list, tuple)) or len(value) != 4:
                continue
            try:
                cache[str(key)] = (int(value[0]), int(value[1]), int(value[2]), int(value[3]))
            except (ValueError, TypeError):
                continue
        return cache

    def _save_roi_cache(self):
        try:
            save_json_config(ROI_CACHE_PATH, self.roi_cache)
        except OSError:
            self._log("[ROI] Failed to save ROI cache.")

    def _format_file_label(self, path):
        tag = "[ROI] " if path in self.roi_map else "      "
        return f"{tag}{path}"

    def _refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        for path in self.selected_files:
            self.file_listbox.insert(tk.END, self._format_file_label(path))

    def _update_file_label(self, path):
        if path not in self.selected_files:
            return
        idx = self.selected_files.index(path)
        self.file_listbox.delete(idx)
        self.file_listbox.insert(idx, self._format_file_label(path))

    def _get_selected_file(self):
        selection = self.file_listbox.curselection()
        if not selection:
            return None
        return self.selected_files[selection[0]]

    def _open_roi_editor(self):
        selection = self.file_listbox.curselection()
        if not self.selected_files:
            messagebox.showinfo("Info", "Select a file to set ROI.")
            return
        idx = selection[0] if selection else 0
        idx = max(0, min(idx, len(self.selected_files) - 1))

        win = tk.Toplevel(self.root)
        win.title("Set ROI")
        canvas = tk.Canvas(win)
        canvas.pack()
        file_label = ttk.Label(win, text="")
        file_label.pack(pady=2)
        roi_label = ttk.Label(win, text="Drag to draw a rectangular ROI.")
        roi_label.pack(pady=4)

        state = {"idx": idx, "img": None, "scale": 1.0, "roi_rect": None, "path": None}
        max_w, max_h = 900, 520

        def load_index(new_idx):
            if new_idx < 0 or new_idx >= len(self.selected_files):
                roi_label.config(text="All ROI tasks are complete.")
                return
            state["idx"] = new_idx
            path = self.selected_files[new_idx]
            if not os.path.exists(path):
                roi_label.config(text="File not found.")
                return
            img = Image.open(path).convert("L")
            scale = min(max_w / img.width, max_h / img.height, 1.0)
            disp_w = int(img.width * scale)
            disp_h = int(img.height * scale)
            display = img.resize((disp_w, disp_h), resample=Image.BILINEAR)
            tk_img = ImageTk.PhotoImage(display)
            canvas.config(width=disp_w, height=disp_h)
            canvas.image = tk_img
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=tk_img)

            state["img"] = img
            state["scale"] = scale
            state["path"] = path
            state["roi_rect"] = None

            if path in self.roi_map:
                x1, y1, x2, y2 = self.roi_map[path]
                x1d, y1d = x1 * scale, y1 * scale
                x2d, y2d = x2 * scale, y2 * scale
                state["roi_rect"] = canvas.create_rectangle(x1d, y1d, x2d, y2d, outline="red", width=2)
                roi_label.config(text=f"ROI loaded: ({x1},{y1})-({x2},{y2})")
            else:
                roi_label.config(text="Drag to draw a rectangular ROI.")

            file_label.config(text=f"File {new_idx + 1}/{len(self.selected_files)}: {os.path.basename(path)}")
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(new_idx)
            self.file_listbox.see(new_idx)

        start = {"x": 0, "y": 0}

        def on_press(event):
            start["x"], start["y"] = event.x, event.y
            if state["roi_rect"]:
                canvas.delete(state["roi_rect"])
                state["roi_rect"] = None

        def on_drag(event):
            if state["roi_rect"]:
                canvas.delete(state["roi_rect"])
            state["roi_rect"] = canvas.create_rectangle(
                start["x"], start["y"], event.x, event.y, outline="red", width=2
            )

        def on_release(event):
            if not state["roi_rect"] or state["img"] is None:
                return
            img = state["img"]
            scale = state["scale"]
            path = state["path"]
            x1 = min(start["x"], event.x) / scale
            y1 = min(start["y"], event.y) / scale
            x2 = max(start["x"], event.x) / scale
            y2 = max(start["y"], event.y) / scale
            x1 = int(max(0, min(img.width - 1, x1)))
            y1 = int(max(0, min(img.height - 1, y1)))
            x2 = int(max(1, min(img.width, x2)))
            y2 = int(max(1, min(img.height, y2)))
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                roi_label.config(text="ROI is too small.")
                return
            self.roi_map[path] = (x1, y1, x2, y2)
            self.roi_cache[path] = (x1, y1, x2, y2)
            self.roi_cache[os.path.basename(path)] = (x1, y1, x2, y2)
            self._save_roi_cache()
            self._update_file_label(path)
            roi_label.config(text=f"ROI saved: ({x1},{y1})-({x2},{y2})")

            next_idx = state["idx"] + 1
            if next_idx < len(self.selected_files):
                load_index(next_idx)

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)

        def clear_roi():
            path = state["path"]
            if path and path in self.roi_map:
                del self.roi_map[path]
            if path and path in self.roi_cache:
                del self.roi_cache[path]
            if path:
                base = os.path.basename(path)
                if base in self.roi_cache:
                    del self.roi_cache[base]
            self._save_roi_cache()
            if state["roi_rect"]:
                canvas.delete(state["roi_rect"])
                state["roi_rect"] = None
            if path:
                self._update_file_label(path)
            roi_label.config(text="ROI cleared.")

        def prev_image():
            load_index(state["idx"] - 1)

        def next_image():
            load_index(state["idx"] + 1)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=6)
        ttk.Button(btn_frame, text="Prev", command=prev_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Next", command=next_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Clear ROI", command=clear_roi).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.LEFT, padx=4)

        load_index(idx)

    def _clear_roi(self):
        path = self._get_selected_file()
        if not path:
            messagebox.showinfo("Info", "Select a file to clear ROI.")
            return
        if path in self.roi_map:
            del self.roi_map[path]
            if path in self.roi_cache:
                del self.roi_cache[path]
            base = os.path.basename(path)
            if base in self.roi_cache:
                del self.roi_cache[base]
            self._save_roi_cache()
            self._update_file_label(path)
            self._log(f"[ROI] Cleared: {os.path.basename(path)}")
        else:
            messagebox.showinfo("Info", "No ROI is set.")

    def _toggle_auto_pause(self):
        if not self.pause_button:
            return
        if self._auto_pause_event.is_set():
            self._auto_pause_event.clear()
            self.pause_button.config(text="Pause")
            self._log("[AUTO] Resumed")
        else:
            self._auto_pause_event.set()
            self.pause_button.config(text="Resume")
            self._log("[AUTO] Paused")

    def _stop_auto_optimize(self):
        if not self.stop_button:
            return
        self._auto_stop_event.set()
        self._auto_pause_event.clear()
        if self.pause_button:
            self.pause_button.config(text="Pause")
        self._log("[AUTO] Stop requested")

    def _format_eta(self, seconds):
        seconds = max(0, int(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _score_to_display(self, value, mode=None):
        """Convert raw score to display value only; learning/optimization always uses raw score."""
        mode = mode or (self.score_display_mode.get() if self.score_display_mode else "scaled")
        v = float(value)
        if mode == "log10":
            return math.log10(max(v, 1e-30))
        if mode == "scaled":
            return v * SCORE_DISPLAY_SCALE
        return v

    def _format_score_for_display(self, value, mode=None):
        """Format score for UI/graph labels. Raw mode uses exponential notation for readability."""
        mode = mode or (self.score_display_mode.get() if self.score_display_mode else "scaled")
        v = float(value)
        if mode == "raw":
            return f"{v:.4e}"
        if mode == "log10":
            return f"{math.log10(max(v, 1e-30)):.4f}"
        if mode == "scaled":
            return f"{v * SCORE_DISPLAY_SCALE:.4f}"
        return f"{v:.4e}"

    def _select_auto_subset(self, files, max_files):
        if max_files >= len(files):
            return list(files)
        step = max(1, len(files) // max_files)
        return list(files)[::step][:max_files]

    def _compute_signature(self, image):
        arr = np.clip(image, 0, 255).astype(np.uint8)
        thumb = Image.fromarray(arr).resize((24, 24), resample=Image.BILINEAR)
        small = np.asarray(thumb, dtype=np.float32) / 255.0
        gx, gy = np.gradient(small)
        mag = np.sqrt(gx**2 + gy**2)
        sig = np.concatenate([small.flatten(), mag.flatten()])
        sig = (sig - sig.mean()) / (sig.std() + 1e-6)
        return sig

    def _cluster_signatures(self, signatures, k, rng):
        n = len(signatures)
        if n <= k:
            labels = np.arange(n, dtype=int)
            centers = signatures.copy()
            return labels, centers
        indices = rng.choice(n, k, replace=False)
        centers = signatures[indices].copy()
        for _ in range(8):
            distances = np.sum((signatures[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            for j in range(k):
                mask = labels == j
                if not np.any(mask):
                    centers[j] = signatures[rng.randint(0, n)]
                else:
                    centers[j] = signatures[mask].mean(axis=0)
        return labels, centers

    def _prepare_auto_data(self, files, settings, auto_config, roi_map, max_files):
        subset = self._select_auto_subset(files, max_files)
        band_min = min(auto_config["auto_band_min"], auto_config["auto_band_max"])
        band_max = max(auto_config["auto_band_min"], auto_config["auto_band_max"])
        band_radii = list(range(band_min, band_max + 1))
        if not band_radii:
            band_radii = [settings["boundary_band_radius"]]
        data = []
        for path in subset:
            image = self.detector.load_image(path)
            roi = roi_map.get(path)
            if roi:
                x1, y1, x2, y2 = roi
                image = image[y1:y2, x1:x2]

            mask_source = image
            if settings["use_mask_blur"]:
                mask_source = self.detector.apply_gaussian_blur(
                    image, settings["mask_blur_kernel_size"], settings["mask_blur_sigma"]
                )
            mask = self.detector.estimate_object_mask(mask_source, settings["object_is_dark"])
            if settings["mask_close_radius"] > 0:
                mask = self.detector.erode_binary(
                    self.detector.dilate_binary(mask, settings["mask_close_radius"]),
                    settings["mask_close_radius"],
                )
            boundary = self._compute_boundary(mask)
            bands = {}
            band_pixels = {}
            for radius in band_radii:
                if radius <= 0:
                    band = boundary.copy()
                else:
                    band = self.detector.dilate_binary(boundary, radius)
                bands[radius] = band
                band_pixels[radius] = int(band.sum())

            data.append(
                {
                    "path": path,
                    "image": image,
                    "mask": mask,
                    "boundary": boundary,
                    "bands": bands,
                    "band_pixels": band_pixels,
                    "weight": 1.0,
                }
            )
        if len(data) < 4:
            return {"coarse": data, "mid": data, "full": data}

        rng = np.random.RandomState(42)
        signatures = np.stack([self._compute_signature(item["image"]) for item in data], axis=0)
        k = min(max(2, int(np.sqrt(len(data)))), 12)
        labels, centers = self._cluster_signatures(signatures, k, rng)
        clusters = {i: [] for i in range(k)}
        for idx, label in enumerate(labels):
            clusters[int(label)].append(idx)

        coarse = []
        mid = []
        full = []
        for cluster_id, indices in clusters.items():
            if not indices:
                continue
            center = centers[cluster_id]
            dists = []
            for idx in indices:
                d = float(np.sum((signatures[idx] - center) ** 2))
                dists.append((d, idx))
            dists.sort(key=lambda x: x[0])
            rep_idx = dists[0][1]
            rep_item = dict(data[rep_idx])
            rep_item["weight"] = float(len(indices))
            coarse.append(rep_item)

            take = min(3, len(dists))
            for _, idx in dists[:take]:
                item = dict(data[idx])
                item["weight"] = float(len(indices)) / float(take)
                mid.append(item)

        for item in data:
            full.append(dict(item))

        return {"coarse": coarse, "mid": mid, "full": full}

    def _collect_values(self, keys):
        return {key: self.param_vars[key].get() for key in keys if key in self.param_vars}

    def _apply_values(self, values, defaults):
        merged = dict(defaults)
        for key, value in values.items():
            if key in defaults:
                merged[key] = value
        for key, value in merged.items():
            if key in self.param_vars:
                self.param_vars[key].set(value)

    def _get_param_settings(self):
        def get_float(key):
            return float(self.param_vars[key].get())

        def get_int(key):
            return int(self.param_vars[key].get())

        nms_relax = get_float("nms_relax")
        low_ratio = get_float("low_ratio")
        high_ratio = get_float("high_ratio")
        if high_ratio <= 0 or low_ratio <= 0:
            raise ValueError("Low/High ratio must be greater than 0.")
        if low_ratio >= high_ratio:
            low_ratio = high_ratio * 0.5

        settings = {
            "nms_relax": nms_relax,
            "low_ratio": low_ratio,
            "high_ratio": high_ratio,
            "use_median_filter": bool(self.param_vars["use_median_filter"].get()),
            "median_kernel_size": max(3, get_int("median_kernel_size") | 1),
            "use_blur": bool(self.param_vars["use_blur"].get()),
            "blur_sigma": get_float("blur_sigma"),
            "blur_kernel_size": max(3, get_int("blur_kernel_size")),
            "magnitude_gamma": get_float("magnitude_gamma"),
            "use_contrast_stretch": bool(self.param_vars["use_contrast_stretch"].get()),
            "contrast_low_pct": get_float("contrast_low_pct"),
            "contrast_high_pct": get_float("contrast_high_pct"),
            "auto_threshold": bool(self.param_vars["auto_threshold"].get()),
            "contrast_ref": get_float("contrast_ref"),
            "min_threshold_scale": get_float("min_threshold_scale"),
            "use_soft_linking": bool(self.param_vars["use_soft_linking"].get()),
            "soft_low_ratio": get_float("soft_low_ratio"),
            "soft_high_ratio": get_float("soft_high_ratio"),
            "link_radius": max(0, get_int("link_radius")),
            "use_closing": bool(self.param_vars["use_closing"].get()),
            "closing_radius": max(0, get_int("closing_radius")),
            "closing_iterations": max(1, get_int("closing_iterations")),
            "use_thinning": bool(self.param_vars["use_thinning"].get()),
            "thinning_max_iter": max(1, get_int("thinning_max_iter")),
            "use_edge_smooth": bool(self.param_vars["use_edge_smooth"].get()),
            "edge_smooth_radius": max(0, get_int("edge_smooth_radius")),
            "edge_smooth_iters": max(1, get_int("edge_smooth_iters")),
            "spur_prune_iters": max(0, get_int("spur_prune_iters")),
            "use_boundary_band_filter": bool(self.param_vars["use_boundary_band_filter"].get()),
            "boundary_band_radius": max(1, get_int("boundary_band_radius")),
            "object_is_dark": bool(self.param_vars["object_is_dark"].get()),
            "use_mask_blur": bool(self.param_vars["use_mask_blur"].get()),
            "mask_blur_sigma": get_float("mask_blur_sigma"),
            "mask_blur_kernel_size": max(3, get_int("mask_blur_kernel_size")),
            "mask_close_radius": max(0, get_int("mask_close_radius")),
            "use_polarity_filter": bool(self.param_vars["use_polarity_filter"].get()),
            "polarity_drop_margin": get_float("polarity_drop_margin"),
            "polarity_min_diff": 1.0,
            "polarity_min_support": 50,
            "mask_min_area": 0.05,
            "mask_max_area": 0.95,
            "use_peak_refine": False,
            "peak_fill_radius": 1,
        }
        return settings

    def _get_auto_config(self):
        config = self._collect_values(AUTO_KEYS)
        overrides = get_auto_profile_overrides(
            getattr(self, "_auto_image_grayscale", None),
            getattr(self, "_auto_image_low_quality", None),
        )
        return {**config, **overrides}

    def _save_param_config(self):
        config = {"params": self._collect_values(PARAM_KEYS)}
        path = filedialog.asksaveasfilename(
            title="Save Parameters",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        save_json_config(path, config)
        self._log(f"[PARAMS] Saved: {path}")

    def _load_param_config(self):
        path = filedialog.askopenfilename(
            title="Load Parameters",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        config = load_json_config(path)
        params = config.get("params", {})
        self._apply_values(params, PARAM_DEFAULTS)
        self._log(f"[PARAMS] Loaded: {path}")

    def _save_auto_config(self):
        config = {"auto": self._collect_values(AUTO_KEYS)}
        path = filedialog.asksaveasfilename(
            title="Save Auto Config",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        save_json_config(path, config)
        self._log(f"[AUTO] Config saved: {path}")

    def _load_auto_config(self):
        path = filedialog.askopenfilename(
            title="Load Auto Config",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        config = load_json_config(path)
        auto_values = config.get("auto", {})
        if "auto_low_factor" in auto_values and "auto_low_factor_min" not in auto_values:
            base = float(auto_values.get("auto_low_factor", AUTO_DEFAULTS["auto_low_factor_min"]))
            auto_values["auto_low_factor_min"] = max(0.1, base - 0.1)
            auto_values["auto_low_factor_max"] = min(0.9, base + 0.1)
            auto_values["auto_low_factor_step"] = 0.05
        self._apply_values(auto_values, AUTO_DEFAULTS)
        self._log(f"[AUTO] Config loaded: {path}")

    def _apply_settings(self, settings):
        for key, var in self.param_vars.items():
            if key in settings:
                var.set(settings[key])

    def _update_auto_score_label(self):
        if getattr(self, "auto_score_label", None) is None:
            return
        if self._last_auto_best_score is not None:
            txt = self._format_score_for_display(self._last_auto_best_score, None)
            self.auto_score_label.config(text=f"Auto best score: {txt}")
        else:
            self.auto_score_label.config(text="Auto best score: —")

    def _show_score_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Score Variables Help")
        help_window.geometry("700x600")
        help_window.transient(self.root)
        help_window.grab_set()

        text_frame = ttk.Frame(help_window, padding=12)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, padx=8, pady=8, font=("Arial", 10))
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        help_text = """Score Variables Explanation

The final score is computed from multiple quality metrics. Each metric is converted to a quality value (0-1) using sigmoid functions, then weighted and combined.

MAIN METRICS (High Weight):

1. Continuity (weight_continuity, default: 24.0)
   - Measures how continuous the detected edge is
   - Lower continuity value = better (fewer gaps)
   - Target: < 0.12 (sigmoid threshold)
   - Most important metric for edge quality

2. Band Fit (weight_band_fit, default: 12.0)
   - Ratio of edge pixels within the boundary band
   - Higher band_ratio = better (more pixels in band)
   - Target: > 0.85 (sigmoid threshold)
   - Second most important metric

SECONDARY METRICS (Medium Weight):

3. Coverage (weight_coverage, default: 1.0)
   - Fraction of boundary band covered by edge
   - Higher coverage = better
   - Target: > 0.85

4. Gap Ratio (weight_gap, default: 1.0)
   - Ratio of gaps in the edge
   - Lower gap_ratio = better
   - Target: < 0.18

5. Outside Ratio (weight_outside, default: 1.0)
   - Ratio of edge pixels outside the boundary band
   - Lower outside = better
   - Target: < 0.05

6. Thickness (weight_thickness, default: 1.2)
   - Average thickness of the edge
   - Lower thickness = better (thinner edge)
   - Target: < 0.15

7. Intrusion (weight_intrusion, default: 1.0)
   - Ratio of edge pixels intruding into interior
   - Lower intrusion = better
   - Target: < 0.03

8. Endpoints (weight_endpoints, default: 1.0)
   - Ratio of endpoint pixels (degree 1)
   - Lower endpoint_ratio = better (fewer endpoints)
   - Target: < 0.05

9. Wrinkle (weight_wrinkle, default: 1.0)
   - Ratio of pixels that differ after smoothing
   - Lower wrinkle_ratio = better (smoother edge)
   - Target: < 0.20

10. Branch (weight_branch, default: 2.0)
    - Ratio of branch pixels (degree >= 3)
    - Lower branch_ratio = better (fewer branches)
    - Target: < 0.08
    - Currently weighted 2x for importance

PENALTY FACTORS:

11. Low Quality (weight_low_quality, default: 0.5)
    - Multiplicative penalty for low quality regions
    - Applied as: score *= (1.0 + weight_low_quality)

12. Endpoint/Wrinkle/Branch Penalty
    - Exponential penalty: exp(-2.5 * (endpoint + wrinkle + branch))
    - Applied automatically regardless of weights

SCORING FORMULA:
- Each metric qi is converted to quality (0-1) via sigmoid
- Weighted geometric mean: score = exp(Σ(wi/W) * log(qi))
- Then multiplied by exponential penalty
- Final score clamped to [0, 1]

ADJUSTING WEIGHTS:
- Increase weight to emphasize that metric more
- Decrease weight to reduce its influence
- Main metrics (continuity, band_fit) have highest defaults
- Branch is currently 2x weighted for importance
"""
        text_widget.insert("1.0", help_text)
        text_widget.config(state=tk.DISABLED)

        button_frame = ttk.Frame(help_window)
        button_frame.pack(fill=tk.X, padx=12, pady=8)
        ttk.Button(button_frame, text="Close", command=help_window.destroy).pack(side=tk.RIGHT)

    def _show_auto_params_help(self):
        win = tk.Toplevel(self.root)
        win.title("Auto Search Params Help")
        win.geometry("720x580")
        win.transient(self.root)
        f = ttk.Frame(win, padding=12)
        f.pack(fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(f)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        txt = tk.Text(f, wrap=tk.WORD, yscrollcommand=sb.set, padx=8, pady=8, font=("Consolas", 9))
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=txt.yview)
        content = """Auto Search Range / Scoring — Parameter Impact and Steps

HIGH IMPACT (use fine steps; small changes affect edge quality a lot):

• NMS min/max/step — Non-maximum suppression threshold. Higher = fewer, stronger edges; lower = more edges, risk of noise. Step: 0.005 recommended.
• High min/max/step — High hysteresis threshold (strong edge pixels). Directly sets edge strength. Step: 0.005.
• Low factor min/max/step — Ratio of low to high threshold. Affects edge continuity. Step: 0.02.
• Margin min/max/step — Polarity drop margin; removes opposite-gradient pixels. Step: 0.02.
• Blur sigma min/max/step — Pre-blur strength. Smooths noise; too high loses detail. Step: 0.1 (0.15 if low-quality images).
• Contrast ref min/max/step — Reference contrast for normalization. Grayscale images often suit 75–105. Step: 5.
• Min scale min/max/step — Minimum threshold scale. Step: 0.05.

MEDIUM IMPACT:

• Band min/max — Boundary band radius (pixels). Step: 1.
• Thinning min/max/step — Zhang-Suen iterations. Step: 2.
• Blur kernel min/max/step — Blur kernel size (odd). Step: 2.
• Magnitude gamma min/max/step — Gradient magnitude curve. Step: 0.1.

LOWER IMPACT (coarser steps or leave default):

• Soft high/min/step, Soft low factor — Soft linking; step 0.02–0.05.
• Link radius min/max — Step 1.
• Edge smooth prob, radius, iters — Post-process smoothing; small effect.
• Spur prune min/max — Removes short spurs; 0–2, step 1.
• Closing prob, radius, iters — Morphological closing; minor.
• Median kernel — Step 2.

SCORING WEIGHTS (importance in final score):

• weight_continuity, weight_band_fit — Highest (24, 12). Edge continuity and band fit.
• weight_thickness, weight_branch — Important (1.2, 2.0). Thinner edges; fewer branches.
• weight_coverage, weight_gap, weight_outside, weight_endpoints, weight_wrinkle, weight_intrusion — Normal (1.0).
• weight_low_quality — Penalty for low-contrast regions (0.5).

USER PRE-ANSWERS (before starting Auto):

• Grayscale? Yes → narrower contrast_ref, tighter magnitude_gamma. No → wider contrast range.
• Low quality? Yes → more blur, wider NMS range, higher spur prune, higher low_quality penalty. No → finer steps, less blur.
"""
        txt.insert("1.0", content)
        txt.config(state=tk.DISABLED)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=8)

    def _compute_boundary(self, mask):
        padded = np.pad(mask, 1, mode="edge")
        center = padded[1:-1, 1:-1]
        boundary = np.zeros_like(center, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = padded[1 + dy : 1 + dy + center.shape[0], 1 + dx : 1 + dx + center.shape[1]]
                boundary |= neighbor != center
        return boundary

    def _count_components(self, mask):
        if mask is None or not mask.any():
            return 0
        visited = np.zeros(mask.shape, dtype=bool)
        coords = np.argwhere(mask)
        components = 0
        for y, x in coords:
            if visited[y, x]:
                continue
            components += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny = cy + dy
                        nx = cx + dx
                        if ny < 0 or nx < 0 or ny >= mask.shape[0] or nx >= mask.shape[1]:
                            continue
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
        return components

    def _evaluate_settings(self, data, settings, auto_config):
        total_score = 0.0
        total_coverage = 0.0
        total_intrusion = 0.0
        total_outside = 0.0
        total_gap = 0.0
        total_thickness = 0.0
        total_continuity = 0.0
        total_band_ratio = 0.0
        total_endpoints = 0.0
        total_wrinkle = 0.0
        total_branch = 0.0
        total_weight = 0.0
        total_q_cont = 0.0
        total_q_band = 0.0
        total_q_thick = 0.0
        total_q_intr = 0.0
        total_q_end = 0.0
        total_q_wrinkle = 0.0
        total_q_branch = 0.0

        settings_eval = dict(settings)
        settings_eval["use_boundary_band_filter"] = False

        def evaluate_item(item):
            image = item["image"]
            mask = item["mask"]
            band_radius = settings["boundary_band_radius"]
            band = item["bands"].get(band_radius)
            if band is None:
                if band_radius <= 0:
                    band = item["boundary"]
                else:
                    band = self.detector.dilate_binary(item["boundary"], band_radius)

            results = self.detector.detect_edges_array(
                image,
                use_nms=True,
                use_hysteresis=True,
                **settings_eval,
            )
            edges_raw = results["edges"] > 0
            edge_pixels = int(edges_raw.sum())
            band_pixels = item["band_pixels"].get(band_radius, int(band.sum()))
            edges_in_band = int((edges_raw & band).sum()) if band_pixels else 0
            intrusion = int((edges_raw & mask & ~band).sum())
            outside = int((edges_raw & ~mask & ~band).sum())

            if edge_pixels == 0:
                coverage = 0.0
                intrusion_ratio = 1.0
                outside_ratio = 1.0
                gap_ratio = 1.0
                continuity_penalty = 1.0
                thickness_penalty = 1.0
                band_ratio = 0.0
                endpoint_ratio = 1.0
                wrinkle_ratio = 1.0
                branch_ratio = 1.0
            else:
                coverage = edges_in_band / band_pixels if band_pixels else 0.0
                intrusion_ratio = intrusion / edge_pixels
                outside_ratio = outside / edge_pixels
                gap_ratio = max(0.0, 1.0 - coverage)
                band_ratio = edges_in_band / edge_pixels if edge_pixels else 0.0
                neighbor_counts = self.detector._neighbor_count(edges_raw)
                endpoint_count = int((edges_raw & (neighbor_counts <= 1)).sum())
                branch_count = int((edges_raw & (neighbor_counts >= 3)).sum())
                endpoint_ratio = endpoint_count / edge_pixels
                branch_ratio = branch_count / edge_pixels
                smooth_mask = self.detector.erode_binary(self.detector.dilate_binary(edges_raw, 1), 1)
                wrinkle_ratio = int((edges_raw ^ smooth_mask).sum()) / edge_pixels
                components = self._count_components(edges_raw & band)
                components_penalty = max(0.0, components - 1) / max(1, edges_in_band)
                continuity_penalty = min(1.0, components_penalty * 5.0)
                edge_density = edge_pixels / max(band_pixels, 1)
                thickness_penalty = max(0.0, edge_density - 1.2)

            metrics = {
                "coverage": coverage,
                "gap": gap_ratio,
                "continuity": continuity_penalty,
                "intrusion": intrusion_ratio,
                "outside": outside_ratio,
                "thickness": thickness_penalty,
                "band_ratio": band_ratio,
                "endpoints": endpoint_ratio,
                "wrinkle": wrinkle_ratio,
                "branch": branch_ratio,
            }
            score, details = compute_auto_score(metrics, auto_config, return_details=True)
            p10, p90 = np.percentile(image, [10, 90])
            contrast = max(float(p90 - p10), 1.0)
            if contrast < settings["contrast_ref"] * 0.6:
                score *= (1.0 + auto_config["weight_low_quality"])

            weight = float(item.get("weight", 1.0))
            return (
                weight,
                score,
                coverage,
                intrusion_ratio,
                outside_ratio,
                gap_ratio,
                thickness_penalty,
                continuity_penalty,
                band_ratio,
                endpoint_ratio,
                wrinkle_ratio,
                branch_ratio,
                details,
            )

        use_parallel = bool(auto_config.get("auto_parallel", False))
        max_workers = max(1, int(auto_config.get("auto_workers", 1)))
        results = []
        if use_parallel and len(data) > 1 and max_workers > 1:
            workers = min(max_workers, len(data))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(evaluate_item, data))
        else:
            results = [evaluate_item(item) for item in data]

        for (
            weight,
            score,
            coverage,
            intrusion_ratio,
            outside_ratio,
            gap_ratio,
            thickness_penalty,
            continuity_penalty,
            band_ratio,
            endpoint_ratio,
            wrinkle_ratio,
            branch_ratio,
            details,
        ) in results:
            total_weight += weight
            total_score += score * weight
            total_coverage += coverage * weight
            total_intrusion += intrusion_ratio * weight
            total_outside += outside_ratio * weight
            total_gap += gap_ratio * weight
            total_thickness += thickness_penalty * weight
            total_continuity += continuity_penalty * weight
            total_band_ratio += band_ratio * weight
            total_endpoints += endpoint_ratio * weight
            total_wrinkle += wrinkle_ratio * weight
            total_branch += branch_ratio * weight
            total_q_cont += details["q_cont"] * weight
            total_q_band += details["q_band"] * weight
            total_q_thick += details["q_thick"] * weight
            total_q_intr += details["q_intr"] * weight
            total_q_end += details["q_end"] * weight
            total_q_wrinkle += details["q_wrinkle"] * weight
            total_q_branch += details["q_branch"] * weight

        if total_weight <= 0:
            return (
                0.0,
                {
                    "coverage": 0.0,
                    "intrusion": 1.0,
                    "outside": 1.0,
                    "gap": 1.0,
                    "thickness": 1.0,
                    "continuity": 1.0,
                    "band_ratio": 0.0,
                    "endpoints": 1.0,
                    "wrinkle": 1.0,
                    "branch": 1.0,
                },
                {
                    "q_cont": 0.0,
                    "q_band": 0.0,
                    "q_thick": 0.0,
                    "q_intr": 0.0,
                    "q_end": 0.0,
                    "q_wrinkle": 0.0,
                    "q_branch": 0.0,
                },
            )

        avg_score = max(0.0, total_score / total_weight)
        summary = {
            "coverage": total_coverage / total_weight,
            "intrusion": total_intrusion / total_weight,
            "outside": total_outside / total_weight,
            "gap": total_gap / total_weight,
            "thickness": total_thickness / total_weight,
            "continuity": total_continuity / total_weight,
            "band_ratio": total_band_ratio / total_weight,
            "endpoints": total_endpoints / total_weight,
            "wrinkle": total_wrinkle / total_weight,
            "branch": total_branch / total_weight,
        }
        qualities = {
            "q_cont": total_q_cont / total_weight,
            "q_band": total_q_band / total_weight,
            "q_thick": total_q_thick / total_weight,
            "q_intr": total_q_intr / total_weight,
            "q_end": total_q_end / total_weight,
            "q_wrinkle": total_q_wrinkle / total_weight,
            "q_branch": total_q_branch / total_weight,
        }
        return avg_score, summary, qualities

    def _sample_candidates(self, items, budget):
        if budget <= 0 or len(items) <= budget:
            return items
        stride = max(1, len(items) // budget)
        sampled = items[::stride]
        return sampled[:budget]

    def _sample_float(self, rng, min_val, max_val, step, center=None, scale=1.0):
        min_val = float(min_val)
        max_val = float(max_val)
        step = float(step) if step else 0.0
        if center is None:
            value = rng.uniform(min_val, max_val)
        else:
            sigma = max(step * scale, (max_val - min_val) * 0.05 * scale)
            value = rng.normal(center, sigma)
        value = min(max_val, max(min_val, value))
        if step > 0:
            value = round(round((value - min_val) / step) * step + min_val, 4)
        return float(value)

    def _sample_int(self, rng, min_val, max_val, step, center=None, scale=1.0, odd=False):
        min_val = int(min_val)
        max_val = int(max_val)
        step = max(1, int(step))
        if center is None:
            value = rng.randint(min_val, max_val + 1)
        else:
            sigma = max(step * scale, max(1, int((max_val - min_val) * 0.05 * scale)))
            value = int(round(rng.normal(center, sigma)))
        value = max(min_val, min(max_val, value))
        if odd and value % 2 == 0:
            value = value + 1 if value + 1 <= max_val else value - 1
        if step > 1:
            value = min_val + (int(round((value - min_val) / step)) * step)
            value = max(min_val, min(max_val, value))
            if odd and value % 2 == 0:
                value = value + 1 if value + 1 <= max_val else value - 1
        return int(value)

    def _build_candidates(self, base_settings, mode, auto_config, count, rng, step_scale=1.0, centers=None, step_multipliers=None):
        candidates = []
        seen = set()
        centers = centers or []
        mult = step_multipliers or {}

        def step_for(key, default_step):
            return float(default_step) * mult.get(key, 1.0)

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
        blur_kernel_min = min(auto_config["auto_blur_kernel_min"], auto_config["auto_blur_kernel_max"])
        blur_kernel_max = max(auto_config["auto_blur_kernel_min"], auto_config["auto_blur_kernel_max"])
        thinning_min = min(auto_config["auto_thinning_min"], auto_config["auto_thinning_max"])
        thinning_max = max(auto_config["auto_thinning_min"], auto_config["auto_thinning_max"])
        contrast_min = min(auto_config["auto_contrast_ref_min"], auto_config["auto_contrast_ref_max"])
        contrast_max = max(auto_config["auto_contrast_ref_min"], auto_config["auto_contrast_ref_max"])
        min_scale_min = min(auto_config["auto_min_scale_min"], auto_config["auto_min_scale_max"])
        min_scale_max = max(auto_config["auto_min_scale_min"], auto_config["auto_min_scale_max"])
        soft_high_min = min(auto_config["auto_soft_high_min"], auto_config["auto_soft_high_max"])
        soft_high_max = max(auto_config["auto_soft_high_min"], auto_config["auto_soft_high_max"])
        soft_low_min = min(auto_config["auto_soft_low_factor_min"], auto_config["auto_soft_low_factor_max"])
        soft_low_max = max(auto_config["auto_soft_low_factor_min"], auto_config["auto_soft_low_factor_max"])
        link_radius_min = min(auto_config["auto_link_radius_min"], auto_config["auto_link_radius_max"])
        link_radius_max = max(auto_config["auto_link_radius_min"], auto_config["auto_link_radius_max"])
        smooth_radius_min = min(auto_config["auto_edge_smooth_radius_min"], auto_config["auto_edge_smooth_radius_max"])
        smooth_radius_max = max(auto_config["auto_edge_smooth_radius_min"], auto_config["auto_edge_smooth_radius_max"])
        smooth_iters_min = min(auto_config["auto_edge_smooth_iters_min"], auto_config["auto_edge_smooth_iters_max"])
        smooth_iters_max = max(auto_config["auto_edge_smooth_iters_min"], auto_config["auto_edge_smooth_iters_max"])
        spur_min = min(auto_config["auto_spur_prune_min"], auto_config["auto_spur_prune_max"])
        spur_max = max(auto_config["auto_spur_prune_min"], auto_config["auto_spur_prune_max"])
        closing_radius_min = min(auto_config["auto_closing_radius_min"], auto_config["auto_closing_radius_max"])
        closing_radius_max = max(auto_config["auto_closing_radius_min"], auto_config["auto_closing_radius_max"])
        closing_iter_min = min(auto_config["auto_closing_iter_min"], auto_config["auto_closing_iter_max"])
        closing_iter_max = max(auto_config["auto_closing_iter_min"], auto_config["auto_closing_iter_max"])
        gamma_min = min(auto_config["auto_magnitude_gamma_min"], auto_config["auto_magnitude_gamma_max"])
        gamma_max = max(auto_config["auto_magnitude_gamma_min"], auto_config["auto_magnitude_gamma_max"])
        median_min = min(auto_config["auto_median_kernel_min"], auto_config["auto_median_kernel_max"])
        median_max = max(auto_config["auto_median_kernel_min"], auto_config["auto_median_kernel_max"])

        def candidate_key(settings):
            return (
                round(settings["nms_relax"], 3),
                round(settings["high_ratio"], 3),
                round(settings["low_ratio"], 3),
                int(settings["boundary_band_radius"]),
                round(settings["polarity_drop_margin"], 3),
                round(settings["blur_sigma"], 2),
                int(settings["blur_kernel_size"]),
                int(settings["thinning_max_iter"]),
                round(settings["contrast_ref"], 1),
                round(settings["min_threshold_scale"], 2),
                round(settings["magnitude_gamma"], 2),
                int(settings["median_kernel_size"]),
                int(settings["link_radius"]),
                round(settings["soft_high_ratio"], 2),
                round(settings["soft_low_ratio"], 2),
                int(settings["edge_smooth_radius"]),
                int(settings["edge_smooth_iters"]),
                int(settings["spur_prune_iters"]),
                int(settings["closing_radius"]),
                int(settings["closing_iterations"]),
            )

        max_tries = max(count * 8, 200)
        tries = 0
        while len(candidates) < count and tries < max_tries:
            tries += 1
            center = centers[rng.randint(0, len(centers))] if centers else None
            nms_center = center["nms_relax"] if center else None
            high_center = center["high_ratio"] if center else None
            low_center = None
            if center:
                low_center = center["low_ratio"] / max(center["high_ratio"], 1e-6)
            margin_center = center["polarity_drop_margin"] if center else None
            band_center = center["boundary_band_radius"] if center else None
            blur_sigma_center = center["blur_sigma"] if center else None
            blur_kernel_center = center["blur_kernel_size"] if center else None
            thinning_center = center["thinning_max_iter"] if center else None
            contrast_center = center["contrast_ref"] if center else None
            min_scale_center = center["min_threshold_scale"] if center else None

            nms_relax = self._sample_float(rng, nms_min, nms_max, step_for("nms_relax", auto_config["auto_nms_step"]), nms_center, step_scale)
            high_ratio = self._sample_float(rng, high_min, high_max, step_for("high_ratio", auto_config["auto_high_step"]), high_center, step_scale)
            low_factor = self._sample_float(
                rng,
                low_min,
                low_max,
                step_for("low_ratio", auto_config["auto_low_factor_step"]),
                low_center,
                step_scale,
            )
            band_radius = self._sample_int(
                rng, band_min, band_max, 1, band_center, step_scale, odd=False
            )
            margin = self._sample_float(
                rng, margin_min, margin_max, step_for("polarity_drop_margin", auto_config["auto_margin_step"]), margin_center, step_scale
            )
            blur_sigma = self._sample_float(
                rng,
                blur_sigma_min,
                blur_sigma_max,
                step_for("blur_sigma", auto_config["auto_blur_sigma_step"]),
                blur_sigma_center,
                step_scale,
            )
            blur_kernel = self._sample_int(
                rng,
                blur_kernel_min,
                blur_kernel_max,
                max(1, int(step_for("blur_kernel_size", auto_config["auto_blur_kernel_step"]))),
                blur_kernel_center,
                step_scale,
                odd=True,
            )
            thinning_iter = self._sample_int(
                rng,
                thinning_min,
                thinning_max,
                max(1, int(step_for("thinning_max_iter", auto_config["auto_thinning_step"]))),
                thinning_center,
                step_scale,
            )
            contrast_ref = self._sample_float(
                rng,
                contrast_min,
                contrast_max,
                step_for("contrast_ref", auto_config["auto_contrast_ref_step"]),
                contrast_center,
                step_scale,
            )
            min_scale = self._sample_float(
                rng,
                min_scale_min,
                min_scale_max,
                step_for("min_threshold_scale", auto_config["auto_min_scale_step"]),
                min_scale_center,
                step_scale,
            )
            magnitude_gamma = self._sample_float(
                rng,
                gamma_min,
                gamma_max,
                step_for("magnitude_gamma", auto_config["auto_magnitude_gamma_step"]),
                None,
                step_scale,
            )
            median_kernel = self._sample_int(
                rng,
                median_min,
                median_max,
                max(1, int(step_for("median_kernel_size", auto_config["auto_median_kernel_step"]))),
                None,
                step_scale,
                odd=True,
            )
            use_soft_linking = rng.rand() < float(auto_config["auto_soft_link_prob"])
            soft_high = self._sample_float(
                rng,
                soft_high_min,
                soft_high_max,
                step_for("soft_high_ratio", auto_config["auto_soft_high_step"]),
                None,
                step_scale,
            )
            soft_factor = self._sample_float(
                rng,
                soft_low_min,
                soft_low_max,
                step_for("soft_low_ratio", auto_config["auto_soft_low_factor_step"]),
                None,
                step_scale,
            )
            soft_low = max(0.01, soft_high * soft_factor)
            link_radius = self._sample_int(
                rng,
                link_radius_min,
                link_radius_max,
                max(1, int(step_for("link_radius", auto_config["auto_link_radius_step"]))),
                None,
                step_scale,
            )
            use_edge_smooth = rng.rand() < float(auto_config["auto_use_edge_smooth_prob"])
            edge_smooth_radius = self._sample_int(rng, smooth_radius_min, smooth_radius_max, 1, None, step_scale)
            edge_smooth_iters = self._sample_int(rng, smooth_iters_min, smooth_iters_max, 1, None, step_scale)
            spur_prune_iters = self._sample_int(rng, spur_min, spur_max, 1, None, step_scale)
            use_closing = rng.rand() < float(auto_config["auto_use_closing_prob"])
            closing_radius = self._sample_int(
                rng, closing_radius_min, closing_radius_max, 1, None, step_scale
            )
            closing_iters = self._sample_int(
                rng, closing_iter_min, closing_iter_max, 1, None, step_scale
            )

            low_ratio = max(0.02, high_ratio * low_factor)

            settings = dict(base_settings)
            settings.update(
                {
                    "nms_relax": round(nms_relax, 3),
                    "high_ratio": float(high_ratio),
                    "low_ratio": float(low_ratio),
                    "boundary_band_radius": int(band_radius),
                    "polarity_drop_margin": float(margin),
                    "use_boundary_band_filter": int(band_radius) > 0,
                    "blur_sigma": float(blur_sigma),
                    "blur_kernel_size": int(max(3, blur_kernel)),
                    "thinning_max_iter": int(max(1, thinning_iter)),
                    "contrast_ref": float(contrast_ref),
                    "min_threshold_scale": float(min_scale),
                    "magnitude_gamma": float(magnitude_gamma),
                    "median_kernel_size": int(max(3, median_kernel)),
                    "use_soft_linking": bool(use_soft_linking),
                    "soft_high_ratio": float(soft_high),
                    "soft_low_ratio": float(soft_low),
                    "link_radius": int(max(0, link_radius)),
                    "use_edge_smooth": bool(use_edge_smooth),
                    "edge_smooth_radius": int(edge_smooth_radius if use_edge_smooth else 0),
                    "edge_smooth_iters": int(edge_smooth_iters),
                    "spur_prune_iters": int(spur_prune_iters),
                    "use_closing": bool(use_closing),
                    "closing_radius": int(closing_radius if use_closing else 0),
                    "closing_iterations": int(closing_iters),
                }
            )
            key = candidate_key(settings)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(settings)

        return candidates

    def _build_local_grid(self, best, base_settings, auto_config, size=32):
        """Build a small grid of settings around best (nms, high_ratio, margin, band_radius)."""
        out = []
        nms_s = max(0.002, auto_config.get("auto_nms_step", 0.01) * 0.5)
        high_s = max(0.002, auto_config.get("auto_high_step", 0.01) * 0.5)
        margin_s = max(0.02, auto_config.get("auto_margin_step", 0.05) * 0.5)
        nms_vals = [round(best["nms_relax"] + d, 3) for d in (-nms_s, 0, nms_s)]
        high_vals = [max(0.05, best["high_ratio"] + d) for d in (-high_s, 0, high_s)]
        margin_vals = [max(0.0, best["polarity_drop_margin"] + d) for d in (-margin_s, 0, margin_s)]
        band_vals = [max(0, min(4, best["boundary_band_radius"] + d)) for d in (-1, 0, 1)]
        low_factor = 0.5 * (auto_config.get("auto_low_factor_min", 0.3) + auto_config.get("auto_low_factor_max", 0.6))
        for nms in nms_vals:
            nms = round(min(1.0, max(0.85, nms)), 3)
            for high in high_vals:
                low = max(0.02, high * low_factor)
                for margin in margin_vals:
                    for band in band_vals:
                        if len(out) >= size:
                            return out
                        s = dict(base_settings)
                        s.update({
                            "nms_relax": nms,
                            "high_ratio": float(high),
                            "low_ratio": float(low),
                            "polarity_drop_margin": max(0.0, margin),
                            "boundary_band_radius": int(band),
                            "use_boundary_band_filter": int(band) > 0,
                        })
                        out.append(s)
        return out

    def _ask_image_type_and_quality(self):
        """Show dialog: grayscale? low quality? Returns True to proceed, False to cancel."""
        win = tk.Toplevel(self.root)
        win.title("Input Image Type (for Auto Search Defaults)")
        win.transient(self.root)
        win.grab_set()
        win.geometry("480x280")
        f = ttk.Frame(win, padding=12)
        f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f, text="Answer the following so auto search defaults can be tuned. You can skip with 'Not sure'.", wraplength=420).pack(anchor="w", pady=(0, 12))
        grayscale_var = tk.StringVar(value="not_sure")
        ttk.Label(f, text="Are the input images grayscale (black/white only)?", font=("", 10, "bold")).pack(anchor="w", pady=(4, 2))
        ttk.Radiobutton(f, text="Yes (grayscale)", variable=grayscale_var, value="yes").pack(anchor="w")
        ttk.Radiobutton(f, text="No (color)", variable=grayscale_var, value="no").pack(anchor="w")
        ttk.Radiobutton(f, text="Not sure", variable=grayscale_var, value="not_sure").pack(anchor="w")
        low_quality_var = tk.StringVar(value="not_sure")
        ttk.Label(f, text="Is the image quality low (noisy, low resolution, blurry)?", font=("", 10, "bold")).pack(anchor="w", pady=(12, 2))
        ttk.Radiobutton(f, text="Yes (low quality)", variable=low_quality_var, value="yes").pack(anchor="w")
        ttk.Radiobutton(f, text="No (normal or high quality)", variable=low_quality_var, value="no").pack(anchor="w")
        ttk.Radiobutton(f, text="Not sure", variable=low_quality_var, value="not_sure").pack(anchor="w")
        result = [None]

        def on_ok():
            g = grayscale_var.get()
            self._auto_image_grayscale = True if g == "yes" else (False if g == "no" else None)
            q = low_quality_var.get()
            self._auto_image_low_quality = True if q == "yes" else (False if q == "no" else None)
            result[0] = True
            win.destroy()

        def on_cancel():
            result[0] = False
            win.destroy()

        btn_f = ttk.Frame(f)
        btn_f.pack(fill=tk.X, pady=(16, 0))
        ttk.Button(btn_f, text="OK (apply and start)", command=on_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_f, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=4)
        win.protocol("WM_DELETE_WINDOW", on_cancel)
        win.wait_window()
        return result[0] is True

    def _start_auto_optimize(self):
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showwarning("In Progress", "Processing is already running.")
            return
        if not self.selected_files:
            self._add_files()
        if not self.selected_files:
            return
        if not self._ask_image_type_and_quality():
            return

        try:
            base_settings = self._get_param_settings()
        except ValueError as exc:
            messagebox.showerror("Parameter Error", str(exc))
            return
        auto_config = self._get_auto_config()

        mode = self.auto_mode.get()
        self.status_var.set("Auto optimization started...")
        self.start_button.config(state=tk.DISABLED)
        self.auto_button.config(state=tk.DISABLED)
        if self.pause_button:
            self.pause_button.config(state=tk.NORMAL, text="Pause")
        if self.stop_button:
            self.stop_button.config(state=tk.NORMAL)
        self._auto_pause_event.clear()
        self._auto_stop_event.clear()
        self._auto_start_time = time.time()
        self._auto_last_best_time = self._auto_start_time
        self._auto_scores = []
        self._auto_best_scores = []
        self._auto_best_time_series = []
        self._auto_cont_scores = []
        self._auto_band_scores = []
        self._auto_penalty_scores = []
        self._auto_wrinkle_scores = []
        self._auto_endpoint_scores = []
        self._auto_branch_scores = []
        self._refresh_auto_graphs()
        display_mode = self.score_display_mode.get()
        self._worker_thread = threading.Thread(
            target=self._auto_optimize_worker,
            args=(list(self.selected_files), base_settings, auto_config, mode, dict(self.roi_map), display_mode),
            daemon=True,
        )
        self._worker_thread.start()
        self.root.after(100, self._poll_messages)

    def _auto_optimize_worker(self, files, base_settings, auto_config, mode, roi_map, display_mode):
        max_files = min(8, len(files)) if mode == "Fast" else len(files)
        is_perfect = mode == "Perfect"
        data = self._prepare_auto_data(files, base_settings, auto_config, roi_map, max_files)
        data_coarse = data.get("coarse", [])
        data_mid = data.get("mid", [])
        data_full = data.get("full", [])
        if not data_coarse:
            data_coarse = data_full
        if not data_mid:
            data_mid = data_full
        best = None
        best_score = 0.0
        report_lines = [f"[INFO] score_display={display_mode}"]
        scores = []
        best_progress = []
        evaluated = []
        seen_keys = set()
        stop_reason = None
        start_time = time.time()
        last_best_time = start_time
        early_stop_enabled = bool(auto_config.get("early_stop_enabled", True))
        early_stop_minutes = float(auto_config.get("early_stop_minutes", 10.0))
        round_early_exit_frac = float(auto_config.get("auto_round_early_exit_no_improve_frac", 0.25))
        no_improve_rounds_stop = int(auto_config.get("auto_no_improve_rounds_stop", 2))
        processed = 0

        if not data_full:
            report_dir = self._create_batch_output_dir()
            report_path = os.path.join(report_dir, "auto_optimize_report.txt")
            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write("No data available for auto optimization.\n")
            self._message_queue.put(("auto_done", None, report_path, "no_data"))
            return
        rng = np.random.RandomState(42)
        if mode == "Fast":
            target_eval = 3000
        elif is_perfect:
            target_eval = 45000
        else:
            target_eval = 9000
        round_budget = 200 if mode == "Fast" else (1200 if is_perfect else 500)
        step_mults = PERFECT_STEP_MULTIPLIERS if is_perfect else None
        phase1_frac = float(auto_config.get("auto_phase1_frac", 0.5))
        phase1_max_thickness = float(auto_config.get("auto_phase1_max_thickness", 0.25))
        phase1_min_evals = int(auto_config.get("auto_phase1_min_evals", 150))
        phase1_budget = max(phase1_min_evals, int(target_eval * phase1_frac))
        phase2_budget = target_eval
        base_pre = dict(base_settings)
        base_pre["use_thinning"] = False
        report_lines.append(
            f"[INFO] Two-phase optimization: Phase1 (no thinning) budget={phase1_budget}, "
            f"max_thickness={phase1_max_thickness}, then Phase2 (with thinning) up to {target_eval} evals."
        )

        def wait_if_paused():
            while self._auto_pause_event.is_set() and not self._auto_stop_event.is_set():
                time.sleep(0.2)
            return not self._auto_stop_event.is_set()

        def key_from(settings):
            return (
                bool(settings.get("use_thinning", True)),
                round(settings["nms_relax"], 3),
                round(settings["high_ratio"], 3),
                round(settings["low_ratio"], 3),
                int(settings["boundary_band_radius"]),
                round(settings["polarity_drop_margin"], 3),
                round(settings["blur_sigma"], 2),
                int(settings["blur_kernel_size"]),
                int(settings["thinning_max_iter"]),
                round(settings["contrast_ref"], 1),
                round(settings["min_threshold_scale"], 2),
                round(settings.get("magnitude_gamma", 1.0), 2),
                int(settings.get("median_kernel_size", 3)),
                int(settings.get("link_radius", 0)),
                round(settings.get("soft_high_ratio", 0.0), 2),
                round(settings.get("soft_low_ratio", 0.0), 2),
                int(settings.get("edge_smooth_radius", 0)),
                int(settings.get("edge_smooth_iters", 0)),
                int(settings.get("spur_prune_iters", 0)),
                int(settings.get("closing_radius", 0)),
                int(settings.get("closing_iterations", 0)),
            )

        def check_stagnation():
            if not early_stop_enabled:
                return False
            stagnant = (time.time() - last_best_time) > (early_stop_minutes * 60.0)
            if stagnant:
                return True
            return False

        no_improve_rounds = 0
        round_num = 0
        seq = 0
        best_thickness = None
        current_base = base_pre
        phase = 1
        phase_cap = phase1_budget
        thinning_min = min(auto_config["auto_thinning_min"], auto_config["auto_thinning_max"])
        thinning_max = max(auto_config["auto_thinning_min"], auto_config["auto_thinning_max"])
        mid_thinning = max(1, (thinning_min + thinning_max) // 2)

        while not stop_reason:
            if processed >= phase_cap:
                if phase == 1:
                    report_lines.append(
                        f"[INFO] Phase 1 (pre-thinning) done: processed={processed}, best_thickness={best_thickness}, best_score={best_score:.6e}"
                    )
                    if best is not None:
                        base_phase2 = dict(best)
                        base_phase2["use_thinning"] = True
                        base_phase2["thinning_max_iter"] = mid_thinning
                        current_base = base_phase2
                        phase_cap = target_eval
                        phase = 2
                        no_improve_rounds = 0
                        report_lines.append("[INFO] Starting Phase 2 (with thinning).")
                    else:
                        break
                else:
                    break

            round_num += 1
            if not wait_if_paused():
                stop_reason = "stopped"
                break
            if check_stagnation():
                stop_reason = f"stagnation>{early_stop_minutes:.0f}min"
                break
            top_list = [e[1] for e in sorted(evaluated, key=lambda e: e[0], reverse=True)[:5]] if evaluated else []
            centers_explore = top_list[:2] if top_list else []
            n_explore = min(round_budget // 3, 150)
            n_exploit = min(round_budget // 2, 220) if best else 0
            n_local = min(round_budget // 6, 48) if best else 0
            explore = self._build_candidates(
                current_base, mode, auto_config, n_explore, rng,
                step_scale=0.75, centers=centers_explore, step_multipliers=step_mults
            )
            exploit = self._build_candidates(
                current_base, mode, auto_config, n_exploit, rng,
                step_scale=0.2, centers=[best], step_multipliers=step_mults
            ) if best else []
            local_list = self._build_local_grid(best, current_base, auto_config, size=n_local) if best else []
            combined = (explore or []) + (exploit or []) + (local_list or [])
            if phase == 1:
                for s in combined:
                    s["use_thinning"] = False
            rng.shuffle(combined)
            pool = []
            for s in combined:
                k = key_from(s)
                if k not in seen_keys:
                    pool.append(s)
                    if len(pool) >= round_budget:
                        break
            if not pool:
                no_improve_rounds += 1
                if no_improve_rounds >= no_improve_rounds_stop:
                    stop_reason = f"no_new_candidates_{no_improve_rounds_stop}_rounds"
                    break
                report_lines.append(f"[INFO] Round {round_num}: no new candidates, best={best_score:.6e}")
                continue
            round_improved = False
            early_exit_after = max(8, int(round_early_exit_frac * len(pool)))
            no_improve_count = 0
            report_lines.append(f"[INFO] Round {round_num} pool={len(pool)} (explore+exploit+local), early_exit_after={early_exit_after} ({round_early_exit_frac*100:.0f}%)")

            for idx, settings in enumerate(pool):
                if processed >= phase_cap:
                    break
                if not wait_if_paused():
                    stop_reason = "stopped"
                    break
                if check_stagnation():
                    stop_reason = f"stagnation>{early_stop_minutes:.0f}min"
                    break
                if no_improve_count >= early_exit_after:
                    report_lines.append(f"[INFO] Round {round_num} early exit (no improvement after {early_exit_after})")
                    break
                key = key_from(settings)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                score, summary, qualities = self._evaluate_settings(data_full, settings, auto_config)
                processed += 1
                seq += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / max(1, processed)
                eta_seconds = avg_time * max(0, target_eval - processed)
                score_base = qualities["q_cont"] * qualities["q_band"]
                penalty = max(0.0, 1.0 - score)
                evaluated.append((score, settings, summary))
                score_disp = self._score_to_display(score, display_mode)
                report_lines.append(
                    f"r{round_num}_{seq:04d} score={score_disp:.12f} raw={score:.12e} base={score_base:.6f} pen={penalty:.6f} "
                    f"coverage={summary['coverage']:.4f} gap={summary['gap']:.4f} "
                    f"cont={summary['continuity']:.4f} band={summary['band_ratio']:.4f} "
                    f"end={summary['endpoints']:.4f} wrk={summary['wrinkle']:.4f} br={summary['branch']:.4f} "
                    f"nms={settings['nms_relax']:.2f} high={settings['high_ratio']:.3f} band={settings['boundary_band_radius']} margin={settings['polarity_drop_margin']:.2f}"
                )
                scores.append(score)
                if best is None or score > best_score:
                    best_score = score
                    best = settings
                    best_thickness = summary["thickness"]
                    last_best_time = time.time()
                    round_improved = True
                    no_improve_count = 0
                    self._message_queue.put(("auto_best", best_score, elapsed))
                else:
                    no_improve_count += 1
                best_progress.append(best_score)
                self._auto_best_time_series.append((elapsed, best_score))
                self._message_queue.put(
                    (
                        "auto_progress",
                        seq,
                        target_eval,
                        score,
                        best_score,
                        eta_seconds,
                        f"round{round_num}",
                        elapsed,
                        summary,
                        qualities,
                        score_base,
                        penalty,
                    )
                )

            if not round_improved:
                no_improve_rounds += 1
                if no_improve_rounds >= no_improve_rounds_stop:
                    stop_reason = f"no_improve_{no_improve_rounds_stop}_rounds"
                    break
            else:
                no_improve_rounds = 0
            report_lines.append(f"[INFO] Round {round_num} done best={best_score:.6e} processed={processed} no_improve_rounds={no_improve_rounds}/{no_improve_rounds_stop}")

            if phase == 1 and best is not None and best_thickness is not None and best_thickness <= phase1_max_thickness and processed >= phase1_min_evals:
                report_lines.append(
                    f"[INFO] Phase 1 (pre-thinning) thickness target met: best_thickness={best_thickness:.4f}<={phase1_max_thickness}, switching to Phase 2."
                )
                base_phase2 = dict(best)
                base_phase2["use_thinning"] = True
                base_phase2["thinning_max_iter"] = mid_thinning
                current_base = base_phase2
                phase_cap = target_eval
                phase = 2
                no_improve_rounds = 0
                report_lines.append("[INFO] Starting Phase 2 (with thinning).")

        if stop_reason:
            report_lines.append(f"[STOP] Optimization ended: {stop_reason}")

        if best is not None:
            report_lines.append("")
            report_lines.append("[BEST] Parameters and score at end of optimization:")
            report_lines.append(f"  best_score (raw) = {best_score:.12e}")
            report_lines.append(f"  best_score (display) = {self._format_score_for_display(best_score, display_mode)}")
            report_lines.append("  nms_relax=%s high_ratio=%s low_ratio=%s boundary_band_radius=%s polarity_drop_margin=%s" % (
                best.get("nms_relax"), best.get("high_ratio"), best.get("low_ratio"),
                best.get("boundary_band_radius"), best.get("polarity_drop_margin")))

        report_dir = self._create_batch_output_dir()
        report_path = os.path.join(report_dir, "auto_optimize_report.txt")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(report_lines))

        time_csv = os.path.join(report_dir, "auto_optimize_best_time.csv")
        with open(time_csv, "w", encoding="utf-8") as handle:
            handle.write("elapsed_sec,best_score\n")
            for elapsed, score in self._auto_best_time_series:
                handle.write(f"{elapsed:.2f},{score:.6f}\n")

        graph_path = os.path.join(report_dir, "auto_optimize_scores.png")
        best_path = os.path.join(report_dir, "auto_optimize_best.png")
        time_path = os.path.join(report_dir, "auto_optimize_best_time.png")
        display_scores = [self._score_to_display(v, display_mode) for v in scores]
        display_best = [self._score_to_display(v, display_mode) for v in best_progress]
        display_time = [(t, self._score_to_display(v, display_mode)) for t, v in self._auto_best_time_series]
        self._draw_score_graph(display_scores, graph_path, "Score by Candidate")
        self._draw_score_graph(display_best, best_path, "Best Score Progress")
        self._draw_time_graph(display_time, time_path, "Best Score vs Time (min)")

        config_path = os.path.join(report_dir, "auto_optimize_config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"best": best, "best_score": best_score, "auto_config": auto_config, "base_settings": base_settings},
                handle,
                ensure_ascii=True,
                indent=2,
            )

        self._message_queue.put(("auto_done", best, report_path, stop_reason, best_score))

    def _downsample_values(self, values, max_points=600):
        if not values:
            return [], []
        n = len(values)
        if n <= max_points:
            xs = list(range(1, n + 1))
            return xs, list(values)
        block = n / float(max_points)
        xs = []
        ys = []
        for i in range(max_points):
            start = int(i * block)
            end = int((i + 1) * block)
            if start >= n:
                break
            segment = values[start:end] if end > start else [values[start]]
            avg = float(sum(segment)) / max(1, len(segment))
            xs.append((start + end) * 0.5 + 1)
            ys.append(avg)
        return xs, ys

    def _render_graph(self, values, title, width=400, height=220, display_mode=None):
        display_mode = display_mode or (self.score_display_mode.get() if self.score_display_mode else "scaled")
        # Professional math/physics style: generous spacing, thin lines, clear axes
        margin_left = 52
        margin_right = 24
        margin_top = 56
        margin_bottom = 44
        title_gap = 14
        img = Image.new("RGB", (width, height), (252, 252, 252))
        draw = ImageDraw.Draw(img)

        left = margin_left
        top = margin_top
        right = width - margin_right
        bottom = height - margin_bottom
        draw.rectangle([left, top, right, bottom], outline=(40, 40, 40), width=1)
        draw.text((left, title_gap), title, fill=(20, 20, 20))

        if not values:
            return img

        xs, ys = self._downsample_values(values, max_points=700)
        min_val = min(0.0, float(min(ys)))
        max_val = float(max(ys))
        if max_val <= min_val:
            max_val = min_val + 1.0

        min_x = min(xs)
        max_x = max(xs)
        if max_x <= min_x:
            max_x = min_x + 1.0

        def scale_x(xval):
            return left + (xval - min_x) * (right - left) / (max_x - min_x)

        def scale_y(val):
            return bottom - (val - min_val) * (bottom - top) / (max_val - min_val)

        def fmt_y(val):
            return self._format_score_for_display(val, display_mode) if display_mode == "raw" else f"{val:.2f}"

        # Light grid
        ticks = 5
        for i in range(1, ticks):
            tx = left + i * (right - left) / ticks
            draw.line([(tx, top), (tx, bottom)], fill=(230, 230, 230), width=1)
            ty = bottom - i * (bottom - top) / ticks
            draw.line([(left, ty), (right, ty)], fill=(230, 230, 230), width=1)
        for i in range(ticks + 1):
            tx = left + i * (right - left) / ticks
            tick_val = int(round(min_x + i * (max_x - min_x) / ticks))
            draw.line([(tx, bottom), (tx, bottom + 5)], fill=(40, 40, 40), width=1)
            draw.text((tx - 10, bottom + 8), str(tick_val), fill=(30, 30, 30))

            ty = bottom - i * (bottom - top) / ticks
            y_val = min_val + i * (max_val - min_val) / ticks
            draw.line([(left - 5, ty), (left, ty)], fill=(40, 40, 40), width=1)
            draw.text((2, ty - 7), fmt_y(y_val), fill=(30, 30, 30))

        points = [(scale_x(xval), scale_y(v)) for xval, v in zip(xs, ys)]
        if len(points) >= 2:
            draw.line(points, fill=(0, 80, 160), width=1)
        else:
            draw.ellipse([points[0][0] - 1, points[0][1] - 1, points[0][0] + 1, points[0][1] + 1], fill=(0, 80, 160))
        return img

    def _render_time_graph(self, series, title, width=400, height=220, display_mode=None):
        display_mode = display_mode or (self.score_display_mode.get() if self.score_display_mode else "scaled")
        margin_left = 52
        margin_right = 24
        margin_top = 56
        margin_bottom = 44
        title_gap = 14
        img = Image.new("RGB", (width, height), (252, 252, 252))
        draw = ImageDraw.Draw(img)

        left = margin_left
        top = margin_top
        right = width - margin_right
        bottom = height - margin_bottom
        draw.rectangle([left, top, right, bottom], outline=(40, 40, 40), width=1)
        draw.text((left, title_gap), title, fill=(20, 20, 20))

        if not series:
            return img

        if len(series) > 700:
            stride = max(1, len(series) // 700)
            series = series[::stride]
        times = [float(t) for t, _ in series]
        values = [float(v) for _, v in series]
        min_x = min(0.0, min(times))
        max_x = max(times)
        if max_x <= min_x:
            max_x = min_x + 1.0

        min_y = min(0.0, min(values))
        max_y = max(values)
        if max_y <= min_y:
            max_y = min_y + 1.0

        def scale_x(t):
            return left + (t - min_x) * (right - left) / (max_x - min_x)

        def scale_y(val):
            return bottom - (val - min_y) * (bottom - top) / (max_y - min_y)

        ticks = 5
        for i in range(1, ticks):
            tx = left + i * (right - left) / ticks
            draw.line([(tx, top), (tx, bottom)], fill=(230, 230, 230), width=1)
            ty = bottom - i * (bottom - top) / ticks
            draw.line([(left, ty), (right, ty)], fill=(230, 230, 230), width=1)
        for i in range(ticks + 1):
            tx = left + i * (right - left) / ticks
            t_val = min_x + i * (max_x - min_x) / ticks
            draw.line([(tx, bottom), (tx, bottom + 5)], fill=(40, 40, 40), width=1)
            draw.text((tx - 12, bottom + 8), f"{t_val/60.0:.1f}", fill=(30, 30, 30))

            ty = bottom - i * (bottom - top) / ticks
            y_val = min_y + i * (max_y - min_y) / ticks
            draw.line([(left - 5, ty), (left, ty)], fill=(40, 40, 40), width=1)
            y_str = self._format_score_for_display(y_val, display_mode) if display_mode == "raw" else f"{y_val:.2f}"
            draw.text((2, ty - 7), y_str, fill=(30, 30, 30))

        points = [(scale_x(t), scale_y(v)) for t, v in series]
        if len(points) >= 2:
            draw.line(points, fill=(0, 80, 160), width=1)
        else:
            draw.ellipse([points[0][0] - 1, points[0][1] - 1, points[0][0] + 1, points[0][1] + 1], fill=(0, 80, 160))
        return img

    def _render_multi_graph(self, series_list, title, labels, colors, width=400, height=220, display_mode=None):
        display_mode = display_mode or (self.score_display_mode.get() if self.score_display_mode else "scaled")
        margin_left = 52
        margin_right = 24
        margin_top = 56
        margin_bottom = 44
        title_gap = 14
        legend_gap = 28
        img = Image.new("RGB", (width, height), (252, 252, 252))
        draw = ImageDraw.Draw(img)

        left = margin_left
        top = margin_top
        right = width - margin_right
        bottom = height - margin_bottom
        draw.rectangle([left, top, right, bottom], outline=(40, 40, 40), width=1)
        draw.text((left, title_gap), title, fill=(20, 20, 20))

        values = [v for series in series_list for v in series]
        if not values:
            return img

        min_y = min(0.0, min(values))
        max_y = max(values)
        if max_y <= min_y:
            max_y = min_y + 1.0

        max_len = max(1, max(len(series) for series in series_list))
        def scale_x(idx, total):
            return left + idx * (right - left) / max(1, total - 1)

        def scale_y(val):
            return bottom - (val - min_y) * (bottom - top) / (max_y - min_y)

        ticks = 5
        for i in range(1, ticks):
            tx = left + i * (right - left) / ticks
            draw.line([(tx, top), (tx, bottom)], fill=(230, 230, 230), width=1)
            ty = bottom - i * (bottom - top) / ticks
            draw.line([(left, ty), (right, ty)], fill=(230, 230, 230), width=1)
        for i in range(ticks + 1):
            tx = left + i * (right - left) / ticks
            tick_val = int(round(1 + i * (max_len - 1) / ticks))
            draw.line([(tx, bottom), (tx, bottom + 5)], fill=(40, 40, 40), width=1)
            draw.text((tx - 10, bottom + 8), str(tick_val), fill=(30, 30, 30))

            ty = bottom - i * (bottom - top) / ticks
            y_val = min_y + i * (max_y - min_y) / ticks
            draw.line([(left - 5, ty), (left, ty)], fill=(40, 40, 40), width=1)
            y_str = self._format_score_for_display(y_val, display_mode) if display_mode == "raw" else f"{y_val:.2f}"
            draw.text((2, ty - 7), y_str, fill=(30, 30, 30))

        for series, label, color in zip(series_list, labels, colors):
            if not series:
                continue
            xs, ys = self._downsample_values(series, max_points=700)
            points = [(scale_x(xval, max_len), scale_y(v)) for xval, v in zip(xs, ys)]
            if len(points) >= 2:
                draw.line(points, fill=color, width=1)
            else:
                draw.ellipse(
                    [points[0][0] - 1, points[0][1] - 1, points[0][0] + 1, points[0][1] + 1],
                    fill=color,
                )

        legend_y = top - legend_gap
        legend_x = left
        for label, color in zip(labels, colors):
            draw.rectangle([legend_x, legend_y, legend_x + 10, legend_y + 10], fill=color, outline=(60, 60, 60))
            draw.text((legend_x + 14, legend_y - 2), label, fill=(30, 30, 30))
            legend_x += 92

        return img

    def _draw_score_graph(self, values, path, title):
        img = self._render_graph(values, title, width=800, height=300)
        img.save(path)

    def _draw_time_graph(self, series, path, title):
        img = self._render_time_graph(series, title, width=800, height=300)
        img.save(path)

    def _refresh_auto_graphs(self):
        if not self.score_graph_label or not self.best_graph_label:
            return
        mode_label = self.score_display_mode.get() if self.score_display_mode else "raw"
        display_scores = [self._score_to_display(v) for v in self._auto_scores]
        display_best_series = [
            (t, self._score_to_display(v)) for t, v in self._auto_best_time_series
        ] or [(i, self._score_to_display(v)) for i, v in enumerate(self._auto_best_scores)]
        score_img = self._render_graph(display_scores, f"Score ({mode_label})", width=460, height=260)
        best_img = self._render_time_graph(display_best_series, f"Best score ({mode_label})", width=460, height=260)
        metric_img = self._render_multi_graph(
            [self._auto_cont_scores, self._auto_band_scores],
            "Continuity & Band Fit",
            ["Continuity", "Band fit"],
            ["green", "blue"],
            width=460,
            height=260,
        )
        detail_img = self._render_multi_graph(
            [self._auto_endpoint_scores, self._auto_wrinkle_scores, self._auto_branch_scores],
            "Endpoints & Wrinkle",
            ["Endpoints", "Wrinkle", "Branch"],
            ["purple", "orange", "red"],
            width=460,
            height=260,
        )
        self._score_graph_photo = ImageTk.PhotoImage(score_img)
        self._best_graph_photo = ImageTk.PhotoImage(best_img)
        self._metric_graph_photo = ImageTk.PhotoImage(metric_img)
        self._detail_graph_photo = ImageTk.PhotoImage(detail_img)
        self.score_graph_label.config(image=self._score_graph_photo)
        self.best_graph_label.config(image=self._best_graph_photo)
        if self.metric_graph_label:
            self.metric_graph_label.config(image=self._metric_graph_photo)
        if self.detail_graph_label:
            self.detail_graph_label.config(image=self._detail_graph_photo)

    def _open_graph_window(self, kind):
        win = tk.Toplevel(self.root)
        win.title(f"Graph: {kind}")
        win.geometry("980x640")

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(frame, background="white")
        h_scroll = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        v_scroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        mode_label = self.score_display_mode.get() if self.score_display_mode else "raw"
        display_scores = [self._score_to_display(v) for v in self._auto_scores]
        display_best_series = [
            (t, self._score_to_display(v)) for t, v in self._auto_best_time_series
        ] or [(i, self._score_to_display(v)) for i, v in enumerate(self._auto_best_scores)]

        if kind == "score":
            base_img = self._render_graph(display_scores, f"Score ({mode_label})", width=900, height=520)
        elif kind == "best":
            base_img = self._render_time_graph(
                display_best_series, f"Best score ({mode_label})", width=900, height=520
            )
        elif kind == "metric":
            base_img = self._render_multi_graph(
                [self._auto_cont_scores, self._auto_band_scores],
                "Continuity & Band Fit",
                ["Continuity", "Band fit"],
                ["green", "blue"],
                width=900,
                height=520,
            )
        else:
            base_img = self._render_multi_graph(
                [self._auto_endpoint_scores, self._auto_wrinkle_scores, self._auto_branch_scores],
                "Endpoints & Wrinkle",
                ["Endpoints", "Wrinkle", "Branch"],
                ["purple", "orange", "red"],
                width=900,
                height=520,
            )

        state = {"scale": 1.0, "image": base_img, "photo": None}

        def redraw():
            scale = state["scale"]
            w, h = state["image"].size
            scaled = state["image"].resize((int(w * scale), int(h * scale)), resample=Image.BILINEAR)
            state["photo"] = ImageTk.PhotoImage(scaled)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=state["photo"])
            canvas.configure(scrollregion=(0, 0, scaled.size[0], scaled.size[1]))

        def zoom(factor):
            state["scale"] = max(0.3, min(6.0, state["scale"] * factor))
            redraw()

        def on_wheel(event):
            if event.delta > 0:
                zoom(1.1)
            elif event.delta < 0:
                zoom(0.9)

        def on_button4(_event):
            zoom(1.1)

        def on_button5(_event):
            zoom(0.9)

        def start_pan(event):
            canvas.scan_mark(event.x, event.y)

        def drag_pan(event):
            canvas.scan_dragto(event.x, event.y, gain=1)

        canvas.bind("<MouseWheel>", on_wheel)
        canvas.bind("<Button-4>", on_button4)
        canvas.bind("<Button-5>", on_button5)
        canvas.bind("<ButtonPress-1>", start_pan)
        canvas.bind("<B1-Motion>", drag_pan)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, pady=4)
        ttk.Button(btn_frame, text="Zoom In", command=lambda: zoom(1.2)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Zoom Out", command=lambda: zoom(0.85)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Reset", command=lambda: (state.update({"scale": 1.0}), redraw())).pack(side=tk.LEFT, padx=4)

        redraw()

    def _choose_output_dir(self):
        selected = filedialog.askdirectory(title="Select output folder")
        if selected:
            self.output_root = selected
            self.output_label.config(text=self.output_root)

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="Select image files (up to 500)",
            filetypes=[
                ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if not files:
            return

        remaining = self.max_files - len(self.selected_files)
        if remaining <= 0:
            messagebox.showwarning("Limit Reached", "You already selected 500 files.")
            return

        files = list(files)
        if len(files) > remaining:
            messagebox.showwarning("File Limit", f"Only {remaining} more files can be added.")
            files = files[:remaining]

        for path in files:
            if path in self.selected_files:
                continue
            self.selected_files.append(path)
            roi = self.roi_cache.get(path)
            if roi is None:
                roi = self.roi_cache.get(os.path.basename(path))
            if roi:
                self.roi_map[path] = tuple(roi)
        self._refresh_file_list()

    def _clear_files(self):
        self.selected_files = []
        self.file_listbox.delete(0, tk.END)
        self.status_var.set("Idle...")

    def _start_processing(self):
        if not self.selected_files:
            messagebox.showinfo("Info", "Select files to process first.")
            return
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showwarning("In Progress", "Processing is already running.")
            return

        try:
            settings = self._get_param_settings()
        except ValueError as exc:
            messagebox.showerror("Parameter Error", str(exc))
            return

        batch_dir = self._create_batch_output_dir()
        self.status_var.set(f"Processing started... (Output: {batch_dir})")
        self.start_button.config(state=tk.DISABLED)
        self.auto_button.config(state=tk.DISABLED)

        self._worker_thread = threading.Thread(
            target=self._process_batch,
            args=(batch_dir, list(self.selected_files), settings),
            daemon=True,
        )
        self._worker_thread.start()
        self.root.after(100, self._poll_messages)

    def _create_batch_output_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(self.output_root, f"edge_results_{timestamp}")
        candidate = base_dir
        counter = 1
        while os.path.exists(candidate):
            candidate = f"{base_dir}_{counter:02d}"
            counter += 1
        os.makedirs(candidate, exist_ok=True)
        return candidate

    def _process_batch(self, batch_dir, files, settings):
        total = len(files)
        for idx, path in enumerate(files, start=1):
            try:
                process_image_file(path, batch_dir, self.detector, settings)
                self._message_queue.put(("progress", idx, total, os.path.basename(path)))
            except Exception as exc:
                self._message_queue.put(("error", path, str(exc)))
        self._message_queue.put(("done", batch_dir))

    def _poll_messages(self):
        try:
            while True:
                msg = self._message_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass

        if self._worker_thread and self._worker_thread.is_alive():
            self.root.after(120, self._poll_messages)
        else:
            self.start_button.config(state=tk.NORMAL)
            if self.auto_button:
                self.auto_button.config(state=tk.NORMAL)
            if self.pause_button:
                self.pause_button.config(state=tk.DISABLED, text="Pause")
            if self.stop_button:
                self.stop_button.config(state=tk.DISABLED)

    def _handle_message(self, msg):
        msg_type = msg[0]
        if msg_type == "progress":
            idx, total, name = msg[1], msg[2], msg[3]
            self.status_var.set(f"Processing... ({idx}/{total}) {name}")
        elif msg_type == "error":
            path, detail = msg[1], msg[2]
            messagebox.showerror("Processing Failed", f"{path}\n{detail}")
        elif msg_type == "done":
            batch_dir = msg[1]
            self.status_var.set(f"Done! Output: {batch_dir}")
        elif msg_type == "auto_progress":
            idx, total, score, best_score = msg[1], msg[2], msg[3], msg[4]
            eta_seconds = msg[5] if len(msg) > 5 else None
            phase = msg[6] if len(msg) > 6 else "coarse"
            elapsed = msg[7] if len(msg) > 7 else None
            summary = msg[8] if len(msg) > 8 else None
            qualities = msg[9] if len(msg) > 9 else None
            penalty = msg[11] if len(msg) > 11 else None
            score_disp = self._score_to_display(score)
            best_disp = self._score_to_display(best_score)
            self._auto_scores.append(score)
            self._auto_best_scores.append(best_score)
            if summary:
                if qualities:
                    self._auto_cont_scores.append(float(qualities.get("q_cont", 0.0)))
                    self._auto_band_scores.append(float(qualities.get("q_band", 0.0)))
                if penalty is not None:
                    self._auto_penalty_scores.append(float(penalty))
                self._auto_wrinkle_scores.append(float(summary.get("wrinkle", 0.0)))
                self._auto_endpoint_scores.append(float(summary.get("endpoints", 0.0)))
                self._auto_branch_scores.append(float(summary.get("branch", 0.0)))
            self._refresh_auto_graphs()
            eta_text = f" ETA {self._format_eta(eta_seconds)}" if eta_seconds is not None else ""
            phase_text = f"{phase} " if phase else ""
            score_str = self._format_score_for_display(score, None)
            best_str = self._format_score_for_display(best_score, None)
            self.status_var.set(
                f"Auto optimizing... ({phase_text}{idx}/{total}) score={score_str} "
                f"best={best_str}{eta_text}"
            )
        elif msg_type == "auto_best":
            best_score, elapsed = msg[1], msg[2]
            best_str = self._format_score_for_display(best_score, None)
            minutes = elapsed / 60.0
            self._log(f"[AUTO] Best improved to {best_str} at {minutes:.1f} min")
        elif msg_type == "auto_done":
            settings, report_path = msg[1], msg[2]
            stop_reason = msg[3] if len(msg) > 3 else None
            best_score = msg[4] if len(msg) > 4 else None
            if best_score is not None:
                self._last_auto_best_score = best_score
                self._update_auto_score_label()
            if settings:
                self._apply_settings(settings)
            if stop_reason:
                label = "stopped by user" if stop_reason == "stopped" else stop_reason
                self.status_var.set(f"Auto optimization stopped ({label}). Report: {report_path}")
                self._log(f"[AUTO] Stopped ({label}): {report_path}")
            else:
                self.status_var.set(f"Auto optimization complete! Report: {report_path}")
                self._log(f"[AUTO] Completed: {report_path}")


def main():
    if tk is None:
        raise RuntimeError("tkinter is not installed. Install it to run the GUI.")
    root = tk.Tk()
    EdgeBatchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
