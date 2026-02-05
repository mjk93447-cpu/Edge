import json
import os
import queue
import threading
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

PARAM_DEFAULTS = {
    "nms_relax": 0.95,
    "low_ratio": 0.04,
    "high_ratio": 0.12,
    "use_blur": True,
    "blur_sigma": 1.2,
    "blur_kernel_size": 5,
    "auto_threshold": True,
    "contrast_ref": 80.0,
    "min_threshold_scale": 0.5,
    "use_thinning": True,
    "thinning_max_iter": 15,
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

AUTO_DEFAULTS = {
    "auto_nms_min": 0.90,
    "auto_nms_max": 1.00,
    "auto_nms_step": 0.02,
    "auto_high_min": 0.08,
    "auto_high_max": 0.16,
    "auto_high_step": 0.02,
    "auto_low_factor": 0.33,
    "auto_band_min": 1,
    "auto_band_max": 3,
    "auto_margin_min": 0.0,
    "auto_margin_max": 0.8,
    "auto_margin_step": 0.2,
    "weight_coverage": 1.0,
    "weight_intrusion": 1.2,
    "weight_outside": 0.7,
    "weight_gap": 1.0,
    "weight_thickness": 0.5,
    "weight_low_quality": 0.5,
}

PARAM_KEYS = tuple(PARAM_DEFAULTS.keys())
AUTO_KEYS = tuple(AUTO_DEFAULTS.keys())


def save_json_config(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def load_json_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

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
        use_thinning=True,
        thinning_max_iter=15,
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

        if use_thinning:
            edge_mask = edges_final > 0
            edge_mask = self.thin_edges_zhang_suen(edge_mask, thinning_max_iter)
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
        self.root.title("Sobel Edge Batch Processor")
        self.root.geometry("920x720")

        self.detector = SobelEdgeDetector()
        self.selected_files = []
        self.roi_map = {}
        self.max_files = 100
        self.output_root = os.path.abspath("outputs")

        self._message_queue = queue.Queue()
        self._worker_thread = None
        self._auto_thread = None
        self.param_vars = self._init_param_vars()
        self.auto_mode = tk.StringVar(value="Fast")
        self.log_text = None
        self.score_graph_label = None
        self.best_graph_label = None
        self._auto_scores = []
        self._auto_best_scores = []

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
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(
            main_frame,
            text="Select up to 100 images and process sequentially.",
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

        ttk.Label(auto_frame, text="Low factor").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.2,
            to=0.6,
            increment=0.05,
            textvariable=self.param_vars["auto_low_factor"],
            width=6,
        ).grid(row=2, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Band min").grid(row=2, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0,
            to=6,
            increment=1,
            textvariable=self.param_vars["auto_band_min"],
            width=6,
        ).grid(row=2, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Band max").grid(row=2, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0,
            to=6,
            increment=1,
            textvariable=self.param_vars["auto_band_max"],
            width=6,
        ).grid(row=2, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Margin min").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.0,
            to=1.0,
            increment=0.1,
            textvariable=self.param_vars["auto_margin_min"],
            width=6,
        ).grid(row=3, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Margin max").grid(row=3, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.0,
            to=1.0,
            increment=0.1,
            textvariable=self.param_vars["auto_margin_max"],
            width=6,
        ).grid(row=3, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="Margin step").grid(row=3, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.1,
            to=1.0,
            increment=0.1,
            textvariable=self.param_vars["auto_margin_step"],
            width=6,
        ).grid(row=3, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W coverage").grid(row=4, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.5,
            to=2.0,
            increment=0.1,
            textvariable=self.param_vars["weight_coverage"],
            width=6,
        ).grid(row=4, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W gap").grid(row=4, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.5,
            to=3.0,
            increment=0.1,
            textvariable=self.param_vars["weight_gap"],
            width=6,
        ).grid(row=4, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W intrusion").grid(row=4, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.5,
            to=3.0,
            increment=0.1,
            textvariable=self.param_vars["weight_intrusion"],
            width=6,
        ).grid(row=4, column=5, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W outside").grid(row=5, column=0, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.5,
            to=3.0,
            increment=0.1,
            textvariable=self.param_vars["weight_outside"],
            width=6,
        ).grid(row=5, column=1, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W thickness").grid(row=5, column=2, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.0,
            to=2.0,
            increment=0.1,
            textvariable=self.param_vars["weight_thickness"],
            width=6,
        ).grid(row=5, column=3, sticky="w", padx=(4, 12))

        ttk.Label(auto_frame, text="W low quality").grid(row=5, column=4, sticky="w")
        ttk.Spinbox(
            auto_frame,
            from_=0.0,
            to=2.0,
            increment=0.1,
            textvariable=self.param_vars["weight_low_quality"],
            width=6,
        ).grid(row=5, column=5, sticky="w", padx=(4, 12))

        auto_btn_frame = ttk.Frame(auto_frame)
        auto_btn_frame.grid(row=6, column=0, columnspan=6, sticky="w", pady=(4, 0))
        ttk.Button(auto_btn_frame, text="Save Auto Config", command=self._save_auto_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(auto_btn_frame, text="Load Auto Config", command=self._load_auto_config).pack(side=tk.LEFT, padx=4)

        param_btn_frame = ttk.Frame(param_frame)
        param_btn_frame.grid(row=6, column=0, columnspan=6, sticky="w", pady=(4, 0))
        ttk.Button(param_btn_frame, text="Save Params", command=self._save_param_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(param_btn_frame, text="Load Params", command=self._load_param_config).pack(side=tk.LEFT, padx=4)

        guide_frame = ttk.LabelFrame(main_frame, text="Tuning Guide")
        guide_frame.pack(fill=tk.X, pady=(6, 6))
        guide_text = tk.Text(guide_frame, height=9, wrap="word")
        guide_text.insert(
            "1.0",
            "- ROI: Draw a rectangle to focus auto-optimization on the true boundary area.\n"
            "- Auto Optimize uses the average score across multiple images (Precise mode uses all).\n"
            "- NMS relax: Lower reduces gaps, too low can thicken edges or create inner curl.\n"
            "- Low/High ratio: Lower increases recall but can add intrusion/noise.\n"
            "- Boundary band radius: Smaller forces edges to stay on the outer boundary.\n"
            "- Polarity filter: Removes inner curves by checking edge brightness direction.\n"
            "- Thinning: Keeps a single-pixel contour; adjust NMS relax if gaps appear.\n"
            "- Auto threshold: Adapts to low-contrast images automatically.\n"
            "- Weights: Raise intrusion/outside to penalize inner or external edges.\n"
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
            values=["Fast", "Precise"],
            width=6,
            state="readonly",
        ).pack(side=tk.LEFT)
        self.auto_button = ttk.Button(button_frame, text="Auto Optimize", command=self._start_auto_optimize)
        self.auto_button.pack(side=tk.LEFT, padx=6)
        self.start_button = ttk.Button(button_frame, text="Start Processing", command=self._start_processing)
        self.start_button.pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Idle...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(anchor="w", pady=(6, 0))

    def _log(self, message):
        if not self.log_text:
            return
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def _get_selected_file(self):
        selection = self.file_listbox.curselection()
        if not selection:
            return None
        return self.file_listbox.get(selection[0])

    def _open_roi_editor(self):
        path = self._get_selected_file()
        if not path:
            messagebox.showinfo("Info", "Select a file to set ROI.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Error", "File not found.")
            return

        img = Image.open(path).convert("L")
        max_w, max_h = 800, 500
        scale = min(max_w / img.width, max_h / img.height, 1.0)
        disp_w = int(img.width * scale)
        disp_h = int(img.height * scale)
        display = img.resize((disp_w, disp_h), resample=Image.BILINEAR)

        win = tk.Toplevel(self.root)
        win.title("Set ROI")
        canvas = tk.Canvas(win, width=disp_w, height=disp_h)
        canvas.pack()
        tk_img = ImageTk.PhotoImage(display)
        canvas.image = tk_img
        canvas.create_image(0, 0, anchor="nw", image=tk_img)

        roi_rect = None
        start = {"x": 0, "y": 0}
        roi_label = ttk.Label(win, text="Drag to draw a rectangular ROI.")
        roi_label.pack(pady=4)

        if path in self.roi_map:
            x1, y1, x2, y2 = self.roi_map[path]
            x1d, y1d = x1 * scale, y1 * scale
            x2d, y2d = x2 * scale, y2 * scale
            roi_rect = canvas.create_rectangle(x1d, y1d, x2d, y2d, outline="red", width=2)

        def on_press(event):
            start["x"], start["y"] = event.x, event.y
            nonlocal roi_rect
            if roi_rect:
                canvas.delete(roi_rect)
                roi_rect = None

        def on_drag(event):
            nonlocal roi_rect
            if roi_rect:
                canvas.delete(roi_rect)
            roi_rect = canvas.create_rectangle(
                start["x"], start["y"], event.x, event.y, outline="red", width=2
            )

        def on_release(event):
            nonlocal roi_rect
            if not roi_rect:
                return
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
            roi_label.config(text=f"ROI saved: ({x1},{y1})-({x2},{y2})")

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)

        def clear_roi():
            if path in self.roi_map:
                del self.roi_map[path]
            if roi_rect:
                canvas.delete(roi_rect)
            roi_label.config(text="ROI cleared.")

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=6)
        ttk.Button(btn_frame, text="Clear ROI", command=clear_roi).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.LEFT, padx=4)

    def _clear_roi(self):
        path = self._get_selected_file()
        if not path:
            messagebox.showinfo("Info", "Select a file to clear ROI.")
            return
        if path in self.roi_map:
            del self.roi_map[path]
            self._log(f"[ROI] Cleared: {os.path.basename(path)}")
        else:
            messagebox.showinfo("Info", "No ROI is set.")

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
            "use_blur": bool(self.param_vars["use_blur"].get()),
            "blur_sigma": get_float("blur_sigma"),
            "blur_kernel_size": max(3, get_int("blur_kernel_size")),
            "auto_threshold": bool(self.param_vars["auto_threshold"].get()),
            "contrast_ref": get_float("contrast_ref"),
            "min_threshold_scale": get_float("min_threshold_scale"),
            "use_thinning": bool(self.param_vars["use_thinning"].get()),
            "thinning_max_iter": max(1, get_int("thinning_max_iter")),
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
        return self._collect_values(AUTO_KEYS)

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
        self._apply_values(auto_values, AUTO_DEFAULTS)
        self._log(f"[AUTO] Config loaded: {path}")

    def _apply_settings(self, settings):
        for key, var in self.param_vars.items():
            if key in settings:
                var.set(settings[key])

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

    def _evaluate_settings(self, files, settings, max_files, auto_config, roi_map):
        if max_files >= len(files):
            subset = list(files)
        else:
            step = max(1, len(files) // max_files)
            subset = list(files)[::step][:max_files]
        total_score = 0.0
        total_coverage = 0.0
        total_intrusion = 0.0
        total_outside = 0.0
        total_gap = 0.0
        total_thickness = 0.0
        total_continuity = 0.0
        total_edges = 0

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
            band = self.detector.dilate_binary(boundary, settings["boundary_band_radius"])

            results = self.detector.detect_edges_array(
                image,
                use_nms=True,
                use_hysteresis=True,
                **settings,
            )
            edges = results["edges"] > 0
            edge_pixels = int(edges.sum())
            if edge_pixels == 0:
                continue

            band_pixels = int(band.sum())
            edges_in_band = int((edges & band).sum())
            intrusion = int((edges & mask & ~band).sum())
            outside = int((edges & ~mask & ~band).sum())

            coverage = edges_in_band / band_pixels if band_pixels else 0.0
            intrusion_ratio = intrusion / edge_pixels
            outside_ratio = outside / edge_pixels
            gap_ratio = max(0.0, 1.0 - coverage)
            components = self._count_components(edges & band)
            components_penalty = max(0.0, components - 1) / max(1, edges_in_band)
            continuity_penalty = min(1.0, components_penalty * 5.0)
            edge_density = edge_pixels / max(band_pixels, 1)
            thickness_penalty = max(0.0, edge_density - 1.2)
            w_cov = auto_config["weight_coverage"]
            w_gap = auto_config["weight_gap"]
            w_intr = auto_config["weight_intrusion"]
            w_out = auto_config["weight_outside"]
            w_thick = auto_config["weight_thickness"]
            score = (
                w_cov * coverage
                - w_gap * (gap_ratio + continuity_penalty)
                - w_intr * intrusion_ratio
                - w_out * outside_ratio
                - w_thick * thickness_penalty
            )

            p10, p90 = np.percentile(image, [10, 90])
            contrast = max(float(p90 - p10), 1.0)
            if contrast < settings["contrast_ref"] * 0.6:
                score *= (1.0 + auto_config["weight_low_quality"])

            score = max(0.0, score)
            total_score += score
            total_coverage += coverage
            total_intrusion += intrusion_ratio
            total_outside += outside_ratio
            total_gap += gap_ratio
            total_thickness += thickness_penalty
            total_continuity += continuity_penalty
            total_edges += 1

        if total_edges == 0:
            return 0.0, {"coverage": 0.0, "intrusion": 1.0, "outside": 1.0, "gap": 1.0, "thickness": 1.0}

        avg_score = max(0.0, total_score / total_edges)
        summary = {
            "coverage": total_coverage / total_edges,
            "intrusion": total_intrusion / total_edges,
            "outside": total_outside / total_edges,
            "gap": total_gap / total_edges,
            "thickness": total_thickness / total_edges,
            "continuity": total_continuity / total_edges,
        }
        return avg_score, summary

    def _build_candidates(self, base_settings, mode):
        auto_config = self._get_auto_config()

        def frange(start, stop, step):
            values = []
            if step <= 0:
                return values
            v = start
            while v <= stop + 1e-6:
                values.append(round(v, 4))
                v += step
            return values

        nms_list = frange(auto_config["auto_nms_min"], auto_config["auto_nms_max"], auto_config["auto_nms_step"])
        high_list = frange(auto_config["auto_high_min"], auto_config["auto_high_max"], auto_config["auto_high_step"])
        margin_list = frange(auto_config["auto_margin_min"], auto_config["auto_margin_max"], auto_config["auto_margin_step"])
        band_min = auto_config["auto_band_min"]
        band_max = auto_config["auto_band_max"]

        if mode == "Fast":
            nms_list = nms_list[::2] if len(nms_list) > 2 else nms_list
            high_list = high_list[::2] if len(high_list) > 2 else high_list
            margin_list = margin_list[::2] if len(margin_list) > 2 else margin_list

        candidates = []
        for nms_relax in nms_list:
            if nms_relax < 0.85 or nms_relax > 1.0:
                continue
            for high_ratio in high_list:
                low_ratio = max(0.02, high_ratio * auto_config["auto_low_factor"])
                for band_radius in range(band_min, band_max + 1):
                    for margin in margin_list:
                        settings = dict(base_settings)
                        settings.update(
                            {
                                "nms_relax": round(nms_relax, 3),
                                "high_ratio": float(high_ratio),
                                "low_ratio": float(low_ratio),
                                "boundary_band_radius": int(band_radius),
                                "polarity_drop_margin": float(margin),
                                "use_boundary_band_filter": band_radius > 0,
                            }
                        )
                        candidates.append(settings)
        return candidates

    def _start_auto_optimize(self):
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showwarning("In Progress", "Processing is already running.")
            return
        if not self.selected_files:
            self._add_files()
        if not self.selected_files:
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
        self._auto_scores = []
        self._auto_best_scores = []
        self._refresh_auto_graphs()
        self._worker_thread = threading.Thread(
            target=self._auto_optimize_worker,
            args=(list(self.selected_files), base_settings, auto_config, mode, dict(self.roi_map)),
            daemon=True,
        )
        self._worker_thread.start()
        self.root.after(100, self._poll_messages)

    def _auto_optimize_worker(self, files, base_settings, auto_config, mode, roi_map):
        candidates = self._build_candidates(base_settings, mode)
        max_files = min(8, len(files)) if mode == "Fast" else len(files)
        best = None
        best_score = 0.0
        report_lines = []
        scores = []
        best_progress = []

        for idx, settings in enumerate(candidates, start=1):
            score, summary = self._evaluate_settings(files, settings, max_files, auto_config, roi_map)
            report_lines.append(
                f"{idx:03d} score={score:.4f} coverage={summary['coverage']:.4f} gap={summary['gap']:.4f} "
                f"cont={summary['continuity']:.4f} intrusion={summary['intrusion']:.4f} outside={summary['outside']:.4f} "
                f"thickness={summary['thickness']:.4f} nms={settings['nms_relax']:.2f} low={settings['low_ratio']:.3f} "
                f"high={settings['high_ratio']:.3f} band={settings['boundary_band_radius']} "
                f"margin={settings['polarity_drop_margin']:.2f}"
            )
            scores.append(score)
            if best is None or score > best_score:
                best_score = score
                best = settings
            best_progress.append(best_score)
            self._message_queue.put(("auto_progress", idx, len(candidates), score, best_score))

        if best and mode != "Fast":
            refine_step = max(auto_config["auto_nms_step"] * 0.5, 0.005)
            high_step = max(auto_config["auto_high_step"] * 0.5, 0.005)
            margin_step = max(auto_config["auto_margin_step"] * 0.5, 0.05)
            refine_candidates = []
            for dnms in (-refine_step, 0.0, refine_step):
                for dhigh in (-high_step, 0.0, high_step):
                    for dband in (-1, 0, 1):
                        for dmargin in (-margin_step, 0.0, margin_step):
                            settings = dict(best)
                            settings["nms_relax"] = round(min(1.0, max(0.85, best["nms_relax"] + dnms)), 3)
                            settings["high_ratio"] = max(0.05, best["high_ratio"] + dhigh)
                            settings["low_ratio"] = max(0.02, settings["high_ratio"] * auto_config["auto_low_factor"])
                            settings["boundary_band_radius"] = max(0, best["boundary_band_radius"] + dband)
                            settings["polarity_drop_margin"] = max(0.0, best["polarity_drop_margin"] + dmargin)
                            settings["use_boundary_band_filter"] = settings["boundary_band_radius"] > 0
                            refine_candidates.append(settings)

            for idx, settings in enumerate(refine_candidates, start=1):
                score, summary = self._evaluate_settings(files, settings, max_files, auto_config, roi_map)
                report_lines.append(
                    f"refine {idx:03d} score={score:.4f} coverage={summary['coverage']:.4f} gap={summary['gap']:.4f} "
                    f"cont={summary['continuity']:.4f} intrusion={summary['intrusion']:.4f} "
                    f"outside={summary['outside']:.4f} thickness={summary['thickness']:.4f} "
                    f"nms={settings['nms_relax']:.2f} low={settings['low_ratio']:.3f} high={settings['high_ratio']:.3f} "
                    f"band={settings['boundary_band_radius']} margin={settings['polarity_drop_margin']:.2f}"
                )
                scores.append(score)
                if best is None or score > best_score:
                    best_score = score
                    best = settings
                best_progress.append(best_score)
                self._message_queue.put(("auto_progress", idx, len(refine_candidates), score, best_score))

        report_dir = self._create_batch_output_dir()
        report_path = os.path.join(report_dir, "auto_optimize_report.txt")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(report_lines))

        graph_path = os.path.join(report_dir, "auto_optimize_scores.png")
        best_path = os.path.join(report_dir, "auto_optimize_best.png")
        self._draw_score_graph(scores, graph_path, "Score by Candidate")
        self._draw_score_graph(best_progress, best_path, "Best Score Progress")

        config_path = os.path.join(report_dir, "auto_optimize_config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"best": best, "auto_config": auto_config, "base_settings": base_settings},
                handle,
                ensure_ascii=True,
                indent=2,
            )

        self._message_queue.put(("auto_done", best, report_path))

    def _render_graph(self, values, title, width=400, height=220):
        margin = 36
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        left = margin
        top = margin
        right = width - margin
        bottom = height - margin
        draw.rectangle([left, top, right, bottom], outline="black")
        draw.text((left, 8), title, fill="black")

        if not values:
            return img

        min_val = min(0.0, float(min(values)))
        max_val = float(max(values))
        if max_val <= min_val:
            max_val = min_val + 1.0

        def scale_x(idx):
            return left + idx * (right - left) / max(1, len(values) - 1)

        def scale_y(val):
            return bottom - (val - min_val) * (bottom - top) / (max_val - min_val)

        ticks = 4
        for i in range(ticks + 1):
            tx = left + i * (right - left) / ticks
            tick_val = int(round(1 + i * (len(values) - 1) / ticks))
            draw.line([(tx, bottom), (tx, bottom + 4)], fill="black")
            draw.text((tx - 6, bottom + 6), str(tick_val), fill="black")

            ty = bottom - i * (bottom - top) / ticks
            y_val = min_val + i * (max_val - min_val) / ticks
            draw.line([(left - 4, ty), (left, ty)], fill="black")
            draw.text((4, ty - 6), f"{y_val:.2f}", fill="black")

        points = [(scale_x(i), scale_y(v)) for i, v in enumerate(values)]
        if len(points) >= 2:
            draw.line(points, fill="blue", width=2)
        else:
            draw.ellipse([points[0][0] - 2, points[0][1] - 2, points[0][0] + 2, points[0][1] + 2], fill="blue")
        return img

    def _draw_score_graph(self, values, path, title):
        img = self._render_graph(values, title, width=800, height=300)
        img.save(path)

    def _refresh_auto_graphs(self):
        if not self.score_graph_label or not self.best_graph_label:
            return
        score_img = self._render_graph(self._auto_scores, "Score (current)", width=400, height=220)
        best_img = self._render_graph(self._auto_best_scores, "Best score", width=400, height=220)
        self._score_graph_photo = ImageTk.PhotoImage(score_img)
        self._best_graph_photo = ImageTk.PhotoImage(best_img)
        self.score_graph_label.config(image=self._score_graph_photo)
        self.best_graph_label.config(image=self._best_graph_photo)

    def _choose_output_dir(self):
        selected = filedialog.askdirectory(title="Select output folder")
        if selected:
            self.output_root = selected
            self.output_label.config(text=self.output_root)

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="Select image files (up to 100)",
            filetypes=[
                ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if not files:
            return

        remaining = self.max_files - len(self.selected_files)
        if remaining <= 0:
            messagebox.showwarning("Limit Reached", "You already selected 100 files.")
            return

        files = list(files)
        if len(files) > remaining:
            messagebox.showwarning("File Limit", f"Only {remaining} more files can be added.")
            files = files[:remaining]

        for path in files:
            self.selected_files.append(path)
            self.file_listbox.insert(tk.END, path)

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
            self._auto_scores.append(score)
            self._auto_best_scores.append(best_score)
            self._refresh_auto_graphs()
            self.status_var.set(
                f"Auto optimizing... ({idx}/{total}) score={score:.4f} best={best_score:.4f}"
            )
        elif msg_type == "auto_done":
            settings, report_path = msg[1], msg[2]
            if settings:
                self._apply_settings(settings)
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
