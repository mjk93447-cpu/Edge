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
from PIL import Image

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
    
    def detect_edges(
        self,
        image_path,
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
        # 1. 이미지 로드
        image = self.load_image(image_path)
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

def _make_overlay_image(original_gray, edges):
    """그레이스케일 원본 위에 초록색 에지 dot를 표시"""
    overlay = np.stack([original_gray] * 3, axis=-1)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    edge_mask = edges > 0
    overlay[edge_mask] = [0, 255, 0]
    return overlay, edge_mask


def process_image_file(image_path, output_dir, detector):
    """단일 이미지 처리 및 결과 저장"""
    results = detector.detect_edges(image_path, use_nms=True, use_hysteresis=True)
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
        self.root.geometry("720x480")

        self.detector = SobelEdgeDetector()
        self.selected_files = []
        self.max_files = 100
        self.output_root = os.path.abspath("outputs")

        self._message_queue = queue.Queue()
        self._worker_thread = None

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(
            main_frame,
            text="최대 100개 이미지 파일을 선택하여 연속 처리합니다.",
            font=("Arial", 12, "bold"),
        )
        header.pack(anchor="w")

        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=(10, 6))

        ttk.Label(output_frame, text="출력 폴더:").pack(side=tk.LEFT)
        self.output_label = ttk.Label(output_frame, text=self.output_root)
        self.output_label.pack(side=tk.LEFT, padx=(6, 12))
        ttk.Button(output_frame, text="폴더 변경", command=self._choose_output_dir).pack(side=tk.RIGHT)

        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.file_listbox = tk.Listbox(list_frame, height=12)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 6))

        ttk.Button(button_frame, text="파일 추가", command=self._add_files).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="목록 비우기", command=self._clear_files).pack(side=tk.LEFT, padx=6)
        self.start_button = ttk.Button(button_frame, text="처리 시작", command=self._start_processing)
        self.start_button.pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="대기 중...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(anchor="w", pady=(6, 0))

    def _choose_output_dir(self):
        selected = filedialog.askdirectory(title="출력 폴더 선택")
        if selected:
            self.output_root = selected
            self.output_label.config(text=self.output_root)

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="이미지 파일 선택 (최대 100개)",
            filetypes=[
                ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if not files:
            return

        remaining = self.max_files - len(self.selected_files)
        if remaining <= 0:
            messagebox.showwarning("제한 초과", "이미 최대 100개 파일이 선택되었습니다.")
            return

        files = list(files)
        if len(files) > remaining:
            messagebox.showwarning("파일 제한", f"{remaining}개까지만 추가됩니다.")
            files = files[:remaining]

        for path in files:
            self.selected_files.append(path)
            self.file_listbox.insert(tk.END, path)

    def _clear_files(self):
        self.selected_files = []
        self.file_listbox.delete(0, tk.END)
        self.status_var.set("대기 중...")

    def _start_processing(self):
        if not self.selected_files:
            messagebox.showinfo("알림", "처리할 파일을 먼저 선택해주세요.")
            return
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showwarning("진행 중", "현재 처리 중입니다.")
            return

        batch_dir = self._create_batch_output_dir()
        self.status_var.set(f"처리 시작... (저장 위치: {batch_dir})")
        self.start_button.config(state=tk.DISABLED)

        self._worker_thread = threading.Thread(
            target=self._process_batch,
            args=(batch_dir, list(self.selected_files)),
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

    def _process_batch(self, batch_dir, files):
        total = len(files)
        for idx, path in enumerate(files, start=1):
            try:
                process_image_file(path, batch_dir, self.detector)
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

    def _handle_message(self, msg):
        msg_type = msg[0]
        if msg_type == "progress":
            idx, total, name = msg[1], msg[2], msg[3]
            self.status_var.set(f"처리 중... ({idx}/{total}) {name}")
        elif msg_type == "error":
            path, detail = msg[1], msg[2]
            messagebox.showerror("처리 실패", f"{path}\n{detail}")
        elif msg_type == "done":
            batch_dir = msg[1]
            self.status_var.set(f"완료! 결과 저장 위치: {batch_dir}")


def main():
    if tk is None:
        raise RuntimeError("tkinter가 설치되어 있지 않습니다. GUI 실행을 위해 설치하세요.")
    root = tk.Tk()
    EdgeBatchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
