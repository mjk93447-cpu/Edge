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
    
    def compute_gradient(self, image):
        """Sobel 필터로 그래디언트 계산"""
        # x, y 방향 그래디언트 계산
        grad_x = self.apply_convolution(image, self.sobel_x)
        grad_y = self.apply_convolution(image, self.sobel_y)
        
        # 그래디언트 크기와 방향 계산
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        return magnitude, direction, grad_x, grad_y
    
    def non_maximum_suppression(self, magnitude, direction):
        """Non-Maximum Suppression으로 얇은 에지 생성"""
        magnitude = magnitude.astype(np.float32, copy=False)
        suppressed = np.zeros_like(magnitude, dtype=np.float32)

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

        keep_0 = mask_0 & (center >= right) & (center >= left)
        keep_45 = mask_45 & (center >= up_right) & (center >= down_left)
        keep_90 = mask_90 & (center >= up) & (center >= down)
        keep_135 = mask_135 & (center >= up_left) & (center >= down_right)

        suppressed[keep_0 | keep_45 | keep_90 | keep_135] = center[
            keep_0 | keep_45 | keep_90 | keep_135
        ]

        suppressed[0, :] = 0
        suppressed[-1, :] = 0
        suppressed[:, 0] = 0
        suppressed[:, -1] = 0
        return suppressed
    
    def double_threshold(self, image, low_ratio=0.06, high_ratio=0.18):
        """이중 임계값 적용"""
        high_threshold = image.max() * high_ratio
        low_threshold = image.max() * low_ratio
        
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
    
    def detect_edges(
        self,
        image_path,
        use_nms=True,
        use_hysteresis=True,
        use_blur=True,
        blur_kernel_size=5,
        blur_sigma=1.2,
        low_ratio=0.06,
        high_ratio=0.18,
    ):
        """전체 에지 검출 파이프라인"""
        # 1. 이미지 로드
        image = self.load_image(image_path)
        original = image.copy()

        # 1-1. 블러로 노이즈 완화
        if use_blur:
            image = self.apply_gaussian_blur(image, blur_kernel_size, blur_sigma)
        
        # 2. Sobel 필터 적용
        magnitude, direction, grad_x, grad_y = self.compute_gradient(image)
        
        # 3. Non-Maximum Suppression (얇은 에지)
        if use_nms:
            edges = self.non_maximum_suppression(magnitude, direction)
        else:
            edges = magnitude
        
        # 4. 이중 임계값 및 에지 추적 (연결된 에지)
        if use_hysteresis:
            edges_threshold, weak, strong = self.double_threshold(
                edges, low_ratio=low_ratio, high_ratio=high_ratio
            )
            edges_final = self.edge_tracking(edges_threshold, weak, strong)
        else:
            # 단순 임계값
            threshold = edges.max() * 0.15
            edges_final = np.where(edges > threshold, 255, 0)
        
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
