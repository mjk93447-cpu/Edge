import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

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
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        # 패딩 추가
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        result = np.zeros_like(image)
        
        # 컨볼루션 수행
        for i in range(h):
            for j in range(w):
                region = padded[i:i+k_h, j:j+k_w]
                result[i, j] = np.sum(region * kernel)
        
        return result
    
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
        h, w = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        # 각도를 0, 45, 90, 135도로 양자화
        angle = direction * 180.0 / np.pi
        angle[angle < 0] += 180
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                q = 255
                r = 255
                
                # 0도 (수평)
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = magnitude[i, j+1]
                    r = magnitude[i, j-1]
                # 45도
                elif 22.5 <= angle[i,j] < 67.5:
                    q = magnitude[i+1, j-1]
                    r = magnitude[i-1, j+1]
                # 90도 (수직)
                elif 67.5 <= angle[i,j] < 112.5:
                    q = magnitude[i+1, j]
                    r = magnitude[i-1, j]
                # 135도
                elif 112.5 <= angle[i,j] < 157.5:
                    q = magnitude[i-1, j-1]
                    r = magnitude[i+1, j+1]
                
                # 현재 픽셀이 이웃보다 크면 유지, 아니면 억제
                if magnitude[i,j] >= q and magnitude[i,j] >= r:
                    suppressed[i,j] = magnitude[i,j]
                else:
                    suppressed[i,j] = 0
        
        return suppressed
    
    def double_threshold(self, image, low_ratio=0.05, high_ratio=0.15):
        """이중 임계값 적용"""
        high_threshold = image.max() * high_ratio
        low_threshold = high_threshold * low_ratio
        
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
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                if result[i, j] == weak:
                    # 주변 8개 픽셀 중 강한 에지가 있으면 연결
                    if ((result[i+1, j-1] == strong) or (result[i+1, j] == strong) or 
                        (result[i+1, j+1] == strong) or (result[i, j-1] == strong) or 
                        (result[i, j+1] == strong) or (result[i-1, j-1] == strong) or 
                        (result[i-1, j] == strong) or (result[i-1, j+1] == strong)):
                        result[i, j] = strong
                    else:
                        result[i, j] = 0
        
        return result
    
    def detect_edges(self, image_path, use_nms=True, use_hysteresis=True):
        """전체 에지 검출 파이프라인"""
        # 1. 이미지 로드
        image = self.load_image(image_path)
        
        # 2. Sobel 필터 적용
        magnitude, direction, grad_x, grad_y = self.compute_gradient(image)
        
        # 3. Non-Maximum Suppression (얇은 에지)
        if use_nms:
            edges = self.non_maximum_suppression(magnitude, direction)
        else:
            edges = magnitude
        
        # 4. 이중 임계값 및 에지 추적 (연결된 에지)
        if use_hysteresis:
            edges_threshold, weak, strong = self.double_threshold(edges)
            edges_final = self.edge_tracking(edges_threshold, weak, strong)
        else:
            # 단순 임계값
            threshold = edges.max() * 0.15
            edges_final = np.where(edges > threshold, 255, 0)
        
        return {
            'original': image,
            'magnitude': magnitude,
            'grad_x': grad_x,
            'grad_y': grad_y,
            'edges': edges_final
        }
    
    def visualize_results(self, results, save_path=None):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Sobel Edge Detection Results', fontsize=16)
        
        # 원본 이미지
        axes[0, 0].imshow(results['original'], cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # X 방향 그래디언트
        axes[0, 1].imshow(results['grad_x'], cmap='gray')
        axes[0, 1].set_title('Gradient X (Sobel X)')
        axes[0, 1].axis('off')
        
        # Y 방향 그래디언트
        axes[0, 2].imshow(results['grad_y'], cmap='gray')
        axes[0, 2].set_title('Gradient Y (Sobel Y)')
        axes[0, 2].axis('off')
        
        # 그래디언트 크기
        axes[1, 0].imshow(results['magnitude'], cmap='gray')
        axes[1, 0].set_title('Gradient Magnitude')
        axes[1, 0].axis('off')
        
        # 최종 에지
        axes[1, 1].imshow(results['edges'], cmap='gray')
        axes[1, 1].set_title('Final Edges (Thin & Connected)')
        axes[1, 1].axis('off')
        
        # 에지 오버레이
        overlay = np.stack([results['original']]*3, axis=-1) / 255.0
        edge_mask = results['edges'] > 0
        overlay[edge_mask] = [1, 0, 0]  # 빨간색으로 에지 표시
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Edge Overlay')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"결과 이미지 저장됨: {save_path}")
        
        plt.show()


def create_sample_image():
    """테스트용 샘플 이미지 생성"""
    img = np.ones((300, 300)) * 255
    
    # 사각형
    img[50:150, 50:150] = 0
    
    # 원
    y, x = np.ogrid[:300, :300]
    circle_mask = (x - 220)**2 + (y - 80)**2 <= 40**2
    img[circle_mask] = 0
    
    # 삼각형
    for i in range(100):
        img[180+i, 50+i:150-i] = 0
    
    return img.astype(np.uint8)


# 메인 실행 코드
if __name__ == "__main__":
    # Sobel Edge Detector 인스턴스 생성
    detector = SobelEdgeDetector()
    
    # 샘플 이미지 생성 및 저장
    sample_img = create_sample_image()
    sample_path = "/home/claude/sample_image.png"
    Image.fromarray(sample_img).save(sample_path)
    print(f"샘플 이미지 생성됨: {sample_path}")
    
    # 에지 검출 실행
    print("\n에지 검출 시작...")
    results = detector.detect_edges(
        sample_path,
        use_nms=True,        # Non-Maximum Suppression (얇은 에지)
        use_hysteresis=True  # Hysteresis (연결된 에지)
    )
    
    # 결과 저장
    output_path = "/mnt/user-data/outputs/sobel_edge_result.png"
    detector.visualize_results(results, save_path=output_path)
    
    # 최종 에지만 저장
    edge_only_path = "/mnt/user-data/outputs/edges_only.png"
    Image.fromarray(results['edges'].astype(np.uint8)).save(edge_only_path)
    print(f"\n에지 이미지 저장됨: {edge_only_path}")
    
    print("\n프로그램 실행 완료!")
    print("\n[사용 방법]")
    print("detector = SobelEdgeDetector()")
    print("results = detector.detect_edges('your_image.png')")
    print("detector.visualize_results(results)")
