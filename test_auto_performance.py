"""
Auto Optimization Performance Test Framework
체계적인 성능 측정 및 비교를 위한 테스트 스크립트
"""
import os
import time
import json
import numpy as np
from PIL import Image
import tempfile
import shutil
from datetime import datetime
from sobel_edge_detection import SobelEdgeDetector, AUTO_DEFAULTS, PARAM_DEFAULTS


class AutoPerformanceTester:
    def __init__(self, test_dir="test_images_auto"):
        self.test_dir = test_dir
        self.detector = SobelEdgeDetector()
        self.results = []
        
    def generate_test_images(self, count=10, size=(400, 300)):
        """테스트용 이미지 생성: 다양한 패턴과 난이도"""
        os.makedirs(self.test_dir, exist_ok=True)
        generated = []
        
        print(f"[TEST] Generating {count} test images...")
        
        for i in range(count):
            # 다양한 패턴 생성
            if i % 3 == 0:
                # 원형 객체 (쉬움)
                img = self._generate_circle_image(size)
            elif i % 3 == 1:
                # 사각형 객체 (중간)
                img = self._generate_rectangle_image(size)
            else:
                # 복잡한 형태 (어려움)
                img = self._generate_complex_image(size)
            
            path = os.path.join(self.test_dir, f"test_{i:03d}.png")
            Image.fromarray(img).save(path)
            generated.append(path)
            
        print(f"[TEST] Generated {len(generated)} images in {self.test_dir}/")
        return generated
    
    def _generate_circle_image(self, size):
        """원형 객체 이미지 생성"""
        h, w = size
        img = np.zeros((h, w), dtype=np.uint8)
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        img[mask] = 200  # 어두운 객체
        
        # 노이즈 추가
        noise = np.random.randint(-20, 20, (h, w), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def _generate_rectangle_image(self, size):
        """사각형 객체 이미지 생성"""
        h, w = size
        img = np.zeros((h, w), dtype=np.uint8)
        margin = min(h, w) // 5
        img[margin:h-margin, margin:w-margin] = 180
        
        # 노이즈 추가
        noise = np.random.randint(-30, 30, (h, w), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def _generate_complex_image(self, size):
        """복잡한 형태 이미지 생성"""
        h, w = size
        img = np.zeros((h, w), dtype=np.uint8)
        
        # 여러 객체
        for i in range(3):
            cy = h // 4 + (i * h // 4)
            cx = w // 2
            radius = min(h, w) // 8
            y, x = np.ogrid[:h, :w]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            img[mask] = 160 + i * 20
        
        # 노이즈 추가
        noise = np.random.randint(-40, 40, (h, w), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def measure_auto_optimization(self, files, mode="Fast", iterations=3):
        """Auto optimization 성능 측정"""
        results = []
        
        for iteration in range(iterations):
            print(f"\n[TEST] Iteration {iteration + 1}/{iterations}")
            print(f"[TEST] Mode: {mode}, Files: {len(files)}")
            
            # GUI 없이 직접 worker 호출을 위한 래퍼 필요
            # 대신 간단한 평가 루프로 측정
            start_time = time.time()
            
            # 간단한 후보 평가 테스트
            base_settings = dict(PARAM_DEFAULTS)
            auto_config = dict(AUTO_DEFAULTS)
            
            # 첫 번째 후보 평가 시간 측정
            first_eval_start = time.time()
            # 실제 평가는 GUI 클래스를 통해야 하므로 여기서는 시간만 측정
            first_eval_time = time.time() - first_eval_start
            
            # 전체 시간 측정 (실제로는 GUI를 통해야 함)
            total_time = time.time() - start_time
            
            result = {
                "iteration": iteration + 1,
                "mode": mode,
                "file_count": len(files),
                "first_eval_time": first_eval_time,
                "total_time": total_time,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            print(f"[TEST] First eval time: {first_eval_time:.3f}s")
            print(f"[TEST] Total time: {total_time:.3f}s")
        
        return results
    
    def save_results(self, results, filename="auto_performance_results.json"):
        """결과 저장"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[TEST] Results saved to {filename}")
    
    def compare_results(self, baseline_file, improved_file):
        """결과 비교"""
        with open(baseline_file, "r") as f:
            baseline = json.load(f)
        with open(improved_file, "r") as f:
            improved = json.load(f)
        
        print("\n[COMPARE] Performance Comparison:")
        print("=" * 60)
        
        baseline_avg = np.mean([r["total_time"] for r in baseline])
        improved_avg = np.mean([r["total_time"] for r in improved])
        
        speedup = baseline_avg / improved_avg if improved_avg > 0 else 0
        
        print(f"Baseline average time: {baseline_avg:.3f}s")
        print(f"Improved average time: {improved_avg:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        print("=" * 60)


def main():
    """메인 테스트 실행"""
    tester = AutoPerformanceTester()
    
    # 1. 테스트 이미지 생성
    print("=" * 60)
    print("STEP 1: Generate Test Images")
    print("=" * 60)
    test_files = tester.generate_test_images(count=10)
    
    # 2. 성능 측정 (실제로는 GUI를 통해야 하므로 여기서는 구조만 제공)
    print("\n" + "=" * 60)
    print("STEP 2: Performance Measurement")
    print("=" * 60)
    print("[NOTE] Actual measurement requires GUI execution.")
    print("[NOTE] Use manual testing with GUI for now.")
    
    # 결과 구조 예시
    example_results = [
        {
            "iteration": 1,
            "mode": "Fast",
            "file_count": 10,
            "first_eval_time": 0.0,
            "total_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }
    ]
    tester.save_results(example_results, "baseline_results.json")
    
    print("\n[TEST] Test framework ready!")
    print("[TEST] Next steps:")
    print("  1. Run GUI manually and measure times")
    print("  2. Save results to baseline_results.json")
    print("  3. Make code improvements")
    print("  4. Re-test and save to improved_results.json")
    print("  5. Run compare_results() to see improvements")


if __name__ == "__main__":
    main()
