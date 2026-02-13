"""
Automated Auto Optimization Test System
GUI 없이 직접 auto optimization을 실행하고 성능을 측정하는 시스템
"""
import os
import sys
import time
import json
import numpy as np
from PIL import Image
from datetime import datetime
from sobel_edge_detection import (
    SobelEdgeDetector, AUTO_DEFAULTS, PARAM_DEFAULTS,
    compute_auto_score, evaluate_one_candidate_mp
)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing


class AutomatedAutoTester:
    def __init__(self, test_dir="test_images_auto"):
        self.test_dir = test_dir
        self.detector = SobelEdgeDetector()
        self.results_history = []
        
    def generate_test_images(self, count=10, size=(400, 300)):
        """다양한 패턴의 테스트 이미지 생성"""
        os.makedirs(self.test_dir, exist_ok=True)
        generated = []
        
        print(f"[GEN] Generating {count} test images ({size[0]}x{size[1]})...")
        
        for i in range(count):
            if i % 3 == 0:
                img = self._generate_circle_image(size)
            elif i % 3 == 1:
                img = self._generate_rectangle_image(size)
            else:
                img = self._generate_complex_image(size)
            
            path = os.path.join(self.test_dir, f"test_{i:03d}.png")
            Image.fromarray(img).save(path)
            generated.append(path)
            
        print(f"[GEN] Generated {len(generated)} images")
        return generated
    
    def _generate_circle_image(self, size):
        h, w = size
        img = np.zeros((h, w), dtype=np.uint8)
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        img[mask] = 200
        noise = np.random.randint(-20, 20, (h, w), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def _generate_rectangle_image(self, size):
        h, w = size
        img = np.zeros((h, w), dtype=np.uint8)
        margin = min(h, w) // 5
        img[margin:h-margin, margin:w-margin] = 180
        noise = np.random.randint(-30, 30, (h, w), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def _generate_complex_image(self, size):
        h, w = size
        img = np.zeros((h, w), dtype=np.uint8)
        for i in range(3):
            cy = h // 4 + (i * h // 4)
            cx = w // 2
            radius = min(h, w) // 8
            y, x = np.ogrid[:h, :w]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            img[mask] = 160 + i * 20
        noise = np.random.randint(-40, 40, (h, w), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def prepare_auto_data(self, files, settings, auto_config, max_files=None):
        """Auto optimization용 데이터 준비 (GUI 없이)"""
        if max_files:
            files = files[:max_files]
        
        band_min = min(auto_config["auto_band_min"], auto_config["auto_band_max"])
        band_max = max(auto_config["auto_band_min"], auto_config["auto_band_max"])
        band_radii = list(range(band_min, band_max + 1))
        if not band_radii:
            band_radii = [settings["boundary_band_radius"]]
        
        def process_one_image(path):
            image = self.detector.load_image(path)
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
            
            return {
                "path": path,
                "image": image,
                "mask": mask,
                "boundary": boundary,
                "bands": bands,
                "band_pixels": band_pixels,
                "weight": 1.0,
            }
        
        # 병렬 처리
        data = []
        num_workers = min(max(1, (os.cpu_count() or 4) - 1), len(files))
        if num_workers > 1 and len(files) > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                data = list(executor.map(process_one_image, files))
        else:
            data = [process_one_image(f) for f in files]
        
        return {"coarse": data, "mid": data, "full": data}
    
    def _compute_boundary(self, mask):
        """경계 계산"""
        # 간단한 경계 계산 (erode 사용)
        h, w = mask.shape
        boundary = np.zeros_like(mask, dtype=bool)
        for y in range(1, h-1):
            for x in range(1, w-1):
                if mask[y, x]:
                    if not (mask[y-1, x] and mask[y+1, x] and mask[y, x-1] and mask[y, x+1]):
                        boundary[y, x] = True
        return boundary
    
    def test_auto_optimization_speed(self, files, mode="Fast", candidate_workers=0):
        """Auto optimization 속도 테스트"""
        print(f"\n[TEST] Mode: {mode}, Files: {len(files)}, Workers: {candidate_workers}")
        
        base_settings = dict(PARAM_DEFAULTS)
        auto_config = dict(AUTO_DEFAULTS)
        auto_config["auto_candidate_workers"] = candidate_workers
        
        # 데이터 준비 시간 측정
        prep_start = time.time()
        data = self.prepare_auto_data(files, base_settings, auto_config, 
                                      max_files=8 if mode == "Fast" else len(files))
        prep_time = time.time() - prep_start
        print(f"[TEST] Data preparation: {prep_time:.3f}s")
        
        data_full = data.get("full", [])
        if not data_full:
            return None
        
        # 첫 번째 후보 생성 및 평가
        rng = np.random.RandomState(42)
        target_eval = 3000 if mode == "Fast" else 9000
        round_budget = 200 if mode == "Fast" else 500
        
        # 첫 번째 라운드 후보 생성
        first_round_start = time.time()
        n_explore = min(round_budget // 4, 80) if mode == "Fast" else min(round_budget // 4, 80)
        
        # 간단한 후보 생성 (실제 _build_candidates 대신)
        candidates = []
        for _ in range(n_explore):
            s = dict(base_settings)
            s["nms_relax"] = np.random.uniform(0.90, 0.97)
            s["high_ratio"] = np.random.uniform(0.07, 0.14)
            s["low_ratio"] = s["high_ratio"] * np.random.uniform(0.30, 0.42)
            s["boundary_band_radius"] = np.random.randint(1, 4)
            s["polarity_drop_margin"] = np.random.uniform(0.10, 0.45)
            candidates.append(s)
        
        pool_time = time.time() - first_round_start
        print(f"[TEST] Candidate pool generation: {pool_time:.3f}s")
        
        # 첫 번째 평가 시간 측정
        first_eval_start = time.time()
        first_score = None
        
        if candidate_workers >= 1:
            # 병렬 평가 - 모듈 레벨 함수 직접 사용
            batch_size = min(candidate_workers, 8)
            batch = candidates[:batch_size]
            try:
                from sobel_edge_detection import _eval_candidate_wrapper_mp
                
                with ProcessPoolExecutor(max_workers=len(batch)) as ex:
                    args_list = [(data_full, s, auto_config) for s in batch]
                    results = list(ex.map(_eval_candidate_wrapper_mp, args_list, chunksize=1))
                if results:
                    first_score = results[0][0]
            except Exception as e:
                print(f"[WARN] ProcessPool failed: {e}, using sequential")
                result = evaluate_one_candidate_mp(data_full, candidates[0], auto_config)
                first_score = result[0]
        else:
            # 순차 평가
            result = evaluate_one_candidate_mp(data_full, candidates[0], auto_config)
            first_score = result[0]
        
        first_eval_time = time.time() - first_eval_start
        print(f"[TEST] First evaluation: {first_eval_time:.3f}s (score: {first_score:.6e})")
        
        # 처음 10개 평가 시간
        batch10_start = time.time()
        if candidate_workers >= 1:
            batch = candidates[:min(10, len(candidates))]
            try:
                from sobel_edge_detection import _eval_candidate_wrapper_mp
                
                with ProcessPoolExecutor(max_workers=min(candidate_workers, len(batch))) as ex:
                    args_list = [(data_full, s, auto_config) for s in batch]
                    results = list(ex.map(_eval_candidate_wrapper_mp, args_list, chunksize=1))
            except Exception as e:
                print(f"[WARN] ProcessPool batch10 failed: {e}, using sequential")
                for s in batch:
                    evaluate_one_candidate_mp(data_full, s, auto_config)
        else:
            for s in candidates[:10]:
                evaluate_one_candidate_mp(data_full, s, auto_config)
        
        batch10_time = time.time() - batch10_start
        print(f"[TEST] First 10 evaluations: {batch10_time:.3f}s")
        
        total_time = time.time() - prep_start
        
        result = {
            "mode": mode,
            "file_count": len(files),
            "candidate_workers": candidate_workers,
            "prep_time": prep_time,
            "pool_gen_time": pool_time,
            "first_eval_time": first_eval_time,
            "batch10_time": batch10_time,
            "total_time": total_time,
            "first_score": float(first_score) if first_score else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def run_comparison_test(self, files, iterations=3):
        """여러 설정으로 비교 테스트"""
        print("\n" + "="*60)
        print("COMPARISON TEST")
        print("="*60)
        
        all_results = []
        
        # 테스트 케이스
        test_cases = [
            ("Fast", 0),   # 순차
            ("Fast", 4),   # 병렬 4 workers
        ]
        
        for mode, workers in test_cases:
            for i in range(iterations):
                print(f"\n[TEST] Iteration {i+1}/{iterations}: {mode}, workers={workers}")
                result = self.test_auto_optimization_speed(files, mode=mode, candidate_workers=workers)
                if result:
                    result["iteration"] = i + 1
                    all_results.append(result)
                time.sleep(0.5)  # 간격
        
        return all_results
    
    def analyze_results(self, results):
        """결과 분석 및 리포트"""
        print("\n" + "="*60)
        print("RESULTS ANALYSIS")
        print("="*60)
        
        # 그룹별 평균 계산
        groups = {}
        for r in results:
            key = (r["mode"], r["candidate_workers"])
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        
        for (mode, workers), group_results in groups.items():
            avg_prep = np.mean([r["prep_time"] for r in group_results])
            avg_first_eval = np.mean([r["first_eval_time"] for r in group_results])
            avg_batch10 = np.mean([r["batch10_time"] for r in group_results])
            avg_total = np.mean([r["total_time"] for r in group_results])
            
            print(f"\n[{mode}, workers={workers}]:")
            print(f"  Avg prep time: {avg_prep:.3f}s")
            print(f"  Avg first eval: {avg_first_eval:.3f}s")
            print(f"  Avg batch10: {avg_batch10:.3f}s")
            print(f"  Avg total: {avg_total:.3f}s")
        
        # 속도 개선 계산
        if len(groups) >= 2:
            sequential = groups.get(("Fast", 0), [])
            parallel = groups.get(("Fast", 4), [])
            
            if sequential and parallel:
                seq_avg = np.mean([r["first_eval_time"] for r in sequential])
                par_avg = np.mean([r["first_eval_time"] for r in parallel])
                speedup = seq_avg / par_avg if par_avg > 0 else 0
                print(f"\n[SPEEDUP] First eval: {speedup:.2f}x")
                
                seq_batch = np.mean([r["batch10_time"] for r in sequential])
                par_batch = np.mean([r["batch10_time"] for r in parallel])
                batch_speedup = seq_batch / par_batch if par_batch > 0 else 0
                print(f"[SPEEDUP] Batch10: {batch_speedup:.2f}x")
        
        return groups


def main():
    """메인 테스트 실행"""
    tester = AutomatedAutoTester()
    
    # 테스트 이미지 생성
    print("="*60)
    print("STEP 1: Generate Test Images")
    print("="*60)
    test_files = tester.generate_test_images(count=10, size=(400, 300))
    
    # 비교 테스트 실행
    print("\n" + "="*60)
    print("STEP 2: Run Comparison Tests")
    print("="*60)
    results = tester.run_comparison_test(test_files, iterations=3)
    
    # 결과 분석
    groups = tester.analyze_results(results)
    
    # 결과 저장
    output_file = f"automated_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[TEST] Results saved to {output_file}")
    
    return results, groups


if __name__ == "__main__":
    # Windows multiprocessing 지원
    if sys.platform == "win32":
        multiprocessing.freeze_support()
    
    main()
