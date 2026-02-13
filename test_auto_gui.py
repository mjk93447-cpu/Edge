"""
Auto Optimization GUI Test Script
실제 GUI를 통해 auto optimization을 테스트하고 성능을 측정합니다.
"""
import os
import sys
import time
import json
from datetime import datetime
from sobel_edge_detection import EdgeBatchGUI
import tkinter as tk


class AutoOptimizationTester:
    def __init__(self):
        self.results = []
        self.test_images_dir = "test_images_auto"
        
    def run_test(self, image_files, mode="Fast", test_name="test1"):
        """GUI를 통해 auto optimization 테스트 실행"""
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"Mode: {mode}, Images: {len(image_files)}")
        print(f"{'='*60}")
        
        # GUI 생성
        root = tk.Tk()
        root.withdraw()  # 처음에는 숨김
        
        gui = EdgeBatchGUI(root)
        
        # 이미지 파일 추가
        gui.selected_files = list(image_files)
        for f in image_files:
            gui.roi_map[f] = None
        
        # Auto optimization 시작
        start_time = time.time()
        first_result_time = None
        first_graph_update_time = None
        
        # 콜백으로 시간 측정
        original_handle = gui._handle_message
        def timed_handle_message(msg):
            nonlocal first_result_time, first_graph_update_time
            current_time = time.time() - start_time
            
            if msg[0] == "auto_progress" and len(msg) > 3:
                if msg[3] > 0.0:  # 첫 번째 실제 결과
                    if first_result_time is None:
                        first_result_time = current_time
                        print(f"[TIMING] First result: {first_result_time:.3f}s")
            
            if msg[0] == "auto_progress":
                if first_graph_update_time is None:
                    first_graph_update_time = current_time
                    print(f"[TIMING] First graph update: {first_graph_update_time:.3f}s")
            
            return original_handle(msg)
        
        gui._handle_message = timed_handle_message
        
        # Auto optimization 시작
        print("[TEST] Starting auto optimization...")
        gui._start_auto_optimize()
        
        # 최대 대기 시간 (5분)
        max_wait = 300
        elapsed = 0
        check_interval = 0.5
        
        while elapsed < max_wait:
            root.update()
            time.sleep(check_interval)
            elapsed += check_interval
            
            # 완료 확인
            if gui._worker_thread and not gui._worker_thread.is_alive():
                break
        
        total_time = time.time() - start_time
        
        result = {
            "test_name": test_name,
            "mode": mode,
            "image_count": len(image_files),
            "first_result_time": first_result_time,
            "first_graph_update_time": first_graph_update_time,
            "total_time": total_time,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"[TEST] Total time: {total_time:.3f}s")
        print(f"[TEST] First result: {first_result_time:.3f}s" if first_result_time else "[TEST] No results")
        
        root.destroy()
        return result


def main():
    """메인 테스트 실행"""
    tester = AutoOptimizationTester()
    
    # 테스트 이미지 확인
    if not os.path.exists(tester.test_images_dir):
        print(f"[ERROR] Test images directory not found: {tester.test_images_dir}")
        print("[INFO] Run test_auto_performance.py first to generate test images")
        return
    
    # 테스트 이미지 파일 찾기
    image_files = []
    for f in os.listdir(tester.test_images_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(tester.test_images_dir, f))
    
    if not image_files:
        print(f"[ERROR] No images found in {tester.test_images_dir}")
        return
    
    print(f"[INFO] Found {len(image_files)} test images")
    
    # 테스트 실행
    results = []
    for i in range(3):  # 3번 반복
        result = tester.run_test(image_files[:5], mode="Fast", test_name=f"iteration_{i+1}")
        results.append(result)
        time.sleep(1)  # 간격
    
    # 결과 저장
    output_file = f"gui_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[TEST] Results saved to {output_file}")
    
    # 결과 요약
    if results:
        avg_first_result = sum(r["first_result_time"] or 0 for r in results) / len([r for r in results if r["first_result_time"]])
        avg_total = sum(r["total_time"] for r in results) / len(results)
        print(f"\n[SUMMARY] Average first result time: {avg_first_result:.3f}s")
        print(f"[SUMMARY] Average total time: {avg_total:.3f}s")


if __name__ == "__main__":
    main()
