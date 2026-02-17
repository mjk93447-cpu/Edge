# Auto Optimization 속도 개선 검토: GPU 및 내장그래픽

## 1. 현재 구조와 병목

- **파이프라인**: `detect_edges_array()` — NumPy 기반 (OpenCV 미사용)
  - 메디안 필터 → 가우시안 블러 → Sobel 그래디언트 → NMS → 이중 임계값/히스테리시스 → (선택) 소프트 링크 → 폐색 → 극성 필터 → 밴드 필터 → 스무딩 → **Thinning (Zhang-Suen)** → 스퍼 제거
- **병렬화**: `_evaluate_settings()` 내부에서 **이미지 단위**만 `ThreadPoolExecutor`로 병렬 처리 (한 후보당 여러 장 동시 평가). **후보(설정) 단위는 순차**.
- **병목**: 후보 1개 평가 = 이미지 N장 × (NumPy 컨볼루션, NMS, 이중 임계값, 추적, morphology, thinning 등) 전부 **CPU 단일 스레드** per 이미지.

---

## 2. GPU 가속 (NVIDIA 전용)

| 방식 | 설명 | 예상 속도 개선 | 비고 |
|------|------|----------------|------|
| **CuPy** | `numpy` → `cupy` 치환. 컨볼루션/요소연산은 GPU 실행. | **전체 파이프라인 기준 약 3~10배** | NMS·edge_tracking·Zhang-Suen 등은 커널 이식 또는 CPU 유지 필요. 전송/동기화 비용 있음. |
| **OpenCV CUDA** | `cv2.cuda.GaussianBlur`, `Sobel` 등 사용. | **약 3~8배** | 블러·Sobel 등만 GPU, 나머지 로직은 CPU 또는 별도 CUDA 커널 필요. |

- **가능한 이유**: 블러·Sobel·이중 임계값·morphology 등이 **데이터 병렬**에 적합해 GPU에서 5~20배 이상 빨라질 수 있음. CuPy 문서상 일부 연산은 CPU 대비 100배대도 보고됨.
- **제한 요인**:
  - NMS·edge_tracking·thinning은 분기·반복이 많아 GPU 이식이 쉽지 않음. CPU에서 처리하거나 최소한의 커널만 GPU로 옮기는 방식이 현실적.
  - 이미지당 CPU↔GPU 전송 비용. 이미지가 작으면 이득이 줄어듦.
- **결론**: **NVIDIA GPU 사용 시, 현재 대비 대략 3~10배** 정도까지 개선 가능. 5~7배는 현실적인 목표 구간.

---

## 3. 내장그래픽 (Intel UHD/Iris, AMD APU 등)

| 환경 | 기술 | 예상 속도 개선 | 비고 |
|------|------|----------------|------|
| **Intel 내장** | OpenCL 또는 Intel oneAPI (DPC++) | **약 1.5~3배** (단일 스레드 CPU 대비) | CUDA 미지원. 메모리 대역·커널 실행 오버헤드로 GPU 이득이 제한적. |
| **AMD APU** | OpenCL (일부 ROCm) | **약 1.5~2.5배** | 드라이버·플랫폼 의존. |

- **가능한 이유**: 블러·Sobel 같은 연산은 OpenCL로 이식 가능. 내장 GPU도 데이터 병렬에는 유리.
- **제한 요인**:
  - 내장 GPU는 연산력·메모리 대역이 외장 GPU보다 낮음.
  - 전송·커널 런치 비용이 상대적으로 커서, 작은 이미지에서는 1배대만 나올 수 있음.
  - 현재 이미 **멀티스레드(이미지 병렬)** 로 돌고 있어서, 내장 GPU만 써서는 **전체 기준 1.2~2배** 정도가 현실적.
- **결론**: **내장그래픽만으로는 지금 대비 대략 1.2~2.5배** 정도 개선 가능. 2배 전후를 기대하는 수준.

---

## 4. CPU만 써서 빨리 만들기 (즉시 적용 가능)

- **후보(설정) 단위 병렬화**:  
  지금은 후보를 하나씩만 평가함. **여러 후보를 동시에 평가** (예: `ProcessPoolExecutor`로 2~4개 프로세스가 각각 다른 후보의 이미지들을 처리)하면 **코어 수에 따라 2~4배** 추가 개선 가능.
- **추가**: `auto_workers`를 이미지 수에 맞게 키우고, 후보 병렬까지 적용하면 **종합 2~5배**까지는 GPU 없이도 가능.

---

## 5. 요약: 대략 몇 배까지 가능한지

| 구분 | 조건 | 현재 대비 예상 속도 개선 |
|------|------|--------------------------|
| **GPU (NVIDIA)** | CuPy 또는 OpenCV CUDA로 핵심 연산 이식 | **약 3~10배** (5~7배 목표 권장) |
| **내장그래픽** | OpenCL/oneAPI로 블러·Sobel 등 이식 | **약 1.2~2.5배** |
| **CPU만** | 후보 단위 멀티프로세스 + 기존 이미지 병렬 | **약 2~5배** |

- **실제 적용 순서 제안**  
  1) **CPU 후보 병렬화** (구현 난이도 낮음, 2~4배).  
  2) **NVIDIA 있을 때** CuPy/OpenCV CUDA 도입 (3~10배).  
  3) **내장만 있을 때** OpenCL/oneAPI는 선택 사항 (1.2~2.5배, 공수 대비 이득 제한적).

---

## 6. 참고

- CuPy: [cupy.dev](https://cupy.dev), NVIDIA CUDA 필요.
- OpenCV CUDA: `cv2.cuda` 모듈, NVIDIA GPU + CUDA 빌드 필요.
- Intel oneAPI: [oneapi.com](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) — CPU/내장 GPU 타깃.
- 현재 코드: `sobel_edge_detection.py` — `SobelEdgeDetector.detect_edges_array()`, `_evaluate_settings()` (이미지 단위 `ThreadPoolExecutor`만 사용).
