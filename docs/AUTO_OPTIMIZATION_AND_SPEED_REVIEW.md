# Auto Optimization 및 연산 속도/성능 코드 리뷰

## 1. Auto Optimization 관련 코드 요약

### 1.1 진입점 및 데이터 흐름

| 위치 | 함수/변수 | 역할 |
|------|-----------|------|
| `EdgeBatchGUI._start_auto_optimize()` | 사용자 클릭 시 | `_auto_optimize_worker` 스레드 시작, ROI/설정 수집 |
| `_auto_optimize_worker()` | 메인 루프 | 데이터 준비 → 라운드별 후보 풀 생성 → 평가 → best 갱신 |
| `_prepare_auto_data()` | 데이터 준비 | 이미지 로드, 마스크/경계/밴드 전처리 (병렬 가능) |
| `_evaluate_settings()` | 단일 후보 평가 (GUI 프로세스) | 이미지별 `detect_edges_array` + 메트릭 집계 + `compute_auto_score` |
| `evaluate_one_candidate_mp()` | 단일 후보 평가 (워커 프로세스) | ProcessPool용; 동일 로직, 모듈 임포트 후 실행 |
| `_eval_candidate_wrapper_mp()` | 래퍼 | `(data, settings, auto_config)` → `evaluate_one_candidate_mp` 호출 |

- **설정 소스**: `PARAM_DEFAULTS` + `AUTO_DEFAULTS` + GUI `param_vars` → `_get_param_settings()` / `_get_auto_config()`.
- **점수**: 항상 **raw score** (0~1) 사용. UI 표시만 `SCORE_DISPLAY_SCALE`로 스케일.

### 1.2 후보 생성 및 라운드 예산

| 항목 | Fast 모드 | Normal | Perfect |
|------|-----------|--------|---------|
| `target_eval` | 3000 | 9000 | 45000 |
| `round_budget` | 200 | 500 | 1200 |
| 1라운드 explore | `min(200//4, 80)` = 50 | `min(500//3, 150)` | 동일 비율 |
| exploit / local | 2라운드부터, best 기준 | 동일 | 동일 |

- **`_build_candidates(base_settings, mode, auto_config, count, rng, step_scale, centers, step_multipliers)`**
  - `AUTO_DEFAULTS`의 min/max/step으로 NMS, high/low, margin, band, blur, thinning, contrast, soft link 등 샘플링.
  - `centers`가 있으면 해당 설정 주변으로, `step_scale`로 탐색 범위 조절 (explore 넓게, exploit 좁게).
  - `candidate_key()`로 중복 제거.
- **`_build_local_grid(best, base_settings, auto_config, size=32)`**
  - best 주변 nms/high/margin/band만 소규모 그리드로 생성.

### 1.3 2단계(Phase) 학습

- **Phase 1 (pre-thinning)**  
  - `use_thinning=False`, `phase1_budget = max(phase1_min_evals, target_eval * phase1_frac)` (기본 50%).  
  - 목표: `best_thickness <= phase1_max_thickness`(0.25) 또는 phase1_budget 소진.
- **Phase 2**  
  - Phase 1 best에서 `use_thinning=True`, `thinning_max_iter=mid_thinning`으로 전환 후, 최대 `target_eval`까지 평가.

### 1.4 조기 종료

- **라운드 내 early exit**: `no_improve_count >= early_exit_after` (= `round_early_exit_frac * len(pool)`, 기본 25%) 연속 무개선 시 해당 라운드 중단.
- **라운드 단위**: `no_improve_rounds >= auto_no_improve_rounds_stop` (기본 2)이면 전체 종료.
- **시간 기반**: `early_stop_enabled`이고 `(현재시간 - last_best_time) > early_stop_minutes`이면 stagnation으로 종료.

### 1.5 스코어링

- **`compute_auto_score(metrics, weights, return_details=False)`**
  - 메트릭: coverage, gap, continuity, intrusion, outside, thickness, band_ratio, endpoints, wrinkle, branch.
  - 가중 기하평균 + (endpoint+wrinkle+branch)에 대한 exp 보정.  
  - 가중치: continuity(24), band_fit(12) 등이 지배적.
- **저대비 이미지**: `contrast_approx < contrast_ref*0.6`이면 `score *= (1 + weight_low_quality)`.

---

## 2. 연산 속도/성능 관련 코드 정리

### 2.1 파이프라인 (이미지당 1회)

- **`SobelEdgeDetector.detect_edges_array()`**  
  - 메디안 → 가우시안 블러 → Sobel → NMS → 이중 임계값/히스테리시스 → (선택) 소프트 링크 → 폐색 → 극성 필터 → 밴드 필터 → 스무딩 → **Zhang-Suen Thinning** → 스퍼 제거.  
  - 전부 NumPy/CPU, OpenCV 미사용.

**주요 연산:**

| 단계 | 함수/위치 | 비고 |
|------|-----------|------|
| 컨볼루션 | `apply_convolution`, `sliding_window_view` + `tensordot` | 공통 블러/메디안/Sobel |
| 메디안 | `apply_median_filter` | `np.median(windows)` |
| NMS | `non_maximum_suppression` | 패딩+방향별 비교 |
| 이중 임계값 | `double_threshold` | ratio 시 max 기반, percentile/MAD 선택 가능 |
| 에지 추적 | `edge_tracking` | 스택 기반 BFS |
| Thinning | `thin_edges_zhang_suen` | 반복 구조, 분기 많음 |
| 스퍼 제거 | `prune_spurs` | `_neighbor_count` + 반복 |
| 밴드 필터 | `boundary_band_filter` | 마스크 추정 + dilate |

- **이미 최적화된 부분**
  - `detect_edges_array` 내 contrast: `np.percentile` 대신 `image.min()/max()` 사용.
  - `evaluate_one_candidate_mp` / `_evaluate_settings` 내 wrinkle: erode/dilate 대신 neighbor 기반 근사.

### 2.2 병렬화 구조

| 레벨 | 방식 | 위치 | 비고 |
|------|------|------|------|
| 이미지 단위 | `ThreadPoolExecutor` | `_evaluate_settings` 내 `evaluate_item` | `auto_workers` 수만큼 이미지 병렬 |
| 후보 단위 | `ProcessPoolExecutor` | `_auto_optimize_worker` 배치 루프 | `auto_candidate_workers`≥1이고 2라운드 이후(또는 processed>0)일 때만 사용 |
| 데이터 준비 | 스레드 풀 가능 | `_prepare_auto_data` | 파일/이미지별 전처리 |

- **병목**
  - 후보 1개 = 이미지 N장 × `detect_edges_array` 전부 CPU.  
  - 후보 단위는 첫 라운드 첫 평가는 순차(`use_parallel = candidate_workers >= 1 and (round_num > 1 or processed > 0)`).

### 2.3 설정으로 조절 가능한 항목

- **`auto_workers`**: 이미지 병렬 워커 수 (기본 `cpu_count-1`).
- **`auto_candidate_workers`**: 후보 병렬 프로세스 수 (0이면 순차).
- **`auto_parallel`**: 이미지 단위 병렬 사용 여부.

### 2.4 문서화된 개선 방향 (GPU_ACCELERATION_REVIEW.md)

- CPU만: 후보 단위 `ProcessPoolExecutor` 확대 → 2~5배.
- NVIDIA GPU: CuPy/OpenCV CUDA → 약 3~10배.
- 내장 GPU: OpenCL/oneAPI → 약 1.2~2.5배.

---

## 3. 영점삼(0.3) 이상 빠르게 달성하기 위한 학습 전략

- **목표**: raw score ≥ 0.3을 **가능한 한 적은 평가 수/시간**으로 도달.
- **적용 포인트**
  1. **목표 스코어 조기 종료**: `best_score >= target_score` (예: 0.3)이면, 옵션에 따라 즉시 또는 소수 라운드 더 돌린 뒤 종료.
  2. **Fast 모드 강화**: 0.3 도달이 목표일 때는 Fast의 `round_budget`/phase1 비율 유지, `target_eval` 상한은 유지하되 `target_score` 도달 시 중단.
  3. **Phase1 우선**: 두께 목표만 만족하면 Phase2로 가되, 이미 0.3 넘었으면 Phase2 평가 수를 줄이거나 생략 가능.
  4. **coarse 데이터**: 첫 라운드 첫 평가만 `data_coarse` 사용하는 현재 동작 유지 → 초기 피드백 속도 확보.

구체 구현은 `AUTO_DEFAULTS`에 `auto_target_score`(0.3), `auto_target_score_rounds_after`(0 또는 1~2) 추가하고, `_auto_optimize_worker`에서 `best_score >= auto_target_score`일 때 카운트 후 종료하면 됨.

---

## 4. 구현 완료 사항 (영점삼 빠른 도달)

- **AUTO_DEFAULTS**
  - `auto_target_score`: 0.3 (0이면 비활성)
  - `auto_target_score_rounds_after`: 1 (도달 후 추가 라운드 수)
- **`_auto_optimize_worker`**
  - 목표 도달 시 `target_reached = True`, 라운드 종료 시 `rounds_since_target` 증가.
  - `rounds_since_target > auto_target_score_rounds_after`이면 `stop_reason = "target_score_reached"`로 종료.
- **CLI 테스트**: `run_fast_target_score_test.py`
  - GUI 없이 데이터 준비 후 목표 0.3 도달까지 평가 루프 실행.
  - 3회 테스트-평가-재개발: Loop1 순차(workers=0), Loop2 병렬 2, Loop3 병렬 4.
  - 결과는 `fast_target_score_loop_results.json`에 저장.
