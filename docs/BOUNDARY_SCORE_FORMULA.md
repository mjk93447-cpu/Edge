# 테두리 감지 최적화 Score 공식

## 1. 합성 테스트 이미지에서 점수가 0.1 수준에 머문 원인

기존 **자동 최적화 점수**(`compute_auto_score`)는 다음 때문에 합성 이미지에서 매우 낮게 나왔다.

1. **band_ratio / coverage 기준이 Otsu 마스크 기반**
   - 경계 밴드가 `estimate_object_mask`(Otsu) → 경계 픽셀 → dilate로 정의된다.
   - 합성 이미지도 노이즈·그라디언트 때문에 Otsu 경계가 실제 윤곽과 어긋나거나, 에지가 밴드 밖에 많이 나가면 **band_ratio**가 낮아진다.
   - `q_band = sigmoid((band_ratio - 0.85) / 0.08)` → band_ratio가 0.85 미만이면 점수가 급격히 떨어진다.

2. **연속성(continuity) 패널티가 가혹함**
   - `continuity_penalty = min(1, (components - 1) * 5 / edges_in_band)`
   - Sobel 결과가 조각나면 **연결 요소 수(components)**가 많아지고, 기하평균에서 **가중치가 큰 continuity(24)** 때문에 전체 점수가 크게 깎인다.

3. **가중 기하평균**
   - 한 항목만 나빠도 전체 점수가 곱해져서 낮아진다. (예: q_cont, q_band가 작으면 score ≈ 0.1대)

4. **endpoint/wrinkle/branch 추가 감쇠**
   - `exp(-2.5 * (endpoint_ratio + wrinkle_ratio + branch_ratio))`로 한 번 더 곱해져, 합성 이미지의 조각난 에지·분기에서 점수가 더 떨어진다.

즉, **“실제 윤곽선과의 정합성”이 아니라 Otsu 밴드·연결 요소 수·분기 등에 강하게 반응**해서, 합성에서 0.1 수준에 머문다.

---

## 2. 실제 테두리 감지에 맞춘 목표

- **다른 물체·반대쪽 경계와 겹치지 않게** 테두리만 구분
- **정확히 테두리를 따라가는**
- **끊어지지 않은 하나의 연결된**
- **얇은 선**

이에 맞추기 위해 **GT(ground truth) 윤곽선**과 비교하는 **boundary 최적화 점수**를 도입했다.

---

## 3. Boundary 최적화 Score 공식

GT가 있을 때(합성 또는 정답 마스크가 있을 때) 사용하는 **`compute_boundary_optimized_score`**:

| 항목 | 의미 | 계산 |
|------|------|------|
| **alignment** | 예측이 GT 경계와 얼마나 맞는지 | F1 (tolerance=1 픽셀) |
| **thinness** | 예측이 얼마나 얇은지 | min(1, gt_pixels / pred_pixels) |
| **connectivity** | 하나의 연결선에 가까운지 | 1 - 0.25*(n_components - 1), 단일 연결이면 1 |
| **single_line** | 끝점·분기 적을수록 좋음 | 1 - 0.4*endpoint_ratio - 0.25*branch_ratio |

**최종 점수 (기본 가중치)**  
`score = 0.45*alignment + 0.25*thinness + 0.20*connectivity + 0.10*single_line`

가중치와 connectivity/endpoint/branch 패널티는 `weights` 인자로 조정 가능하며, `boundary_score_eval.py` 4회 루프로 튜닝한 값은 JSON으로 저장된다.

---

## 4. 4회 테스트/평가/전략도출/코드업데이트 루프 결과

- **스크립트**: `boundary_score_eval.py`
- **동작**: 합성 이미지(원·사각·3원) + GT 윤곽 생성 → 후보 설정으로 검출 → GT 대비 F1·thinness·n_components·endpoint/branch 계산 → boundary_optimized_score 계산 → 오버레이 저장 → 전략 도출(가중치/패널티 조정) → 다음 루프.

**요약**  
- 루프 1: F1 0.97~0.99, thinness ≈ 1, but n_components 4~10 → boundary_score 약 0.78~0.85.
- 루프 2~4: 연결성·single_line 가중치/패널티를 키우면서 같은 예측에 대해 점수는 0.67~0.73대로 감소 (연결 요소가 많은 결과를 더 엄격히 감점).
- **정리**:  
  - **정렬·얇음이 좋고 연결이 잘 되면** (F1 높고, n_components 작고, endpoint/branch 적으면) **점수가 높게 나오는** 방향으로 공식이 동작함.  
  - 합성 이미지에서 0.1대였던 기존 점수와 달리, **GT 기준 정렬·얇음·연결성을 반영**해 0.67~0.85 구간으로 구분된다.

---

## 5. 사용 방법

- **GT가 있는 경우** (합성, 또는 정답 경계 마스크가 있는 경우)  
  - `compute_gt_metrics(pred_edges, gt_boundary, detector)` 로 메트릭 계산.  
  - `compute_boundary_optimized_score(metrics_gt, weights=...)` 로 점수 계산.  
  - 필요하면 `boundary_score_eval.py`에서 사용하는 가중치를 JSON에서 읽어 `weights`로 넘긴다.
- **GT가 없는 실제 이미지**  
  - 기존 `compute_auto_score`(band/continuity 기반)를 그대로 사용하거나, 필요 시 두 점수를 혼합해 사용할 수 있다.

오버레이 이미지: `boundary_score_eval_out/loopN_caseM_overlay.png` (녹색=예측, 빨강=GT, 노랑=일치).
