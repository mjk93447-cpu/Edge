# Formal Development Document (Korean / English)

## 1. Document Control / 문서 관리
- Document Title / 문서명: OLED FCB Edge Detection System
- Version / 버전: 15
- Date / 작성일: 2026-02-06
- Owner / 담당: Min joon kim /Edge Detection AI PIC(Assy/Ins)

## 2. Purpose / 목적
**KR:** OLED FCB 사이드 뷰 이미지에서 외곽선 에지를 안정적으로 검출하여
라인 추적 및 품질 판단에 활용하기 위한 시스템을 정의한다.  
**EN:** Define the system for robust outer-boundary edge detection on OLED FCB
side-view images for line tracing and quality validation.

## 3. Scope / 범위
**KR:** 본 문서는 알고리즘, GUI, 자동 최적화, 평가 도구, 배포 방식까지 포함한다.  
**EN:** This document covers the algorithm, GUI, auto-optimization, evaluation
tools, and deployment method.

## 4. Definitions / 용어 정의
- **Boundary Band**: 객체 외곽선 근방의 허용 대역
- **Continuity**: 외곽선 연결성 지표
- **Intrusion**: 외곽선이 내부로 침범하는 정도
- **Wrinkle**: 곡선이 주름처럼 불규칙하게 꺾이는 현상

## 5. System Overview / 시스템 개요
**KR:**  
입력 이미지를 전처리(블러/중앙값/대비 보정) 후 Sobel 기반 에지 검출을 수행한다.
이후 Non-Maximum Suppression, 이중 임계값, 히스테리시스 추적을 적용하며,
Polarity Filter 및 Boundary Band Filter로 내부 침범을 억제한다.  
자동 최적화는 다중 후보 생성 → 평가 → 적응형 재탐색으로 진행된다.

**EN:**  
Input images are pre-processed (blur/median/contrast) and processed with a
Sobel-based edge detector. NMS, double-thresholding, and hysteresis tracking
follow. Polarity and boundary-band filters suppress internal intrusion.  
Auto-optimization executes candidate generation, evaluation, and adaptive
re-search.

## 6. Functional Requirements / 기능 요구사항
1. 최대 100장 이미지 배치 처리
2. 결과 이미지(녹색 에지) 및 좌표 텍스트 저장
3. ROI 기반 자동 최적화 지원
4. Auto optimization 진행률/그래프 표시
5. 저장/불러오기 지원

1. Up to 100 images in batch processing
2. Save the resulting image (green edge) and coordinate text
3. Support ROI-based automatic optimization
4. Auto optimization progress/graph display
5. Save/Import Support

## 7. Architecture / 아키텍처
**KR:**  
GUI → 파라미터 설정 → 에지 검출 파이프라인 → 결과 저장  
Auto Optimization → 후보 생성 → ROI/클러스터 기반 평가 → 결과 적용

**EN:**  
GUI → parameter settings → edge detection pipeline → output  
Auto Optimization → candidate generation → ROI/cluster evaluation → apply best

## 8. Evaluation & Validation / 검증

- 합성 굽힘 루프 및 복잡한 루프 테스트
- 저품질/노이즈/블러 시나리오
- 메트릭 : 연속성, 밴드핏, 두께, 침입, 주름, 종점
  
- Synthetic bending-loop and complex-loop tests
- Low-quality/noise/blur scenarios
- Metrics: continuity, band-fit, thickness, intrusion, wrinkle, endpoints

## 9. Risks / 리스크
1. 데이터별 특성 차이로 인한 최적 파라미터 변동
2. 과도한 스무딩으로 인한 경계 약화
3. 긴 최적화 시간을 일으키는 원인이 되는 넓은 탐색 범위
      
1. Variation of optimal parameters due to differences in characteristics by data;
2. Weakening of boundaries due to excessive smoothing
3. Wide navigation range, causing long optimization time

## 10. Maintenance / 유지보수
- Auto config 범위를 프로젝트별로 축소 권장
- ROI 캐시를 운영 환경과 분리 관리
- 주요 파라미터 변경 시 평가 재수행

- Reduce Auto config range by project
- Separate ROI cache from production
- Re-perform evaluation when major parameters are changed
- 
## 11. Change Log / 변경 이력

- v13: 자동 최적화 점수 확장 (연속성/밴드 적합)
- v14: ROI 캐시 + 멀티 그래프 GUI + ETA
- v15: 적응 단계 검색, 주름 / 끝점 패널티, 확대 / 축소 그래프
  
- v13: Auto optimization scoring expanded (continuity/band fit)
- v14: ROI cache + multi-graph GUI + ETA
- v15: Adaptive step search, wrinkle/endpoints penalties, zoomable graphs

---

### Appendix A. Operator Quick Guide / 운영자 요약 가이드
1. 이미지 선택 → ROI 설정 → Auto Optimize 실행
2. 그래프 클릭 → 확대/드래그로 확인
3. 최적 결과 적용 후 Batch 처리 수행

1. Select Image → Set ROI → Run Auto Optimize
2. Click on the graph → Confirm by zooming/dragging
3. Batch processing after applying optimal results
