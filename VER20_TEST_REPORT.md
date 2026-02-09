# AI Image Edge Detector — ver20 Test Report & Update Purpose

## 1. Ver20 Update Purpose List

### 1.1 Perfect mode (Auto Optimize)
- **목적**: Precise보다 약 5배 시간이 걸리더라도 더 완벽한 결과를 얻기 위한 모드 추가.
- **구현**:
  - **중요도 기반 파라미터 간격**: `PERFECT_STEP_MULTIPLIERS`로 NMS, high_ratio, low_ratio, polarity_drop_margin 등 중요 파라미터는 2~10배 촘촘하게 탐색 (step × 0.2~0.5).
  - **동일 score 함수**: `compute_auto_score` 및 평가 파이프라인은 변경 없음. Perfect는 탐색 전략만 추가.
  - **고효율 학습 전략**:
    - Coarse 단계: 더 큰 budget, importance-weighted step으로 후보 생성.
    - Refine 단계: 5× 촘촘한 그리드 (nms, high, band, margin).
    - Adaptive 단계: 5라운드 (scale 0.5→0.05), top-12 중심, step_multipliers 유지.
    - **Coordinate descent**: Perfect 전용. nms_relax, high_ratio, low_ratio, boundary_band_radius, polarity_drop_margin에 대해 한 번에 한 파라미터만 소폭 변화시켜 국소 최적화 (one-at-a-time local search).
- **UI**: Auto mode 콤보에 "Perfect" 추가.

### 1.2 Score display — raw 지수 형태
- **목적**: raw 화면에서 숫자가 너무 작아 정확한 결과를 보기 어려운 문제 해결.
- **구현**:
  - `_format_score_for_display(value, mode)`: raw일 때 `f"{value:.4e}"` 형태로 문자열 반환.
  - 그래프 Y축 레이블 (`_render_graph`, `_render_time_graph`, `_render_multi_graph`): display_mode가 raw일 때 지수 형식으로 표시.
  - 상태바 및 로그: `_format_score_for_display(score, None)` 사용하여 raw일 때 지수로 표시.

### 1.3 검증 및 배포
- **테스트**: `python -m unittest test_smoke` — 4 tests OK.
- **목표**: score 함수·다른 모드(Fast/Precise) 동작 변경 없음, Perfect만 추가, raw 표시만 개선.

---

## 2. Test Report (Ver20)

| 항목 | 결과 |
|------|------|
| Unit tests (`test_smoke`) | 4 tests, OK (0.15s) |
| Score function | 변경 없음 (기존 compute_auto_score, _evaluate_settings 그대로 사용) |
| Fast / Precise mode | 기존 로직 유지 (mode 분기만 확장) |
| Perfect mode | 추가됨: step_multipliers, 더 큰 budget, finer refine, 5-round adaptive, coordinate descent |
| Raw display | 지수 형식 적용 (그래프 축, 상태바, 로그) |
| GUI title | ver20 |

---

## 3. GitHub 업데이트 및 실행 파일 생성 안내

### 3.1 변경사항 커밋 및 푸시
1. **로컬에서** 프로젝트 폴더 열기.
2. 아래 중 하나 실행:
   - **배치**: `push_all.bat` 더블클릭 (모든 변경 스테이징 → 커밋 → `origin main` 푸시).
   - **수동**:
     ```bat
     git add -A
     git commit -m "ver20: Perfect mode, raw exponential display, docs"
     git push origin main
     ```

### 3.2 실행 파일(EXE) 생성
- **GitHub Actions**: 푸시 후 [Actions](https://github.com/mjk93447-cpu/Edge/actions)에서 "Build Windows EXE" 워크플로 실행. 완료 시 **Artifacts**에서 `edge_batch_gui_windows` 다운로드.
- **Releases에 올리기**: 태그 푸시 시 자동으로 exe가 릴리스에 첨부됨.
  ```bat
  git tag -a v20 -m "Release ver20"
  git push origin v20
  ```
  또는 `release_ver19.bat`을 참고해 `release_ver20.bat`을 만들어 v20 태그 푸시에 사용.

### 3.3 권한이 부족할 때
- **Git push 실패 (인증)**  
  - GitHub 계정으로 로그인: `git config --global credential.helper manager` (Windows).  
  - Personal Access Token (PAT) 사용: GitHub → Settings → Developer settings → Personal access tokens → 생성 후, 비밀번호 대신 토큰 입력.
- **GitHub Actions 권한**  
  - 저장소 **Settings** → **Actions** → **General** → "Workflow permissions"에서 "Read and write permissions" 선택 후 저장.
- **Releases 업로드 실패**  
  - 워크플로에 `permissions: contents: write` 있음. 위 Actions 권한을 "Read and write"로 두면 해결.

---

## 4. 요약
- **Perfect 모드**: 중요 파라미터 촘촘 탐색 + coordinate descent, score 함수 무변경.
- **Raw 표시**: 지수 형태로 정확히 확인 가능.
- **테스트 통과**, 문서(VERSION, README, FORMAL_DOCUMENTATION) ver20 반영 완료.  
GitHub 푸시 후 Actions에서 exe 생성, 필요 시 v20 태그 푸시로 Releases에 반영하면 됩니다.
