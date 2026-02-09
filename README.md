# Edge — Sobel Edge Batch Processor (ver19)

OLED FCB 측면 이미지의 외곽 경계를 추출하는 **오프라인 GUI 배치 처리** 도구입니다.  
Cursor IDE 및 일반 Python 환경에서 바로 실행·개발할 수 있습니다.

---

## 빠른 실행 (Cursor / 로컬)

```bash
# 가상환경 권장
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

pip install numpy pillow
python sobel_edge_detection.py
```

- **필수**: Python 3.8+, `numpy`, `pillow`
- **tkinter**: Python 기본 내장 (별도 설치 불필요)

---

## Cursor에서 개발할 때

| 항목 | 설명 |
|------|------|
| **실행** | 터미널에서 `python sobel_edge_detection.py` 또는 Run/Debug 사용 |
| **테스트** | `python -m pytest test_smoke.py` 또는 `python test_smoke.py` |
| **성능 평가** | `python edge_performance_eval.py` (합성 이미지 스트레스 테스트) |
| **문서** | `FORMAL_DOCUMENTATION.md`(요구사항/아키텍처), `DEVELOPMENT_NOTES.md`(이력/의사결정) |
| **버전** | GUI 타이틀 및 `FORMAL_DOCUMENTATION.md`에 ver19 기준 반영 |

---

## 사용 흐름

1. **파일 추가**: 최대 100장 이미지 선택
2. **출력 폴더**: 필요 시 폴더 변경
3. **Auto Optimize** (선택): 스코어 기반 자동 파라미터 탐색 후 **처리 시작**

### 스코어 표시 (ver19)

- **scaled** (기본): 스코어 ×10¹⁵ 표시 (실제 환경에서 작은 스코어를 읽기 쉽게)
- **log10**: log₁₀(score)
- **raw**: 원시 스코어  
→ 최적화/학습 알고리즘은 항상 원시 스코어만 사용합니다.

---

## 출력 규칙

- 출력 디렉터리: 선택한 폴더 아래 `edge_results_YYYYMMDD_HHMMSS`
- 이미지당 생성 파일:
  - `원본이름_edges_green.png`: 에지 점(초록) 오버레이
  - `원본이름_edge_coords.txt`: 에지 좌표

`edge_coords.txt` 형식:

```
# x,y
10,25
11,25
...
```

---

## 성능 평가 (합성 이미지)

```bash
python edge_performance_eval.py
```

- 출력: `outputs/perf_eval_YYYYMMDD_HHMMSS/`  
  (입력/마스크/에지/메트릭 등)

---

## EXE 빌드 (오프라인 배포)

```bash
pip install -r requirements-dev.txt
pyinstaller --onefile --windowed --name edge_batch_gui sobel_edge_detection.py
```

- 결과: `dist/edge_batch_gui.exe` (Windows)

### GitHub Actions / Releases

- **Actions** → **Build Windows EXE** → **Run workflow**  
  - 완료 후 **Artifacts**에서 `edge_batch_gui_windows` 다운로드
- **릴리스 EXE 다운로드 (ver19)**  
  - 저장소에서 태그 `v19` 푸시 시 자동으로 [Releases](https://github.com/mjk93447-cpu/Edge/releases)에 exe가 올라갑니다.  
  - **배치 파일**: 프로젝트 루트의 `download_exe.bat` 실행 시 `edge_batch_gui.exe`를 현재 폴더로 받습니다.  
    - 기본: `v19`  
    - 다른 버전: `download_exe.bat v18` 처럼 인자로 태그 지정

---

## 문서

- **FORMAL_DOCUMENTATION.md**: 시스템 개요, 요구사항, 변경 이력 (v19)
- **DEVELOPMENT_NOTES.md**: 개발 이력, 시행착오, 제한사항
- **PROJECT_ASSESSMENT.md**: 서브시스템 성숙도 및 개선 백로그
