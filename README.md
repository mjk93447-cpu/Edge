# Edge

## 오프라인 GUI 배치 처리

이 프로젝트는 Sobel 기반 에지 검출을 **오프라인 환경**에서 **GUI**로 실행할 수 있도록 구성되어 있습니다.

### 실행 방법

```bash
python sobel_edge_detection.py
```

### 사용 흐름
1. **파일 추가** 버튼으로 최대 100개의 이미지를 선택합니다.
2. 필요 시 **폴더 변경**으로 출력 폴더를 지정합니다.
3. **처리 시작** 버튼을 눌러 순차적으로 처리합니다.

### 출력 규칙
- 출력은 선택한 폴더 아래 **배치 단위 폴더**에 저장됩니다.
  - 폴더명: `edge_results_YYYYMMDD_HHMMSS`
- 각 이미지에 대해 다음 파일이 생성됩니다.
  - `원본이름_edges_green.png` : 에지 dot가 **초록색**으로 표시된 이미지
  - `원본이름_edge_coords.txt` : 모든 에지 dot의 좌표

`edge_coords.txt` 파일 형식:
```
# x,y
10,25
11,25
...
```

### 필요 패키지
- numpy
- pillow

> `tkinter`는 파이썬 기본 내장 모듈입니다.

## Edge 성능 평가 (합성 이미지)

OLED FCB 벤딩 루프 형태의 합성 이미지를 생성하고 에지 검출 성능을 평가합니다.

```bash
python3 edge_performance_eval.py
```

출력 폴더:
- `outputs/perf_eval_YYYYMMDD_HHMMSS/`
  - `bending_loop_input.png`
  - `bending_loop_mask.png`
  - `bending_loop_edges_green.png`
  - `bending_loop_edges_binary.png`
  - `bending_loop_edges_gt.png`
  - `edge_metrics.txt`

### exe 빌드 (오프라인 배포용)
아래 명령으로 단일 실행 파일을 생성할 수 있습니다.

```bash
python -m pip install -r requirements-dev.txt
pyinstaller --onefile --windowed --name edge_batch_gui.exe sobel_edge_detection.py
```

생성된 실행 파일:
- `dist/edge_batch_gui.exe`