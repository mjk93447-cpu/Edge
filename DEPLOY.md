# 테스트 · 커밋 · 푸시 · EXE 생성 절차

## 한 번에 실행 (권장)

1. **프로젝트 폴더**에서 **`push_all.bat`** 더블클릭.
2. 순서:
   - **[0/3]** 로컬에서 `python -m unittest test_smoke -v` 실행 → 실패 시 푸시 중단.
   - **[1/3]** `git add -A` → `git commit` (ver20 Wiki 반영 메시지).
   - **[2/3]** `git push origin main`.
3. 푸시가 끝나면 **GitHub Actions**가 자동으로:
   - 저장소 체크아웃
   - 테스트 실행 (`python -m unittest discover -s . -p "test*.py"`)
   - EXE 빌드 (`pyinstaller ...`)
   - **Artifacts**에 `edge_batch_gui_windows` 업로드

## EXE 받는 방법

- **Actions**: https://github.com/mjk93447-cpu/Edge/actions  
  → 가장 최근 "Build Windows EXE" 워크플로 선택 → 맨 아래 **Artifacts** → `edge_batch_gui_windows` 다운로드.
- **Releases**: 태그 `v20` 푸시 시 같은 워크플로가 **Releases**에 exe 첨부.  
  - `release_ver20.bat` 실행 시 `git push origin v20` 까지 수행.

## 수동 실행 (배치 대신)

```bat
cd /d "프로젝트폴더경로"
python -m unittest test_smoke -v
git add -A
git commit -m "ver20: Wiki 반영 - score 균형, global best 유지, 500장, score 표시"
git push origin main
```

## 테스트만 실행

- **`run_tests.bat`** 더블클릭  
또는  
- `python -m unittest test_smoke -v` (프로젝트 폴더에서)

## 실패 시

- **테스트 실패**: `push_all.bat`은 푸시하지 않음. 테스트 통과 후 다시 실행.
- **푸시 실패**: 네트워크·Git 인증(계정/토큰) 확인.
- **Actions 빌드 실패**: GitHub 저장소 Actions 탭에서 로그 확인. (테스트 실패 시 빌드 단계까지 가지 않음.)
