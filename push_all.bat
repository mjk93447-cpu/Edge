@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [1/2] 모든 변경사항 스테이징 및 커밋...
git add -A
git status
git commit -m "ver20: Perfect mode, raw exponential display, VERSION/README/docs"
if %ERRORLEVEL% neq 0 (
  echo 변경사항이 없거나 이미 커밋됨.
)

echo.
echo [2/2] GitHub에 푸시 (main)...
git push origin main
if %ERRORLEVEL% neq 0 (
  echo 푸시 실패. 네트워크/인증 확인 후 다시 실행하세요.
  pause
  exit /b 1
)

echo.
echo 완료. GitHub Actions에서 빌드 후 Artifacts에 edge_batch_gui_windows 가 올라갑니다.
echo https://github.com/mjk93447-cpu/Edge/actions
pause
