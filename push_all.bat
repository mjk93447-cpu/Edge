@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [0/3] 테스트 실행...
python -m unittest test_smoke -v
if %ERRORLEVEL% neq 0 (
  echo 테스트 실패. 수정 후 다시 실행하세요. (또는 이 배치를 프로젝트 폴더에서 실행하세요.)
  pause
  exit /b 1
)
echo Tests OK.

echo.
echo [1/3] 스테이징 및 커밋...
git add -A
git status
git commit -m "ver20: Wiki 반영 - score 균형(곱+합), global best 유지, 500장, 리포트/GUI에 score 표시"
if %ERRORLEVEL% neq 0 (
  echo 변경사항이 없거나 이미 커밋됨.
)

echo.
echo [2/3] GitHub에 푸시 (main)...
git push origin main
if %ERRORLEVEL% neq 0 (
  echo 푸시 실패. 네트워크/인증 확인 후 다시 실행하세요.
  pause
  exit /b 1
)

echo.
echo [3/3] 완료. GitHub Actions에서 자동으로 테스트 후 EXE 빌드합니다.
echo Artifacts: https://github.com/mjk93447-cpu/Edge/actions
echo Releases (태그 v20 푸시 시): https://github.com/mjk93447-cpu/Edge/releases
pause
