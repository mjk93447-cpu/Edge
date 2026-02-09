@echo off
chcp 65001 >nul
cd /d "%~dp0"
:: ver20를 GitHub에 푸시하고 태그 v20으로 릴리스 트리거
echo [1/3] origin main 푸시 중...
git push origin main
if %ERRORLEVEL% neq 0 (
  echo 푸시 실패. 먼저 push_all.bat 또는 git add/commit 후 다시 실행하세요.
  pause
  exit /b 1
)
echo [2/3] 태그 v20 생성...
git tag -a v20 -m "Release ver20: Perfect mode, raw exponential display" 2>nul || echo 태그 v20 이미 존재할 수 있음
echo [3/3] 태그 v20 푸시 (Actions에서 EXE 빌드 후 Releases에 올라갑니다)...
git push origin v20
if %ERRORLEVEL% neq 0 (
  echo 태그 푸시 실패. 이미 v20이 있으면: git push origin v20 --force
  pause
  exit /b 1
)
echo.
echo 완료. GitHub Actions에서 빌드가 끝나면 Releases에서 exe를 받을 수 있습니다.
echo https://github.com/mjk93447-cpu/Edge/releases
pause
