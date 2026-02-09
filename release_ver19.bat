@echo off
chcp 65001 >nul
:: ver19를 GitHub에 푸시하고 태그 v19로 릴리스 트리거
echo [1/3] origin main 푸시 중...
git push origin main
if %ERRORLEVEL% neq 0 (
  echo 푸시 실패. 네트워크/인증 확인 후 다시 실행하세요.
  pause
  exit /b 1
)
echo [2/3] 태그 v19 생성 (이미 있으면 스킵)...
git tag -a v19 -m "Release ver19" 2>nul || echo 태그 v19 이미 존재할 수 있음
echo [3/3] 태그 v19 푸시 (Actions에서 EXE 빌드 후 Releases에 올라갑니다)...
git push origin v19
if %ERRORLEVEL% neq 0 (
  echo 태그 푸시 실패. 이미 v19가 있으면: git push origin v19 --force
  pause
  exit /b 1
)
echo.
echo 완료. GitHub Actions에서 빌드가 끝나면 Releases에서 exe를 받을 수 있습니다.
echo https://github.com/mjk93447-cpu/Edge/releases
pause
