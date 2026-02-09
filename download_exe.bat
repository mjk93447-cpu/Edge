@echo off
chcp 65001 >nul
setlocal
:: Edge Batch GUI - GitHub에서 EXE 다운로드 (ver19)
:: 사용: download_exe.bat [버전]
:: 예: download_exe.bat        → v19 다운로드
:: 예: download_exe.bat v19    → v19 다운로드

set "VER=%~1"
if "%VER%"=="" set "VER=v19"

set "URL=https://github.com/mjk93447-cpu/Edge/releases/download/%VER%/edge_batch_gui.exe"
set "OUT=%~dp0edge_batch_gui.exe"

echo.
echo [Edge] %VER% EXE 다운로드 중...
echo URL: %URL%
echo 저장: %OUT%
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { " ^
  "  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " ^
  "  Invoke-WebRequest -Uri '%URL%' -OutFile '%OUT%' -UseBasicParsing; " ^
  "  Write-Host '다운로드 완료:' '%OUT%' -ForegroundColor Green; " ^
  "} catch { " ^
  "  Write-Host '다운로드 실패. Releases에 %VER% 가 있는지 확인하세요.' -ForegroundColor Red; " ^
  "  Write-Host $_.Exception.Message; exit 1; " ^
  "}"

if %ERRORLEVEL% neq 0 (
  echo.
  echo ※ GitHub에서 태그 %VER% 로 릴리스가 생성되어 있어야 합니다.
  echo   저장소: https://github.com/mjk93447-cpu/Edge/releases
  pause
  exit /b 1
)

echo.
echo 실행: edge_batch_gui.exe
pause
