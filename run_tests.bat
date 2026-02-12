@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Running tests...
python -m unittest test_smoke -v
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
  echo Tests FAILED. exit code %EXITCODE%
  pause
  exit /b %EXITCODE%
)
echo All tests passed.
exit /b 0
