@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1"
if errorlevel 1 (
    echo.
    echo Setup or launch failed. See logs\setup.log for details.
    pause
    exit /b 1
)

exit /b 0
