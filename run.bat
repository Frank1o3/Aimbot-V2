@echo off
setlocal
title Aimbot
mode con: cols=50 lines=10

REM Change to script directory
cd /d "%~dp0"

timeout /t 1 >nul

echo.
echo ===============================================
echo             Launching Python Script
echo ===============================================
echo.

timeout /t 1 >nul

REM Check if venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo [!] Virtual environment not found. Run setup.bat first.
    echo.
    pause
    exit /b 1
)

REM Activate venv
call ".venv\Scripts\activate.bat"

REM Run your main Python script
cls
python src\Main.py

cls

echo.
echo ===============================================
echo             Script Execution Finished
echo ===============================================
echo.
timeout /t 1 >nul
exit /b
