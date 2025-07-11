@echo off
setlocal enabledelayedexpansion
title Python Project Setup

REM Change to script directory
cd /d "%~dp0"

echo.
echo ==================================================
echo             Python Environment Setup
echo ==================================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if errorlevel 1 (
    echo [!] Python is not installed or not in your PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Create a virtual environment
if not exist ".venv\" (
    echo [+] Creating virtual environment...
    python -m venv .venv
) else (
    echo [+] Virtual environment already exists.
)

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Install dependencies
if exist "requirements.txt" (
    echo [+] Installing requirements from requirements.txt...
    pip install -r requirements.txt
) else (
    echo [!] requirements.txt not found. Skipping package installation.
)

echo.
echo ==================================================
echo             Setup Complete!
echo ==================================================
echo.
pause
exit /b
