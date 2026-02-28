@echo off
REM ============================================================
REM  Vision2Voice -- Windows Startup Script
REM ============================================================

echo ========================================
echo   Vision2Voice AI -- Startup Script
echo ========================================

cd /d "%~dp0.."

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not on PATH.
    echo         Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM --- Virtual environment ---
if not exist ".venv" (
    echo [INFO]  Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
echo [INFO]  Virtual environment activated.

REM --- Dependencies ---
echo [INFO]  Installing dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

REM --- Model weight check ---
if not exist "models\modelConcat_1_89.h5" (
    echo.
    echo [WARNING] Model weights not found in models\
    echo           Place the following files there before running inference:
    echo             - models\modelConcat_1_89.h5
    echo             - models\caption_train_tokenizer.pkl
    echo.
)

REM --- Launch ---
echo [INFO]  Launching Vision2Voice dashboard...
set PYTHONPATH=%CD%\src
streamlit run app\streamlit_app.py --server.port=8501

pause
