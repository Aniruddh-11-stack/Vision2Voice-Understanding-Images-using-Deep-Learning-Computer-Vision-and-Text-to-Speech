@echo off
echo ==========================================================
echo Starting Vision2Voice Dashboard Setup
echo ==========================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    pause
    exit /b 1
)

:: Create Virtual Environment
if not exist venv\ (
    echo Creating Python virtual environment...
    python -m venv venv
)

:: Activate and install dependencies
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing requirements...
pip install -r requirements.txt

echo.
echo Starting Vision2Voice Streamlit App...
streamlit run app.py

pause
