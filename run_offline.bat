@echo off
title MiniGPT: Offline Small Language Model
cls
echo ======================================================================
echo MiniGPT: An Offline Small Language Model for Text Generation
echo Running 100%% locally on Windows without Internet or Cloud APIs
echo Hardware: Intel(R) Core(TM) i5-14500 (CPU Native)
echo ======================================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in system PATH. Please install Python 3.10+
    pause
    exit /b
)

echo [*] Starting Streamlit Offline Web Chatbot...
echo [*] Open your web browser at: http://localhost:8501
echo [*] Press Ctrl+C in this terminal to stop the application.
echo.

streamlit run app.py --server.headless=false --server.port=8501 --browser.serverAddress=localhost

pause
