@echo off
title Push MiniGPT to GitHub
cls
echo ======================================================================
echo MiniGPT: Push Local Repository to GitHub
echo ======================================================================
echo.

set GIT_PATH="C:\Users\20231COM0209\AppData\Local\Programs\Git\cmd\git.exe"

%GIT_PATH% status >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not initialized.
    pause
    exit /b
)

echo [*] Local Git repository is ready on branch 'main'.
echo.
echo Please create a new repository on GitHub:
echo   1. Go to https://github.com/new
echo   2. Repository name: MiniGPT (or any name you choose)
echo   3. Leave "Initialize this repository with a README" UNCHECKED
echo   4. Click "Create repository"
echo.
set /p REPO_URL="Enter your GitHub Repository URL (e.g. https://github.com/username/MiniGPT.git): "

if "%REPO_URL%"=="" (
    echo [!] No URL provided. Aborting.
    pause
    exit /b
)

echo.
echo [*] Adding remote origin: %REPO_URL%
%GIT_PATH% remote remove origin >nul 2>&1
%GIT_PATH% remote add origin %REPO_URL%

echo [*] Pushing branch 'main' to GitHub...
%GIT_PATH% push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ======================================================================
    echo [+] Successfully pushed MiniGPT to GitHub!
    echo ======================================================================
) else (
    echo.
    echo [!] Push encountered an issue. Ensure you are signed in or have entered your personal access token / SSH key.
)

pause
