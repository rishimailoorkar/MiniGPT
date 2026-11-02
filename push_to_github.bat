@echo off
title Push MiniGPT to GitHub (@rishimailoorkar)
cls
echo ======================================================================
echo MiniGPT: Push Local Repository to GitHub
echo Account: rishimailoorkar (https://github.com/rishimailoorkar)
echo Target:  https://github.com/rishimailoorkar/MiniGPT.git
echo ======================================================================
echo.

set GIT_PATH="C:\Users\20231COM0209\AppData\Local\Programs\Git\cmd\git.exe"

echo [*] Checking local repository status...
%GIT_PATH% status >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not initialized.
    pause
    exit /b
)

echo [*] Remote configured: https://github.com/rishimailoorkar/MiniGPT.git
echo.
echo Make sure you have created the empty repository on GitHub:
echo   1. Visit: https://github.com/new
echo   2. Repository name: MiniGPT
echo   3. Leave README, .gitignore, and License UNCHECKED
echo   4. Click "Create repository"
echo.
echo Press any key to push your code to GitHub...
pause >nul

echo.
echo [*] Pushing branch 'main' to origin...
%GIT_PATH% push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ======================================================================
    echo [+] SUCCESS! MiniGPT has been pushed to:
    echo     https://github.com/rishimailoorkar/MiniGPT
    echo ======================================================================
) else (
    echo.
    echo [!] Push encountered an issue (authentication or repository not created yet).
    echo     If GitHub asks for credentials, sign in via browser or paste a GitHub Personal Access Token (PAT).
)

echo.
pause
