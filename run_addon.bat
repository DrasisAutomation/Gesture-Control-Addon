@echo off
title Gesture Control Add-on + ChatGPT AI Verification
cd /d "%~dp0"

set PYTHON_CMD=python
if exist "..\backend\.venv\Scripts\python.exe" set PYTHON_CMD="..\backend\.venv\Scripts\python.exe"

echo ====================================================================
echo  Launching Gesture Control Add-on with ChatGPT AI Verification
echo  Port: 8099 (Web UI: http://localhost:8099/)
echo  OpenAI Model: gpt-4o-mini
echo  Prevents False Triggers (Phone calls, mugs, face touching)
echo ====================================================================
echo.

%PYTHON_CMD% app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with code %errorlevel%.
    pause
)
