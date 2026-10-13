@echo off
title DocuMind AI — 100% Offline RAG System
color 0B

echo ========================================================
echo        DocuMind AI: 100%% Offline RAG Launcher
echo        Academic & Professor Demonstration Edition
echo ========================================================
echo.

REM 1. Enforce 100% Offline Mode for HuggingFace / Transformers
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONUNBUFFERED=1

echo [1/3] Checking Local Ollama Inference Engine...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo   [OK] Ollama daemon is already running locally.
) else (
    echo   [INFO] Starting local Ollama service...
    start /B ollama serve > NUL 2>&1
    timeout /t 3 /nobreak > NUL
    echo   [OK] Ollama service started on port 11434.
)

echo.
echo [2/3] Checking Local SLM Models...
ollama list | find /I "llama3.2" > NUL
if "%ERRORLEVEL%"=="0" (
    echo   [OK] Local model (llama3.2) is ready for zero-network inference.
) else (
    echo   [WARNING] llama3.2 not detected in ollama list.
    echo   If you have internet right now, run: ollama pull llama3.2:1b
    echo   Otherwise, the built-in Extractive Synthesizer will activate.
)

echo.
echo [3/3] Launching DocuMind AI Interface...
echo   Open browser at: http://localhost:8501
echo   Wi-Fi can now be turned completely OFF (Airplane Mode).
echo ========================================================
echo.

python -m streamlit run app.py --server.headless false --server.port 8501

pause
