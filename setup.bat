@echo off
setlocal enabledelayedexpansion

echo Initializing AI Image Recognition Environment...

where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo uv not found. Please install it first: https://astral.sh/uv
    pause
    exit /b 1
)

echo Syncing dependencies...
uv sync
if %ERRORLEVEL% neq 0 (
    echo uv sync failed.
    pause
    exit /b %ERRORLEVEL%
)

if not exist "mlruns" (
    mkdir mlruns
)

echo Setup complete!
echo Run the app: uv run streamlit run app/app.py
pause