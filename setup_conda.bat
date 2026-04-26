@echo off
echo ==========================================================
echo   MineralForge - Conda Environment Setup
echo ==========================================================
echo.

where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda not found. Please install Anaconda or Miniconda.
    pause
    exit /b 1
)

echo [1/2] Creating or updating conda environment: jr_mineralforge
call conda env update -f environment.yml --prune
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to update Conda environment.
    pause
    exit /b 1
)

echo [2/2] Activating environment
call conda activate jr_mineralforge

echo.
echo [SUCCESS] Environment is ready.
echo [INFO] Try: python main.py --zone "Stope 3" --stress 2.8
echo ==========================================================
pause
