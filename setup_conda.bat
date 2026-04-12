@echo off
echo ==========================================================
echo   JR MineralForge v2.1 - Anaconda Environment Setup
echo ==========================================================
echo.

:: Check for Conda
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda not found. Please install Anaconda or Miniconda.
    pause
    exit /b 1
)

echo [1/3] Creating/Updating Conda environment: jr_mineralforge...
call conda env update -f environment.yml --prune

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to update Conda environment.
    pause
    exit /b 1
)

echo [2/3] Activating environment...
call conda activate jr_mineralforge

echo [3/3] Setting up GDAL environment variables...
:: Set GDAL paths to ensure compatibility on Windows
set GDAL_DATA=%CONDA_PREFIX%\Library\share\gdal
set PROJ_LIB=%CONDA_PREFIX%\Library\share\proj

echo.
echo [SUCCESS] Environment is ready.
echo [INFO] To start the application, run: python app.py
echo ==========================================================
pause
