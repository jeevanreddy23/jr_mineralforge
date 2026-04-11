"""
JR MineralForge – Dependency Manager (Hardened)
================================================
Validates and attempts repair of the geospatial environment on startup.
Critically: REPORTS failures loudly instead of swallowing them silently.
"""

from __future__ import annotations
import subprocess
import sys
import importlib
import logging

log = logging.getLogger(__name__)

# Packages that require native C++ libs — pip alone often can't fix these.
# We attempt install but warn clearly if they fail.
NATIVE_PACKAGES = {
    "geopandas": "geopandas",
    "fiona": "fiona",
    "rasterio": "rasterio",
    "pyproj": "pyproj",
    "shapely": "shapely",
    "owslib": "owslib",
}

# Pure-Python packages — pip install should always work.
PURE_PACKAGES = {
    "sklearn": "scikit-learn",
    "shap": "shap",
    "folium": "folium",
    "pandas": "pandas",
    "numpy": "numpy",
    "gradio": "gradio",
    "dotenv": "python-dotenv",
    "langchain_core": "langchain-core",
}


def _try_install(pip_name: str) -> bool:
    """Attempt pip install. Returns True if successful."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0
    except Exception as e:
        log.warning(f"pip install {pip_name} raised: {e}")
        return False


def _check_import(import_name: str) -> bool:
    """Returns True if the module can be imported."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def check_and_install_dependencies() -> None:
    """
    Main pre-flight check. Validates all required packages.
    - Pure Python packages: attempts auto-install on failure.
    - Native/geospatial packages: attempts auto-install, but prints a clear
      actionable error message if it fails (rather than silently continuing).
    """
    print("[INFO] JR MineralForge — Running pre-flight dependency check...")
    
    failed_native = []
    failed_pure = []

    # --- Check pure Python packages ---
    for import_name, pip_name in PURE_PACKAGES.items():
        if not _check_import(import_name):
            log.warning(f"Missing: {pip_name}. Attempting auto-install...")
            if _try_install(pip_name):
                print(f"  [OK] Auto-installed: {pip_name}")
            else:
                failed_pure.append(pip_name)

    # --- Check native/geospatial packages ---
    for import_name, pip_name in NATIVE_PACKAGES.items():
        if not _check_import(import_name):
            log.warning(f"Missing native package: {pip_name}. Attempting install...")
            if _try_install(pip_name):
                # Re-check after install
                if _check_import(import_name):
                    print(f"  [OK] Auto-installed: {pip_name}")
                else:
                    # Installed but still can't import = native lib issue
                    failed_native.append(pip_name)
            else:
                failed_native.append(pip_name)

    # --- Report results ---
    if failed_pure:
        print(f"\n[WARN] Could not auto-install pure packages: {failed_pure}")
        print("    Run manually: pip install " + " ".join(failed_pure))

    if failed_native:
        print("\n" + "=" * 60)
        print("[ERROR] CRITICAL: Missing geospatial packages (native C++ required):")
        print(f"   {failed_native}")
        print()
        print("[INFO] FIX — Choose your platform:")
        print()
        print("  Linux/Mac:")
        print("    pip install geopandas fiona rasterio pyproj shapely owslib")
        print()
        print("  Windows (recommended — avoids GDAL build errors):")
        print("    conda install -c conda-forge geopandas fiona rasterio owslib")
        print()
        print("  Windows (pip-only):")
        print("    pip install pipwin && pipwin install gdal && pipwin install fiona")
        print("    pip install geopandas rasterio owslib")
        print("=" * 60 + "\n")

        # Raise so the Gradio UI shows a real error, not a silent crash
        raise EnvironmentError(
            f"Missing geospatial packages: {failed_native}. "
            "Install via conda: `conda install -c conda-forge geopandas fiona rasterio owslib` "
            "or see terminal for platform-specific instructions."
        )

    print("[OK] All dependencies satisfied. Pipeline ready.\n")
