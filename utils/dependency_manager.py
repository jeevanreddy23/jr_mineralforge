import sys
import subprocess
import importlib
import logging

log = logging.getLogger(__name__)

def check_and_install_dependencies():
    """
    Self-healing layer to detect and install missing geospatial dependencies before execution.
    Specifically tuned to handle complex libraries like geopandas and rasterio.
    """
    required_packages = {
        'geopandas': 'geopandas',
        'shapely': 'shapely',
        'fiona': 'fiona',
        'pyproj': 'pyproj',
        'rasterio': 'rasterio',
        'sklearn': 'scikit-learn'
    }

    missing_packages = []

    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        log.warning(f"Missing dependencies detected: {', '.join(missing_packages)}")
        print(f"[PROCESS] Installing missing dependencies: {', '.join(missing_packages)}...")
        
        try:
            # First attempt standard pip install
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("[OK] Dependencies installed successfully.")
        except subprocess.CalledProcessError:
            log.warning("Standard pip install failed. Attempting fallback for geospatial binaries...")
            print("[WAIT] Standard install failed. Attempting no-binary fallback...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-binary", ":all:"] + missing_packages)
                print("[OK] Fallback installation successful.")
            except subprocess.CalledProcessError as e:
                log.error(f"Critical failure installing geospatial dependencies: {e}")
                print("[ERROR] CRITICAL ERROR: Could not install dependencies. Please run 'conda install geopandas rasterio' manually.")
                raise e

# Provide an isolated import wrapper
def safe_import(module_name: str, fallback_package: str = None):
    """
    Wraps single library imports to self-heal on the fly.
    """
    pkg = fallback_package or module_name
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"[INFO] Auto-installing {pkg}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            return importlib.import_module(module_name)
        except Exception as e:
            print(f"[ERROR] Failed to install {pkg}: {e}")
            raise

if __name__ == "__main__":
    check_and_install_dependencies()
