# 🏔️ JR MineralForge: Commercial "Drill or Drop" Prospectivity Pipeline

![License](https://img.shields.io/badge/license-MIT-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green)
![Status](https://img.shields.io/badge/status-Active-success)

**JR MineralForge** is an enterprise-grade Geotechnical Machine Learning pipeline designed for Australian mineral exploration companies. It transforms multi-layered geospatial and geophysical data into highly structured, actionable Copper-Gold drilling targets.

> **Note on Security and Auditing**: This repository contains **100% open-source Python code**. There are **no compiled `.exe` files, hidden tracking scripts, or obfuscated payloads**. All data ingestion is handled through official Australian geospatial REST and WFS architectures (SARIG & GA). You can freely audit every agent under `agents/` and the UI logic under `app.py`.

---

## ✨ Features and Capabilities

* **Dynamic Tenement Bounding:** Upload any custom `.shp`, `.geojson`, or `.kml` exploration tenement boundary. The pipeline intelligently automatically parses EPSG coordinates to structure the computation bounds.
* **SARIG & GA OGC API Ingestion:** Fully circumvents massive legacy `.zip` data downloads. It dynamically intersects and fetches `sa_geology_1m`, faults, and drillholes from the official South Australian Resources Information Gateway WFS nodes at runtime.
* **Isolation Forest ML Pipeline:** An unsupervised machine learning grid processes massive geochemical and geophysical datasets to isolate highly anomalous mineral prospect signatures.
* **SHAP-Driven Explainability:** Replaces "black box" intelligence by mapping `shap.TreeExplainer` anomalies against your targets, calculating exact geologic drivers (e.g., *Radiometric U Anomaly*, *Proximal to Major Shear*).
* **"Drill or Drop" Executive Dashboard:** A natively bound local interface running on Gradio that instantly outputs quantitative KPIs (Confidence %, Dataset Processing Volumes, Target IDs).

---

## 🚀 Installation & Setup

This repository is designed specifically for transparent virtual environment installations.

### 1. Requirements
* Git
* **Python 3.10 to 3.14+** (Recommended)
* **GDAL / GeoPandas System Libraries** (Geospatial bounds processing requires a native C++ compiler or pre-compiled wheel binaries on Windows)

### 2. Standard Installation

Clone the repository and install the clear-text constraints:

```bash
git clone https://github.com/jeevanreddy23/jr_mineralforge.git
cd jr_mineralforge

# Create a virtual environment to isolate dependencies
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on Mac/Linux
source venv/bin/activate

# Install exact dependencies
pip install -r requirements.txt
```

*(Note: If you run into GDAL geometry errors on Windows, download the pre-compiled `GDAL` and `Fiona` wheels from Christopher Gohlke's registry and pip install those first).*

### 3. Launching the App

Never run a random un-audited executable. This system runs transparently via standard python execution:

```bash
python app.py
```

Navigate to `http://localhost:7860` to access the Pipeline GUI in your browser.

---

## ⚙️ How It Works (The Pipeline)

1. **Upload:** User drops a `.shp` into the Gradio UI.
2. **Ingest (Data Ingestion Agent):** GeoPandas parses the `BoundingBox`. OWSLib makes `GetFeature` requests to SARIG `version=1.1.0` endpoints.
3. **Train (ML Engine Agent):** Generates numerical grid arrays. Isolation Forest identifies multivariate spacing anomalies. 
4. **Rank (Prospectivity Agent):** Top targets are graded by confidence (0% to 100%). SHAP is run over the tree to extract the dominant `primary_driver`.
5. **Output:** Rendered onto an interactive Folium web map alongside an actionable CSV/Dataframe export.

---

## 🛡️ Trust & Verification

* **Network Calls:** The app strictly makes outbound GET requests to `.gov.au` domains (`services.sarig.sa.gov.au` and Geoscience Australia APIs).
* **Dependencies:** Defined strictly in `requirements.txt`. Key heavyweights are `geopandas`, `shap`, `scikit-learn`, `rasterio`, and `gradio`. No sketchy third-party arbitrary libraries.
* **Open Source:** Feel free to submit PRs or raise issues to help support transparent exploration intelligence. 

---

## 🛠️ JR MineralForge: Commercial Hardening & Deployment Roadmap

This repository has undergone extensive hardening to ensure production-level stability for Australian exploration teams. Below is the historical deployment log and strategic roadmap.

### Phase 1: Environment Stabilization
* **Dependency Unpinning**: Resolved `ResolutionImpossible` errors in Anaconda/Windows environments by optimizing package version constraints.
* **Pre-flight Manager**: Integrated automated dependency healing on application launch.
* **Socket Management**: Fixed `Errno 10048` (port conflicts) by refining Gradio's server binding logic.

### Phase 2: Pipeline Robustness & Fallbacks
* **Geometry-Aware Synthetic Data**: Patched the `data_ingestion_agent` to support custom KML/SHP uploads. If live OGC data is unreachable, the system generates high-fidelity synthetic grids perfectly aligned with the user-provided geometry.
* **XArray Alignment**: Fixed 2D-to-1D coordinate mapping in the feature datacube assembly, ensuring reliable ML inputs.
* **Unicode Consolidation**: Switched all logging to ASCII standard to prevent terminal crashes on legacy Windows `cp1252` environments.

### Phase 3: ML Engine & Weights
* **Model Bootstrapping**: Added `scripts/bootstrap_model.py` to initialize the `ml_engine.pkl` weights on fresh installs, enabling "Drill or Drop" analysis without requiring manual training first.
* **MLflow Tracking**: Optimized local tracking URIs for Windows filesystem absolute pathing schemas.

---
*Produced dynamically to leverage Open Australian Geodata securely and efficiently.*
