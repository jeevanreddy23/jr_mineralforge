# 🏔️ JR MineralForge: Commercial "Drill or Drop" Prospectivity Pipeline

![License](https://img.shields.io/badge/license-MIT-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green)
![Status](https://img.shields.io/badge/status-Active-success)

**JR MineralForge** is an enterprise-grade Geotechnical Machine Learning pipeline designed for Australian mineral exploration companies. It transforms multi-layered geospatial and geophysical data into highly structured, actionable Copper-Gold drilling targets.

> **Note on Security and Auditing**: This repository contains **100% open-source Python code**. There are **no compiled `.exe` files, hidden tracking scripts, or obfuscated payloads**. All data ingestion is handled through official Australian geospatial REST and WFS architectures (SARIG & GA). You can freely audit every agent under `agents/` and the UI logic under `app.py`.

---

## ✨ Features and Capabilities (v2.1 Swarm-Enabled)

* **Context-Engineered Swarm:** Replaces legacy ingestion with a recursive multi-agent swarm (Geometry, Search, Fallback, Validator, Integrator).
* **High-Resonance Data Ingestion:** Uses 'Field Resonance' scoring to ensure data coverage. If resonance is low (<0.85), the swarm automatically expands its search radius and retries up to 3 times.
* **Intelligent KML Processing:** `GeometryAnalyzerAgent` calculates optimal `BoundingBox` for complex point clouds and polygons (e.g., `mindep_commod_all_con.kml`).
* **Geostatistical Geophysics Fallback:** Integrated `gstools` to generate simulated geophysical fields if actual OGC services (SARIG/GA) return partial results.
* **Dynamic Tenement Bounding:** Upload any custom `.shp`, `.geojson`, or `.kml` exploration tenement boundary. The pipeline intelligently parses EPSG coordinates.
* **SHAP-Driven Explainability:** Maps `shap.TreeExplainer` anomalies against your targets to calculate exact geologic drivers.

---

## 🚀 Installation & Setup

**JR MineralForge v2.1** is optimized for **Anaconda/Conda** to ensure a stable geospatial stack on Windows.

### 1. Requirements
* Git
* [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/free/miniconda/index.html)

### 2. Standard Installation (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/jeevanreddy23/jr_mineralforge.git
   cd jr_mineralforge
   ```

2. Run the automated setup script (Windows):
   ```bash
   setup_conda.bat
   ```
   *This script creates the `jr_mineralforge` environment, installs all geospatial dependencies (GDAL, GSTools, etc.), and configures required paths.*

3. Manual Installation (Conda):
   ```bash
   conda env create -f environment.yml
   conda activate jr_mineralforge
   ```

### 3. Launching the App
```bash
python app.py
```
Navigate to `http://localhost:7860` to access the v2.1 Swarm Dashboard.

---

## ⚙️ Swarm of Agents Architecture

The v2.1 upgrade introduces **Context Engineering** principles to data acquisition:

1. **GeometryAnalyzer**: Evaluates uploaded shapes and sets the computational field.
2. **ParallelSearchers**: Simultaneously queries SARIG and GA OGC/WFS endpoints.
3. **CoverageValidator**: Calculates a **Resonance score** based on data density and layer availability.
4. **FallbackGenerator**: If resonance is <0.85, it simulates high-fidelity layers to fill gaps.
5. **DataIntegrator**: Merges and cleans the finalized context payload for ML processing.
6. **SwarmControlLoop**: Orchestrates recursion and dynamic BBOX expansion if data retrieval is insufficient.

---

## 🛠️ JR MineralForge: Commercial Hardening & Deployment Roadmap

### Phase 3: Swarm Orchestration & v2.1 Release
* **Context Engineering Core**: Implemented `FieldResonanceMeasure` and `SwarmControlLoop`.
* **Resilient Ingestion**: Solved the "No geophysical data found" error for complex KMLs by implementing recursive BBOX expansion.
* **Gradio Monitoring**: Added the **Swarm Status Monitor** to the UI for real-time visibility into agent activities.
* **Geospatial Stack Stability**: Switched to Conda-based dependency management to resolve legacy DLL and GDAL binary issues on Windows.

---
*Produced dynamically to leverage Open Australian Geodata securely and efficiently.*

