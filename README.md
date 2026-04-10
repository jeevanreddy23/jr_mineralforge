# 🏔️ JR MineralForge: Multi-Agent Geospatial Prospectivity

![License](https://img.shields.io/badge/license-MIT-blue)
![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-green)
![Orchestrator](https://img.shields.io/badge/orchestrator-Langchain_ReAct-orange)

**JR MineralForge** is an advanced, multi-agent AI system designed to democratize and automate mineral exploration across Australia. Built by Team JR, this system replicates and exceeds the strategies of the **OZ Minerals Explorer Challenge** winners by dynamically downloading state and national geophysical data, applying spatial machine learning, and discovering new Copper-Gold and VHMS (Volcanogenic Hosted Massive Sulfide) deposits.

Developed natively for local deployment with an integrated **Gradio** web dashboard.

---

## ✨ Core Capabilities

*   **Multi-Agent Orchestration:** Specialized autonomous agents for specific domains:
    *   📡 `DataIngestionAgent`: Connects to SARIG (South Australia), GA (National), and MRT (Tasmania) via native WFS/ArcGIS APIs to bypass Firewalls.
    *   🗺️ `ProspectivityMappingAgent`: Aligns massive geophysical rasters (TMI, Gravity, Radiometrics) and runs Isolation Forest ML models to find structural anomalies.
    *   📊 `ReportingAgent`: Automatically generates precision drill target reports with SHAP feature importance graphs.
    *   🧠 `KnowledgeAgent`: Provides a RAG-backed chatbot loaded with the strategies of world-class exploration teams (DeepSightX, OreFox, Team Guru).
*   **National Scale Sweep:** Pre-configured bounding boxes (Mount Woods, Pine Creek, Tasmanian Mount Read Volcanics) for immediate, on-the-fly execution.
*   **WAF-Resilient Network Backends:** Uses graceful fallback mechanisms mapping dynamic OGC web services when massive `.zip` grids are blocked by State firewalls.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.14+ installed. Due to the heavy geospatial processing requirements (GDAL), deploying via the included Docker container is highly recommended if your local Windows environment lacks native GDAL bindings.

### 2. Installation
```bash
git clone https://github.com/jeevanreddy23/jr_mineralforge.git
cd jr_mineralforge
pip install -r requirements.txt
```

### 3. Run the Core Platform
Launch the interactive Gradio operations dashboard:
```bash
python app.py
```
Open your browser to `http://localhost:7860`.

---

## ⚙️ Architecture

The system avoids legacy agent instability by leveraging a custom Langchain `AIMessage/ToolMessage` looping orchestrator. This guarantees precise parameter passing to tools like `run_national_sweep` and `check_data_availability`.

```mermaid
graph TD;
    GUI[Gradio Web Interface] --> Main[JR Orchestrator];
    Main --> DA[Data Ingestion (GA/MRT)];
    Main --> ML[Prospectivity AI];
    Main --> RAG[Geological Knowledge Base];
    DA --> ML;
    ML --> Folium[Interactive Maps];
```

---

## 🛠️ Usage Example (Tasmanian Sweep)
Want to find the next MMG Rosebery?
1. Open the UI, click **Ingest MRT (Tasmania)** to dynamically stream geological vectors from Mineral Resources Tasmania.
2. Click **Run Full Prospectivity Pipeline**.
3. View the final target heatmap on the interactive folium map or download the text-generated prospect report in the `reports/` directory!

---

*Built for the mission of unlocking Australia's critical mineral future.*
