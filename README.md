# 🏔️ JR MineralForge

**Produced by JR MineralForge for Team JR**
> *Team JR – Advancing Mineral Discovery with Robust AI & Open Australian Geodata*

JR MineralForge is a production-ready, multi-agent AI system built with **LangChain** for identifying and ranking potential mineral deposits (especially IOCG, copper-gold) in the **Mount Woods / Prominent Hill** region using **only publicly available open data** from Australian government sources (SARIG + Geoscience Australia).

---

## 📋 Table of Contents
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Setup (Local)](#setup-local)
- [Setup (Docker)](#setup-docker)
- [Data Sources](#data-sources)
- [Bounding Box Configuration](#bounding-box-configuration)
- [CLI Usage](#cli-usage)
- [Gradio UI](#gradio-ui)
- [How JR MineralForge Improves on OZ Minerals Challenge Winners](#how-jr-mineralforge-improves-on-oz-minerals-challenge-winners)
- [Anti-Noise & Anti-Overfitting](#anti-noise--anti-overfitting)
- [Project Structure](#project-structure)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│             JR MineralForge – Multi-Agent System                │
│                 (LangChain ReAct Orchestrator)                  │
└───────────────────┬─────────────────────────────────────────────┘
                    │
        ┌───────────┼────────────────────────────┐
        ▼           ▼            ▼               ▼               ▼
┌──────────┐ ┌──────────┐ ┌──────────┐  ┌──────────────┐ ┌──────────┐
│  Data    │ │Knowledge │ │   ML     │  │Prospectivity │ │Reporting │
│Ingestion │ │& Winners │ │ Engine   │  │  Mapping     │ │& Explain │
│  Agent   │ │  Agent   │ │  Agent   │  │   Agent      │ │  Agent   │
│(SARIG/GA)│ │(RAG/FAISS│ │(RF+XGB  │  │(Folium/GPKG) │ │(SHAP/PDF)│
└────┬─────┘ └────┬─────┘ │+LGB+CV) │  └──────┬───────┘ └──────────┘
     │            │       └──────────┘         │
     ▼            ▼                            ▼
  SARIG       FAISS                      GeoTIFF /
  GA Open     Vector                    GeoPackage /
  Data        Store                     Folium Map
```

### 5 Agents

| Agent | Role |
|-------|------|
| **Data Ingestion Agent** | Auto-downloads SARIG + GA open data, reprojects to UTM, clips to bbox, applies anti-noise filtering |
| **Geological Knowledge & Winners Agent** | RAG over geoscience knowledge + OZ Minerals winner analysis |
| **Robust ML Engineering Agent** | Spatial CV, ensemble models (RF/XGB/LGB), SHAP, uncertainty, MLflow |
| **Prospectivity Mapping & Target Generation Agent** | Generates ranked drill targets, interactive maps, GIS exports |
| **Explanation, Interpretability & Reporting Agent** | SHAP plots, branded reports, geological explanations |

---

## Quick Start

```bash
# 1. Clone and enter
git clone <your-repo-url> jr_mineralforge
cd jr_mineralforge

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
copy .env.example .env           # Windows
# cp .env.example .env           # Linux/Mac
# Edit .env as needed

# 5. Launch the Gradio UI
python app.py
# → Open http://localhost:7860

# OR use the CLI
python main.py workflow          # Full end-to-end run
python main.py chat              # Interactive geological chat
```

---

## Setup (Local)

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) (for local LLM — free, no API key needed)
- GDAL system libraries (for geospatial processing)

### Install GDAL on Windows
```powershell
# Option A: via OSGeo4W (recommended)
# Download https://trac.osgeo.org/osgeo4w/ and install GDAL

# Option B: via conda (easiest)
conda install -c conda-forge gdal rasterio geopandas
```

### Install and Pull Ollama Model
```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.1:8b          # ~4.7 GB download
```

### Environment Variables
Copy `.env.example` to `.env` and configure:

```env
# Option 1: Ollama (local, free — recommended)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434

# Option 2: OpenAI
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o-mini
# OPENAI_API_KEY=sk-...

# Option 3: Anthropic
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-3-haiku-20240307
# ANTHROPIC_API_KEY=sk-ant-...
```

---

## Setup (Docker)

Docker Compose handles everything: Ollama LLM server, JR MineralForge app, and MLflow tracking.

```bash
# Build and start all services
docker-compose up --build

# Access:
#   Gradio UI:   http://localhost:7860
#   MLflow:      http://localhost:5000
#   Ollama API:  http://localhost:11434

# Pull the LLM model into the Ollama container
docker exec jr_ollama ollama pull llama3.1:8b

# Run the CLI workflow in a one-shot container
docker-compose --profile cli run jr_cli

# Stop everything
docker-compose down
```

---

## Data Sources

### SARIG — South Australia Resources Information Gateway
**Portal:** https://map.sarig.sa.gov.au  
**Catalog:** https://catalog.sarig.sa.gov.au

| Dataset | Type | What JR MineralForge downloads |
|---------|------|-------------------------------|
| SA Surface Geology 1:250k | Shapefile | Lithology units, contacts |
| Mineral Occurrences (MINOCC) | Shapefile | IOCG/Cu-Au deposit locations |
| Drillhole Collars | Shapefile | Historical drill locations |
| Geochemistry (soils) | CSV | Cu, Au, Fe, K geochemistry |
| WFS services | GeoJSON | Real-time bbox-filtered queries |

### Geoscience Australia
**Portal:** https://portal.ga.gov.au  
**GADDS:** https://portal.ga.gov.au/persona/gadds  
**NCI/DAP:** https://dap.nci.org.au

| Dataset | Type | What JR MineralForge downloads |
|---------|------|-------------------------------|
| National Magnetic Compilation (TMI) | ERMapper Grid → GeoTIFF | Total magnetic intensity |
| National Gravity Compilation | ERMapper Grid → GeoTIFF | Bouguer anomaly |
| Radiometric Map of Australia | ERMapper Grid → GeoTIFF | K%, eTh, eU, Total Dose |
| 1 Second SRTM DEM | GeoTIFF | Elevation |
| OZMIN Occurrences | WFS GeoJSON | National mineral occurrences |

---

## Bounding Box Configuration

The default study area is **Mount Woods / Prominent Hill** (Gawler Craton, SA):

```python
# config/settings.py
MOUNT_WOODS_BBOX = BoundingBox(
    name="Mount Woods / Prominent Hill",
    min_lon=134.5,
    max_lon=136.5,
    min_lat=-30.0,
    max_lat=-28.0,
)
```

**To change the study area**, edit `config/settings.py`:
```python
MY_BBOX = BoundingBox(
    name="My Study Area",
    min_lon=135.0,   # West longitude (decimal degrees, WGS84)
    max_lon=137.0,   # East longitude
    min_lat=-31.0,   # South latitude (negative in southern hemisphere)
    max_lat=-29.0,   # North latitude
)
```

Then pass it to agents:
```python
from agents.data_ingestion_agent import SARIGIngestionAgent
agent = SARIGIngestionAgent(bbox=MY_BBOX)
```

---

## CLI Usage

```bash
# Interactive geological chat
python main.py chat
# → Prompt: "Show top copper-gold targets in Mount Woods from SARIG data"
# → Prompt: "How did Team Guru win and how does JR MineralForge improve it?"

# Single query
python main.py chat --query "What geophysical features indicate IOCG mineralisation?"

# Full end-to-end workflow (ingest → ML → targets → report)
python main.py workflow

# Download SARIG + GA open data only
python main.py ingest

# Generate branded report
python main.py report

# Rebuild RAG knowledge base (after adding new docs to data/)
python main.py rebuild-kb

# Use a specific LLM backend
python main.py chat --provider openai --model gpt-4o-mini --query "..."
python main.py chat --provider ollama --model mistral:7b --query "..."
```

---

## Gradio UI

Launch with `python app.py` and open **http://localhost:7860**

| Tab | Features |
|-----|---------|
| 💬 **Geological Chat** | RAG-powered geological Q&A, winner comparisons, target queries |
| 📥 **Data Ingestion** | One-click SARIG + GA download with status feedback |
| 🗺️ **Prospectivity Map** | Run full ML pipeline, view interactive Folium map inline |
| 🏆 **Winner Analysis** | Browse and compare all OZ Minerals Explorer Challenge winning strategies |
| 📄 **Reports** | Generate and download full branded report, rebuild knowledge base |

**Example chat queries:**
```
"Show top copper-gold targets in Mount Woods from SARIG data"
"How did Team Guru win and how does JR MineralForge improve it with GA geophysics?"
"What radiometric proxies should I use for potassic alteration targeting?"
"Compare DeepSightX strategy to Team JR approach"
"Explain the IOCG mineral system in the Gawler Craton"
```

---

## How JR MineralForge Improves on OZ Minerals Challenge Winners

> *"How Team JR improves upon the prize-winning strategies from the OZ Minerals Explorer Challenge using SARIG and GA open data…"*

| Winner | Prize | JR MineralForge Improvement |
|--------|-------|----------------------------|
| **Team Guru** | 1st | Extends their mineral systems framework with automated SARIG/GA ingestion pipelines and LangChain RAG for continuous knowledge accumulation |
| **DeepSightX** | 2nd | Replaces MC dropout uncertainty with bootstrap ensemble (N=50) and adds buffered spatial CV to eliminate data leakage |
| **deCODES** | 3rd | Uses SHAP instead of Bayesian belief networks for geological feature-level explainability at each target |
| **SRK Consulting** | Fusion | Automates multi-dataset fusion via xarray datacube stacking with standardised CRS/resolution alignment |
| **OreFox** | Special | LangChain FAISS RAG replaces manual NLP with a persistent, updatable knowledge base over all open reports |

---

## Anti-Noise & Anti-Overfitting

### Anti-Noise Measures
| Technique | Where | Effect |
|-----------|-------|--------|
| Wavelet soft-threshold denoising | `utils/geospatial_utils.py` | Removes high-frequency noise from geophysical grids |
| Median spatial filtering (5×5) | `utils/geospatial_utils.py` | Suppresses single-cell spikes in magnetics/gravity |
| Isolation Forest outlier removal | `agents/ml_engine_agent.py` | Flags and removes 5% most anomalous training samples |
| RAG score threshold (0.35) | `rag/rag_setup.py` | Discards low-relevance retrieved chunks |
| 3.5σ feature clipping | `agents/ml_engine_agent.py` | Caps extreme feature values before ML training |

### Anti-Overfitting Measures
| Technique | Where | Effect |
|-----------|-------|--------|
| Spatial block CV (10 km buffer) | `agents/ml_engine_agent.py` | Prevents spatial data leakage between folds |
| XGBoost L1/L2 regularisation | `agents/ml_engine_agent.py` | Penalises overly complex trees |
| LightGBM `min_child_samples=20` | `agents/ml_engine_agent.py` | Prevents leaf-level overfitting |
| RF `min_samples_leaf=5` | `agents/ml_engine_agent.py` | Prevents deep trees memorising training data |
| SHAP feature selection (top 80%) | `agents/ml_engine_agent.py` | Removes near-zero-contribution features |
| Platt/isotonic calibration | `agents/ml_engine_agent.py` | Corrects overconfident probability outputs |
| Bootstrap uncertainty (N=50) | `agents/ml_engine_agent.py` | Quantifies epistemic uncertainty per prediction |

---

## Project Structure

```
jr_mineralforge/
├── config/
│   ├── __init__.py
│   └── settings.py              # Central configuration
├── agents/
│   ├── __init__.py
│   ├── data_ingestion_agent.py  # SARIG + GA data download
│   ├── knowledge_agent.py       # RAG geological Q&A + winners
│   ├── ml_engine_agent.py       # Spatial CV + ensemble ML
│   ├── prospectivity_agent.py   # Map + target generation
│   └── reporting_agent.py       # Branded reports + SHAP plots
├── rag/
│   ├── __init__.py
│   └── rag_setup.py             # FAISS vector store + embeddings
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py
│   ├── geospatial_utils.py      # Raster processing + anti-noise
│   └── llm_factory.py           # LLM backend switcher
├── tests/
│   ├── test_ingestion.py
│   ├── test_ml_engine.py
│   └── test_rag.py
├── data/                        # Created automatically
│   ├── raw/                     # Downloaded SARIG/GA data
│   ├── processed/               # Reprojected + clipped rasters
│   ├── vector_store/            # FAISS index
│   └── winners_knowledge/       # Winner knowledge text files
├── reports/                     # Generated outputs
│   ├── jr_mineralforge_report.txt
│   ├── jr_prospectivity_map.html
│   ├── ranked_targets.csv
│   ├── ranked_targets.gpkg
│   └── shap_importance.png
├── models/                      # Saved ML models
├── logs/                        # Log files
├── mlflow_runs/                 # MLflow experiment tracking
├── main.py                      # CLI orchestrator
├── app.py                       # Gradio UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## MLflow Experiment Tracking

MLflow automatically logs all training runs. Access the dashboard:

```bash
# Local
mlflow ui --backend-store-uri ./mlflow_runs
# → http://localhost:5000

# Docker (already running)
# → http://localhost:5000
```

Each run tracks:
- `cv_mean_auc` / `cv_std_auc` — Spatial CV performance
- `n_features` — Features after SHAP selection
- `n_train_samples` — Training set size
- `positive_rate` — Class balance (deposit / background ratio)

---

## Contributing & Self-Improvement Loop

JR MineralForge supports incremental knowledge updates. When new SARIG/GA data is released:

```bash
# Drop new PDFs, CSVs, or text reports into data/raw/
# Then rebuild the knowledge base
python main.py rebuild-kb

# OR via the orchestrator's self-improvement loop
python -c "
from main import JRMineralForgeOrchestrator
from pathlib import Path
orc = JRMineralForgeOrchestrator()
orc.self_improvement_loop(new_data_dir=Path('data/raw/new_release'))
"
```

---

## Licence & Disclaimer

All data used is sourced from Australian government open data portals under their respective licences:
- **SARIG**: © South Australian Government, CC-BY 4.0
- **Geoscience Australia**: © Commonwealth of Australia, CC-BY 4.0

Results are probabilistic and must be validated by a qualified geoscientist before any exploration decision.

---

*Produced by **JR MineralForge** for **Team JR**.*
