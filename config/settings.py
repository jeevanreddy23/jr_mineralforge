"""
JR MineralForge - Central Configuration
========================================
Team JR – Advancing Mineral Discovery with Robust AI & Open Australian Geodata

All configuration constants, paths, and environment-driven settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────
# Root Paths
# ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
MLFLOW_DIR = ROOT_DIR / "mlflow_runs"

# Ensure directories exist
for _dir in [RAW_DIR, PROCESSED_DIR, VECTOR_STORE_DIR, REPORTS_DIR, MODELS_DIR, LOGS_DIR, MLFLOW_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Branding
# ─────────────────────────────────────────────────────────────────
BRAND_NAME = "JR MineralForge"
TEAM_NAME = "Team JR"
BRAND_HEADER = "Team JR – Advancing Mineral Discovery with Robust AI & Open Australian Geodata"
BRAND_FOOTER = f"Produced by {BRAND_NAME} for {TEAM_NAME}."
WINNERS_CONTEXT = (
    "How Team JR improves upon the prize-winning strategies from the OZ Minerals "
    "Explorer Challenge using SARIG and GA open data…"
)

# ─────────────────────────────────────────────────────────────────
# Target Provinces: Major Australian Mineral Systems
# ─────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    name: str
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    description: str = ""
    epsg: int = 4326

    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat) — WGS84"""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    @property
    def centre(self) -> Tuple[float, float]:
        return ((self.min_lat + self.max_lat) / 2, (self.min_lon + self.max_lon) / 2)


AUSTRALIAN_PROVINCES = {
    "mount_woods": BoundingBox(
        name="Mount Woods / Prominent Hill",
        min_lon=134.5, max_lon=136.5, min_lat=-30.0, max_lat=-28.0,
        description="SA Gawler Craton - IOCG and Cu-Au targets."
    ),
    "yilgarn_craton": BoundingBox(
        name="Yilgarn Craton (Kalgoorlie Region)",
        min_lon=120.0, max_lon=122.5, min_lat=-31.5, max_lat=-29.5,
        description="WA - World-class gold, nickel, and lithium province."
    ),
    "mt_isa_inlier": BoundingBox(
        name="Mount Isa Inlier",
        min_lon=138.5, max_lon=140.5, min_lat=-21.5, max_lat=-19.5,
        description="QLD - Major base metal (Pb-Zn-Ag) and Cu-Au province."
    ),
    "lachlan_fold_belt": BoundingBox(
        name="Lachlan Fold Belt (Orange/Parkes)",
        min_lon=147.5, max_lon=149.5, min_lat=-34.0, max_lat=-32.0,
        description="NSW - Porphyry Cu-Au and orogenic gold targets."
    ),
    "pilbara_craton": BoundingBox(
        name="Pilbara Craton (Iron Ore/Gold)",
        min_lon=116.0, max_lon=121.0, min_lat=-23.0, max_lat=-20.0,
        description="WA - Hematite iron ore and Archean gold deposits."
    ),
    "pine_creek": BoundingBox(
        name="Pine Creek Orogen",
        min_lon=130.5, max_lon=132.5, min_lat=-14.5, max_lat=-12.5,
        description="NT - Gold and Uranium rich district."
    ),
    "tasmania_mount_read": BoundingBox(
        name="Tasmania Mount Read (MMG Rosebery)",
        min_lon=145.0, max_lon=146.0, min_lat=-42.5, max_lat=-41.0,
        description="TAS - Polymetallic base metal (Zn-Pb-Cu-Ag-Au) VHMS deposits."
    )
}

# Default target for initialization
MOUNT_WOODS_BBOX = AUSTRALIAN_PROVINCES["mount_woods"]

# ─────────────────────────────────────────────────────────────────
# Open Data Source URLs
# ─────────────────────────────────────────────────────────────────

# SARIG (South Australia Resources Information Gateway)
SARIG_BASE_URL = "https://map.sarig.sa.gov.au"
SARIG_CATALOG_URL = "https://catalog.sarig.sa.gov.au"
SARIG_WFS_URL = "https://map.sarig.sa.gov.au/arcgis/services/SARIGServicesGroup/MapServer/WFSServer"
SARIG_WCS_URL = "https://map.sarig.sa.gov.au/arcgis/services/SARIGServicesGroup/MapServer/WCSServer"

# Geoscience Australia national/regional layers
GA_GEOPHYSICS_URL = "https://services.ga.gov.au/gis/rest/services"
GA_ELEVATION_URL = "https://services.ga.gov.au/site_9/services/DEM_SRTM_1Second_Hydro_Enforced/MapServer/WCSServer"
GA_MINERAL_POTENTIAL_URL = "https://services.ga.gov.au/site_7/services/EFTF_Mineral_Potential_Copper_Gold/MapServer/WFSServer"
GA_SURFACE_GEOLOGY_URL = "https://services.ga.gov.au/site_7/services/GA_Surface_Geology/MapServer/WFSServer"
GA_OZMIN_URL = "https://services.ga.gov.au/site_7/services/GA_OZMIN_Mineral_Deposits/MapServer/WFSServer"

# State-level Geological Survey Portals (WFS/WMS)
STATE_PORTALS = {
    "WA": "https://dasc.dmirs.wa.gov.au/arcgis/services/Geology/MapServer/WFSServer",
    "QLD": "https://georesources.dmp.qld.gov.au/arcgis/services/GeoscientificInformation/MapServer/WFSServer",
    "NSW": "https://services.ga.gov.au/gis/rest/services/NSW_Geology/MapServer/WFSServer",
}

# Specific open geophysics grids (pre-known direct links from GA)
GA_GRID_ENDPOINTS = {
    "magnetics_total_intensity": "https://dap.nci.org.au/thredds/remoteCatalogService?catalog=http://dapds00.nci.org.au/thredds/catalog/rr2/national-geophysical-compilations/magnetics/catalog.xml",
    "gravity_bouguer": "https://dap.nci.org.au/thredds/remoteCatalogService?catalog=http://dapds00.nci.org.au/thredds/catalog/rr2/national-geophysical-compilations/gravity/catalog.xml",
    "radiometrics_k": "https://dap.nci.org.au/thredds/remoteCatalogService?catalog=http://dapds00.nci.org.au/thredds/catalog/rr2/national-geophysical-compilations/radiometrics/catalog.xml",
}

# ─────────────────────────────────────────────────────────────────
# Embedding & LLM Configuration
# ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # set to "cuda" if GPU available
EMBEDDING_BATCH_SIZE = 64

# LLM backend — use Ollama locally or OpenAI API
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "openai" | "anthropic"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Vector store
VECTOR_STORE_BACKEND = "faiss"  # "faiss" | "chroma"
VECTOR_STORE_PATH = str(VECTOR_STORE_DIR)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 6
RETRIEVAL_SCORE_THRESHOLD = 0.35  # anti-noise: discard low-relevance chunks

# ─────────────────────────────────────────────────────────────────
# ML Model Configuration
# ─────────────────────────────────────────────────────────────────
@dataclass
class MLConfig:
    # Random state for reproducibility
    random_state: int = 42
    test_size: float = 0.2

    # Spatial cross-validation
    n_spatial_folds: int = 5
    spatial_buffer_km: float = 10.0  # exclusion buffer between train/test

    # Ensemble models
    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 5  # anti-overfitting

    xgb_n_estimators: int = 500
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    xgb_reg_alpha: float = 0.1   # L1 regularisation (anti-overfitting)
    xgb_reg_lambda: float = 1.0  # L2 regularisation
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_early_stopping_rounds: int = 30

    lgb_n_estimators: int = 500
    lgb_num_leaves: int = 31
    lgb_learning_rate: float = 0.05
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 1.0
    lgb_min_child_samples: int = 20  # anti-overfitting

    # Isolation Forest (anti-noise)
    isolation_forest_contamination: float = 0.05

    # SHAP feature selection
    shap_feature_fraction: float = 0.8  # keep top 80% SHAP-ranked features

    # Calibration
    calibration_method: str = "sigmoid"  # "sigmoid" | "isotonic"

    # Uncertainty quantification
    n_bootstrap: int = 50

    # Target mineral system
    target_mineral_system: str = "IOCG"
    positive_deposit_types: List[str] = field(default_factory=lambda: [
        "IOCG", "Copper-gold", "Iron oxide copper gold",
        "Magnetite-apatite", "Skarn", "Porphyry Cu-Au"
    ])


ML_CONFIG = MLConfig()

# ─────────────────────────────────────────────────────────────────
# Geospatial Configuration
# ─────────────────────────────────────────────────────────────────
TARGET_CRS = "EPSG:32754"   # UTM Zone 54S — appropriate for Mt Woods
WGS84_CRS = "EPSG:4326"
TARGET_RESOLUTION_M = 500   # 500 m grid cell for prospectivity map
RASTER_NODATA = -9999.0

# Wavelet / median filter (anti-noise for geophysics grids)
FILTER_WINDOW_SIZE = 5      # pixels, for median filtering
WAVELET_LEVEL = 3           # decomposition levels

# ─────────────────────────────────────────────────────────────────
# MLflow
# ─────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_PATH = ROOT_DIR / "mlflow_runs"
MLFLOW_TRACKING_URI = f"file:///{MLFLOW_TRACKING_PATH.as_posix()}"
MLFLOW_EXPERIMENT_NAME = "JR_MineralForge_Prospectivity"

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "jr_mineralforge.log"

# ─────────────────────────────────────────────────────────────────
# Previous Winners Knowledge Base
# ─────────────────────────────────────────────────────────────────
WINNERS_KB_PATH = DATA_DIR / "winners_knowledge"
WINNERS_KB_PATH.mkdir(parents=True, exist_ok=True)

OZ_MINERALS_CHALLENGE_WINNERS = [
    {
        "team": "Team Guru",
        "prize": "1st Prize",
        "year": 2019,
        "challenge": "OZ Minerals Explorer Challenge",
        "key_tactics": [
            "Mineral systems framework (IOCG proxies)",
            "Geophysical feature engineering (TMI derivatives, gravity residuals)",
            "Random Forest with spatial cross-validation",
            "Integration of multiple open datasets",
            "Clear geological interpretation layer",
        ],
    },
    {
        "team": "DeepSightX",
        "prize": "2nd Prize",
        "year": 2019,
        "challenge": "OZ Minerals Explorer Challenge",
        "key_tactics": [
            "Deep learning on geophysical images (CNN)",
            "Transfer learning from global mineral deposit data",
            "Ensemble with traditional ML",
            "Uncertainty quantification via MC dropout",
        ],
    },
    {
        "team": "deCODES",
        "prize": "3rd Prize",
        "year": 2019,
        "challenge": "OZ Minerals Explorer Challenge",
        "key_tactics": [
            "Bayesian belief networks",
            "Evidence weighting by expert elicitation",
            "Multi-commodity IOCG targeting",
            "SHAP-driven feature importance reporting",
        ],
    },
    {
        "team": "SRK Consulting",
        "prize": "Fusion Prize",
        "year": 2019,
        "challenge": "OZ Minerals Explorer Challenge",
        "key_tactics": [
            "Data fusion (geophysics + geochemistry + geology)",
            "Weighted overlay GIS analysis",
            "Expert geological review embedded in workflow",
            "Transparent scoring rubric",
        ],
    },
    {
        "team": "OreFox",
        "prize": "Special Mention",
        "year": 2019,
        "challenge": "OZ Minerals Explorer Challenge",
        "key_tactics": [
            "AI-driven target generation from public reports",
            "NLP extraction of historical exploration results",
            "Automated map digitization",
            "Ranking by multi-criteria decision analysis",
        ],
    },
]

# ─────────────────────────────────────────────────────────────────
# HTTP Request Settings
# ─────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 60   # seconds
REQUEST_RETRIES = 3
REQUEST_DELAY = 1.0    # seconds between requests (polite crawling)
USER_AGENT = f"{BRAND_NAME}/1.0 (Team JR; geoscience AI bot; contact@teamjr.example)"
