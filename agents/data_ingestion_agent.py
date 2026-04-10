"""
JR MineralForge – Data Ingestion Agent
========================================
Handles automated download and processing of open geospatial data from:
  - SARIG (South Australia Resources Information Gateway)
  - Geoscience Australia (GADDS, OZMIN, WCS/WFS services)

Anti-noise measures are applied during ingestion (outlier detection,
median/wavelet filtering on geophysical grids).
"""

from __future__ import annotations

import io
import os
import time
import zipfile
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlencode, urljoin

import requests
import numpy as np

from config.settings import (
    RAW_DIR, PROCESSED_DIR, MOUNT_WOODS_BBOX, BoundingBox,
    SARIG_BASE_URL, SARIG_CATALOG_URL, SARIG_WFS_URL, SARIG_WCS_URL,
    GA_GEOPHYSICS_URL, GA_ELEVATION_URL, GA_MINERAL_POTENTIAL_URL, 
    GA_SURFACE_GEOLOGY_URL, GA_OZMIN_URL,
    REQUEST_TIMEOUT, REQUEST_RETRIES, REQUEST_DELAY, USER_AGENT,
    RASTER_NODATA, TARGET_CRS, TARGET_RESOLUTION_M,
    ML_CONFIG,
)
from utils.logging_utils import get_logger
from utils.geospatial_utils import (
    reproject_raster, clip_raster_to_bbox,
    median_filter_grid, wavelet_denoise_grid,
)

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# HTTP Session
# ─────────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _download_with_retry(
    url: str,
    dest_path: Path,
    session: Optional[requests.Session] = None,
    params: Optional[Dict] = None,
) -> Path:
    """Download a URL to dest_path with retry logic. Returns dest_path."""
    if session is None:
        session = _make_session()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            log.info(f"[attempt {attempt}] Downloading: {url}")
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            log.info(f"  [OK] Saved -> {dest_path} ({dest_path.stat().st_size:,} bytes)")
            time.sleep(REQUEST_DELAY)
            return dest_path
        except requests.RequestException as e:
            status = getattr(e.response, "status_code", None)
            if status in (403, 401, 500, 503):
                log.error(f"  [ERROR] Server rejected request (HTTP {status}). WAF block or portal migration suspected.")
                raise RuntimeError(f"Server rejected request (HTTP {status}). SARIG automated downloads may be blocked by WAF.")
            
            log.warning(f"  [WARN] Attempt {attempt} failed: {e}")
            if attempt < REQUEST_RETRIES:
                time.sleep(REQUEST_DELAY * attempt * 2)
    raise RuntimeError(f"Failed to download {url} after {REQUEST_RETRIES} attempts")


def _extract_zip(zip_path: Path, dest_dir: Path) -> List[Path]:
    """Extract all files from a ZIP archive. Returns list of extracted paths."""
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
        for name in z.namelist():
            p = dest_dir / name
            if p.is_file():
                extracted.append(p)
    log.info(f"Extracted {len(extracted)} files from {zip_path.name}")
    return extracted


def _cache_key(url: str, params: Optional[Dict] = None) -> str:
    raw = url + (json.dumps(params, sort_keys=True) if params else "")
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────
# SARIG Data Ingestion
# ─────────────────────────────────────────────────────────────────

class SARIGIngestionAgent:
    """
    Ingests open data from the South Australian Resources Information Gateway.
    Datasets: magnetics, gravity, radiometrics, drillholes, geochemistry,
              surface geology, mineral occurrences, exploration reports.
    """

    SARIG_OPEN_DATA_PACKAGES = {
        "surface_geology": {
            "description": "SA Surface Geology 1:250k",
            "url": "https://catalog.sarig.sa.gov.au/geonetwork/srv/api/records/sa-geology-250k/attachments/SurfaceGeology_250k.zip",
            "format": "shapefile",
        },
        "mineral_occurrences": {
            "description": "SA Mineral Occurrences (MINOCC)",
            "url": "https://catalog.sarig.sa.gov.au/geonetwork/srv/api/records/mineral-occurrences/attachments/MineralOccurrences.zip",
            "format": "shapefile",
        },
        "drillholes": {
            "description": "SA Drillhole Collars",
            "url": "https://catalog.sarig.sa.gov.au/geonetwork/srv/api/records/drillhole-collars/attachments/DrillholeCollars.zip",
            "format": "shapefile",
        },
    }

    # WFS endpoint for bbox-filtered queries
    WFS_URL = "https://map.sarig.sa.gov.au/arcgis/services/SARIGServicesGroup/MapServer/WFSServer"

    def __init__(self, bbox: BoundingBox = MOUNT_WOODS_BBOX):
        self.bbox = bbox
        self.session = _make_session()
        # Create province-specific subdirectories
        safe_name = bbox.name.lower().replace(" ", "_").replace("/", "_")
        self.raw_dir = RAW_DIR / safe_name / "sarig"
        self.processed_dir = PROCESSED_DIR / safe_name / "sarig"
        
        # Check if bbox is in SA (approximate)
        self.is_sa = (129.0 <= bbox.min_lon <= 141.0) and (-38.2 <= bbox.min_lat <= -26.0)
        
        if self.is_sa:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_package(self, key: str) -> List[Path]:
        """Download and extract a named SARIG open data package."""
        if not self.is_sa:
            log.warning(f"Skipping SARIG package {key}: Bounding box outside SA.")
            return []
        pkg = self.SARIG_OPEN_DATA_PACKAGES.get(key)
        if not pkg:
            raise ValueError(f"Unknown SARIG package: {key}")
        dest_zip = self.raw_dir / f"{key}.zip"
        if dest_zip.exists():
            log.info(f"Cache hit: {dest_zip}")
        else:
            _download_with_retry(pkg["url"], dest_zip, session=self.session)
        extract_dir = self.raw_dir / key
        extract_dir.mkdir(exist_ok=True)
        return _extract_zip(dest_zip, extract_dir)

    def fetch_wfs_layer(
        self,
        type_name: str,
        output_format: str = "application/json",
        max_features: int = 5000,
    ) -> Optional[Path]:
        """Fetch a WFS layer filtered to the bounding box as GeoJSON."""
        bbox_str = f"{self.bbox.min_lon},{self.bbox.min_lat},{self.bbox.max_lon},{self.bbox.max_lat}"
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": type_name,
            "outputFormat": output_format,
            "bbox": bbox_str,
            "count": max_features,
        }
        cache_name = f"wfs_{type_name.replace(':', '_').replace('/', '_')}.geojson"
        out_path = self.processed_dir / cache_name
        if out_path.exists():
            log.info(f"WFS cache hit: {out_path}")
            return out_path
        try:
            _download_with_retry(self.WFS_URL, out_path, session=self.session, params=params)
            return out_path
        except Exception as e:
            log.error(f"WFS fetch failed for {type_name}: {e}")
            return None

    def fetch_sarig_geochemistry(self) -> Optional[Path]:
        """Download geochemistry data (soil/rock geochemistry CSV from SARIG)."""
        # SARIG Geochemistry CSV download – filtered to SA
        url = "https://catalog.sarig.sa.gov.au/geonetwork/srv/api/records/geochemistry-soils/attachments/Geochemistry_Soils.csv"
        dest = self.raw_dir / "geochemistry_soils.csv"
        if dest.exists():
            log.info(f"Geochemistry cache hit: {dest}")
            return dest
        try:
            return _download_with_retry(url, dest, session=self.session)
        except Exception as e:
            log.error(f"Geochemistry download failed: {e}")
            return None

    def ingest_all(self) -> Dict[str, Any]:
        """Run full SARIG ingestion pipeline. Returns summary dict."""
        log.info("=== SARIG Ingestion Pipeline START ===")
        results: Dict[str, Any] = {}

        for key in self.SARIG_OPEN_DATA_PACKAGES:
            try:
                files = self.download_package(key)
                results[key] = {"status": "ok", "files": [str(f) for f in files]}
            except Exception as e:
                log.error(f"Package {key} failed: {e}")
                results[key] = {"status": "error", "error": str(e)}

        geochem = self.fetch_sarig_geochemistry()
        results["geochemistry"] = {"status": "ok" if geochem else "error", "file": str(geochem)}

        log.info("=== SARIG Ingestion Pipeline COMPLETE ===")
        return results


# ─────────────────────────────────────────────────────────────────
# Geoscience Australia Data Ingestion
# ─────────────────────────────────────────────────────────────────

class GAIngestionAgent:
    """
    Ingests open data from Geoscience Australia portals:
      - GADDS portal (gravity, magnetics, radiometrics, elevation)
      - OZMIN mineral occurrences
      - National surface geology
    """

    # Pre-known direct download endpoints for national geophysical compilations
    GA_GRID_DOWNLOADS = {
        "magnetics_tmi": {
            "description": "National Magnetic Compilation – Total Magnetic Intensity",
            "url": "https://d28rz98at9flks.cloudfront.net/74007/Geodata.zip",
            "format": "ermapper_grid",
        },
        "gravity_bouguer": {
            "description": "Australian National Gravity Compilation – Bouguer Anomaly",
            "url": "https://d28rz98at9flks.cloudfront.net/12390/Geodata.zip",
            "format": "ermapper_grid",
        },
        "radiometrics_dose": {
            "description": "Radiometric Map of Australia – Total Dose Rate",
            "url": "https://d28rz98at9flks.cloudfront.net/71725/Geodata.zip",
            "format": "ermapper_grid",
        },
        "srtm_elevation": {
            "description": "1 Second SRTM Digital Elevation Model",
            "url": "https://d28rz98at9flks.cloudfront.net/72759/Geodata.zip",
            "format": "geotiff",
        },
    }

    OZMIN_WFS_URL = "https://ogc.ga.gov.au/geoserver/ows"

    def __init__(self, bbox: BoundingBox = MOUNT_WOODS_BBOX):
        self.bbox = bbox
        self.session = _make_session()
        safe_name = bbox.name.lower().replace(" ", "_").replace("/", "_")
        self.raw_dir = RAW_DIR / safe_name / "ga"
        self.processed_dir = PROCESSED_DIR / safe_name / "ga"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_grid(self, key: str) -> List[Path]:
        """Download and extract a GA geophysical grid package."""
        grid_info = self.GA_GRID_DOWNLOADS.get(key)
        if not grid_info:
            raise ValueError(f"Unknown GA grid: {key}")
        dest_zip = self.raw_dir / f"{key}.zip"
        if dest_zip.exists():
            log.info(f"GA cache hit: {dest_zip}")
        else:
            _download_with_retry(grid_info["url"], dest_zip, session=self.session)
        extract_dir = self.raw_dir / key
        extract_dir.mkdir(exist_ok=True)
        return _extract_zip(dest_zip, extract_dir)

    def fetch_ozmin_occurrences(self) -> Optional[Path]:
        """Fetch OZMIN mineral occurrences via WFS filtered to bbox."""
        params = {
            "service": "WFS",
            "version": "1.1.0",
            "request": "GetFeature",
            "typeName": "mineral_occurrences:MineralOccurrences",
            "outputFormat": "application/json",
            "bbox": f"{self.bbox.min_lon},{self.bbox.min_lat},{self.bbox.max_lon},{self.bbox.max_lat},EPSG:4326",
            "maxFeatures": 10000,
        }
        out_path = self.processed_dir / "ozmin_occurrences.geojson"
        if out_path.exists():
            log.info(f"OZMIN cache hit: {out_path}")
            return out_path
        try:
            _download_with_retry(self.OZMIN_WFS_URL, out_path, session=self.session, params=params)
            return out_path
        except Exception as e:
            log.error(f"OZMIN WFS failed: {e}")
            return None

    def reproject_and_clip_grids(self) -> Dict[str, Path]:
        """Reproject and clip all downloaded grids to the target bbox."""
        clipped: Dict[str, Path] = {}
        for key in self.GA_GRID_DOWNLOADS:
            extract_dir = self.raw_dir / key
            if not extract_dir.exists():
                log.warning(f"Grid {key} not yet downloaded; skipping reproject")
                continue
            # Find raster files
            raster_files = (
                list(extract_dir.rglob("*.tif"))
                + list(extract_dir.rglob("*.TIF"))
                + list(extract_dir.rglob("*.ers"))  # ERMapper
                + list(extract_dir.rglob("*.img"))
            )
            if not raster_files:
                log.warning(f"No raster found in {extract_dir}")
                continue
            src_path = raster_files[0]
            reproj_path = self.processed_dir / f"{key}_reproj.tif"
            clip_path = self.processed_dir / f"{key}_clip.tif"
            try:
                reproject_raster(src_path, reproj_path)
                clip_raster_to_bbox(reproj_path, self.bbox, out_path=clip_path)
                clipped[key] = clip_path
            except Exception as e:
                log.error(f"Reproject/clip failed for {key}: {e}")
        return clipped

    def apply_antinoise_filtering(self, grid_paths: Dict[str, Path]) -> Dict[str, Path]:
        """Apply median + wavelet filtering to clipped geophysical grids."""
        import rasterio
        filtered: Dict[str, Path] = {}
        for key, path in grid_paths.items():
            out_path = self.processed_dir / f"{key}_filtered.tif"
            if out_path.exists():
                filtered[key] = out_path
                continue
            try:
                with rasterio.open(path) as src:
                    data = src.read(1).astype(np.float32)
                    nodata_val = src.nodata if src.nodata else RASTER_NODATA
                    data[data == nodata_val] = np.nan
                    meta = src.meta.copy()
                log.info(f"Anti-noise filtering: {key}")
                data = median_filter_grid(data)
                data = wavelet_denoise_grid(data)
                meta.update({"nodata": RASTER_NODATA, "dtype": "float32"})
                with rasterio.open(out_path, "w", **meta) as dst:
                    arr = np.where(np.isnan(data), RASTER_NODATA, data)
                    dst.write(arr.astype(np.float32), 1)
                filtered[key] = out_path
                log.info(f"  ✓ Filtered → {out_path}")
            except Exception as e:
                log.error(f"Filtering failed for {key}: {e}")
                filtered[key] = path  # fall back to unfiltered
        return filtered

    def ingest_all(self) -> Dict[str, Any]:
        """Run full GA ingestion pipeline. Returns summary dict."""
        log.info("=== GA Ingestion Pipeline START ===")
        results: Dict[str, Any] = {}

        for key in self.GA_GRID_DOWNLOADS:
            try:
                files = self.download_grid(key)
                results[key] = {"status": "ok", "files": [str(f) for f in files]}
            except Exception as e:
                log.error(f"Grid {key} failed: {e}")
                results[key] = {"status": "error", "error": str(e)}

        ozmin = self.fetch_ozmin_occurrences()
        results["ozmin"] = {"status": "ok" if ozmin else "error", "file": str(ozmin)}

        # Fetch national mineral potential (EFTF)
        mp = self.fetch_national_mineral_potential()
        results["mineral_potential"] = {"status": "ok" if mp else "error", "file": str(mp)}

        clipped = self.reproject_and_clip_grids()
        filtered = self.apply_antinoise_filtering(clipped)
        results["clipped_grids"] = {k: str(v) for k, v in clipped.items()}
        results["filtered_grids"] = {k: str(v) for k, v in filtered.items()}

        log.info("=== GA Ingestion Pipeline COMPLETE ===")
        return results

    def fetch_national_mineral_potential(self) -> Optional[Path]:
        """Fetch national IOCG/Cu-Au mineral potential layers from GA."""
        from config.settings import GA_MINERAL_POTENTIAL_URL
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": "EFTF_Mineral_Potential_Copper_Gold:IOCG_Mineral_Potential",
            "outputFormat": "application/json",
            "bbox": f"{self.bbox.min_lon},{self.bbox.min_lat},{self.bbox.max_lon},{self.bbox.max_lat}",
        }
        out_path = self.processed_dir / "ga_mineral_potential.geojson"
        if out_path.exists():
            return out_path
        try:
            return _download_with_retry(GA_MINERAL_POTENTIAL_URL, out_path, session=self.session, params=params)
        except Exception as e:
            log.error(f"National mineral potential fetch failed: {e}")
            return None


class StateSurveyIngestionAgent:
    """Ingests data from other Australian State Geological Surveys (WA, QLD, NSW)."""
    
    def __init__(self, bbox: BoundingBox):
        self.bbox = bbox
        self.session = _make_session()
        safe_name = bbox.name.lower().replace(" ", "_").replace("/", "_")
        self.raw_dir = RAW_DIR / safe_name / "state_surveys"
        self.processed_dir = PROCESSED_DIR / safe_name / "state_surveys"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def ingest_from_portal(self, state_code: str, layer_name: str) -> Optional[Path]:
        """Generic WFS fetch for state portals defined in settings."""
        from config.settings import STATE_PORTALS
        url = STATE_PORTALS.get(state_code)
        if not url:
            log.warning(f"No portal URL for state: {state_code}")
            return None
            
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": layer_name,
            "outputFormat": "application/json",
            "bbox": f"{self.bbox.min_lon},{self.bbox.min_lat},{self.bbox.max_lon},{self.bbox.max_lat}",
        }
        out_path = self.processed_dir / f"{state_code}_{layer_name.replace(':', '_')}.geojson"
        try:
            return _download_with_retry(url, out_path, session=self.session, params=params)
        except Exception as e:
            log.error(f"State fetch failed for {state_code} {layer_name}: {e}")
            return None


class TasmaniaIngestionAgent:
    """
    Ingests data directly from Mineral Resources Tasmania (MRT).
    Uses ArcGIS REST services and OGC WFS, completely bypassing rigid file downloads.
    """
    MRT_WFS_BASE = "https://data.stategrowth.tas.gov.au/ags/services/MRT/Mineral_Resources_Tasmania_OGC/MapServer/WFSServer"
    MRT_REST_BASE = "https://data.stategrowth.tas.gov.au/ags/rest/services/MRT"

    def __init__(self, bbox: BoundingBox):
        self.bbox = bbox
        self.processed_dir = PROCESSED_DIR / bbox.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.session = _make_session()

    def fetch_geology_wfs(self) -> Optional[Path]:
        log.info(f"Fetching MRT Geology for {self.bbox.name} via WFS")
        # Params for extracting mineral occurrences directly from MRT WFS.
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": "mrtwfs:MineralOccurrences",
            "outputFormat": "GEOJSON",
            "bbox": f"{self.bbox.min_lon},{self.bbox.min_lat},{self.bbox.max_lon},{self.bbox.max_lat}",
        }
        out = self.processed_dir / "tas_mineral_occurrences.geojson"
        try:
            return _download_with_retry(self.MRT_WFS_BASE, out, session=self.session, params=params)
        except Exception as e:
            log.error(f"MRT WFS error: {e}")
            return None

    def ingest_all(self) -> Dict[str, Any]:
        log.info("=== Tasmania (MRT) Ingestion Pipeline START ===")
        results = {}
        
        geo = self.fetch_geology_wfs()
        results["tas_geology"] = {"status": "ok" if geo else "error", "file": str(geo)}
        
        log.info("=== Tasmania (MRT) Ingestion Pipeline COMPLETE ===")
        return results



# ─────────────────────────────────────────────────────────────────
# LangChain Tool Wrappers
# ─────────────────────────────────────────────────────────────────

from langchain.tools import tool


@tool
def run_sarig_ingestion(bbox_name: str = "mount_woods") -> str:
    """
    Run the full SARIG data ingestion pipeline for the specified area.
    Downloads surface geology, mineral occurrences, drillholes, and geochemistry.
    Returns a JSON summary of ingested datasets.
    """
    agent = SARIGIngestionAgent(bbox=MOUNT_WOODS_BBOX)
    results = agent.ingest_all()
    return json.dumps(results, indent=2)


@tool
def run_tasmania_ingestion(bbox_name: str = "tasmania_mount_read") -> str:
    """
    Run the Tasmania (MRT) data ingestion pipeline via WFS/REST.
    """
    from config.settings import AUSTRALIAN_PROVINCES
    bbox = AUSTRALIAN_PROVINCES.get(bbox_name, AUSTRALIAN_PROVINCES.get("tasmania_mount_read", MOUNT_WOODS_BBOX))
    agent = TasmaniaIngestionAgent(bbox=bbox)
    results = agent.ingest_all()
    return json.dumps(results, indent=2)



@tool
def run_ga_ingestion(bbox_name: str = "mount_woods") -> str:
    """
    Run the full Geoscience Australia data ingestion pipeline.
    Downloads magnetics, gravity, radiometrics, elevation grids and OZMIN occurrences.
    Returns a JSON summary of ingested and processed datasets.
    """
    agent = GAIngestionAgent(bbox=MOUNT_WOODS_BBOX)
    results = agent.ingest_all()
    return json.dumps(results, indent=2)


@tool
def check_data_availability() -> str:
    """
    Check which datasets have already been downloaded and are available locally.
    Returns a formatted summary of available and missing datasets.
    """
    summary_lines = ["=== JR MineralForge – Data Availability Check ===\n"]

    sarig_dir = RAW_DIR / "sarig"
    ga_dir = RAW_DIR / "ga"
    processed_dir = PROCESSED_DIR

    for label, directory in [("SARIG Raw", sarig_dir), ("GA Raw", ga_dir), ("Processed", processed_dir)]:
        summary_lines.append(f"[{label}] ({directory})")
        if directory.exists():
            files = list(directory.rglob("*.*"))
            total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
            summary_lines.append(f"  {len(files)} files, {total_mb:.1f} MB total")
            for f in sorted(files)[:10]:
                summary_lines.append(f"  - {f.name}")
            if len(files) > 10:
                summary_lines.append(f"  ... and {len(files) - 10} more")
        else:
            summary_lines.append("  (not yet downloaded)")
        summary_lines.append("")

    return "\n".join(summary_lines)
