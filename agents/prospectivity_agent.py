"""
JR MineralForge – Prospectivity Mapping & Target Generation Agent
==================================================================
Generates the final prospectivity map from the ML engine outputs,
extracts and ranks drill targets, and exports GIS products.

Outputs:
  - Prospectivity raster (GeoTIFF)
  - Uncertainty map (GeoTIFF)
  - Ranked target points (GeoPackage + CSV)
  - Interactive Folium map (HTML)
  - KML export
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Resilient geospatial imports
try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RIO = True
except ImportError:
    HAS_RIO = False

try:
    import xarray as xr
    HAS_XR = True
except ImportError:
    HAS_XR = False

import folium
from folium.plugins import MarkerCluster

from config.settings import (
    PROCESSED_DIR, REPORTS_DIR, MOUNT_WOODS_BBOX, BoundingBox,
    TARGET_CRS, WGS84_CRS, TARGET_RESOLUTION_M, RASTER_NODATA,
    BRAND_NAME, TEAM_NAME, BRAND_HEADER, BRAND_FOOTER,
    ML_CONFIG,
)
from utils.logging_utils import get_logger
from utils.geospatial_utils import stack_rasters_to_datacube, bbox_to_projected

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Datacube Assembly
# ─────────────────────────────────────────────────────────────────

def assemble_feature_datacube(
    bbox: BoundingBox = MOUNT_WOODS_BBOX,
) -> Optional[xr.Dataset]:
    """
    Load all available processed raster layers into an xr.Dataset.
    Layers: magnetics_tmi, gravity_bouguer, radiometrics (k, eth, eu),
            elevation, and any derived products.
    Returns None if no layers found.
    """
    safe_name = bbox.name.lower().replace(" ", "_").replace("/", "_")
    processed_ga = PROCESSED_DIR / safe_name / "ga"
    raster_map: Dict[str, Path] = {}

    layer_names = {
        "magnetics_tmi_filtered": "tmi",
        "gravity_bouguer_filtered": "gravity",
        "radiometrics_dose_filtered": "total_dose",
        "srtm_elevation_filtered": "elevation",
        "magnetics_tmi_clip": "tmi_raw",
        "gravity_bouguer_clip": "gravity_raw",
    }

    for filename, varname in layer_names.items():
        path = processed_ga / f"{filename}.tif"
        if path.exists():
            raster_map[varname] = path
            log.info(f"Found layer: {varname} → {path.name}")

    if not raster_map:
        log.warning("No processed raster layers found. Run ingestion first.")
        return None

    return stack_rasters_to_datacube(raster_map, bbox)


# ─────────────────────────────────────────────────────────────────
# Prediction Grid Generation
# ─────────────────────────────────────────────────────────────────

def generate_prediction_grid(
    bbox: BoundingBox = MOUNT_WOODS_BBOX,
    resolution_m: float = TARGET_RESOLUTION_M,
) -> pd.DataFrame:
    """
    Create a regular grid of prediction points covering the bbox.
    Returns a DataFrame with x, y (projected) and lat/lon columns.
    """
    from pyproj import Transformer
    import geopandas as gpd

    bbox_gdf = bbox_to_projected(bbox, TARGET_CRS)
    minx, miny, maxx, maxy = bbox_gdf.total_bounds

    xs = np.arange(minx, maxx, resolution_m)
    ys = np.arange(miny, maxy, resolution_m)
    xx, yy = np.meshgrid(xs, ys)
    grid_x = xx.ravel()
    grid_y = yy.ravel()

    transformer = Transformer.from_crs(TARGET_CRS, WGS84_CRS, always_xy=True)
    lons, lats = transformer.transform(grid_x, grid_y)

    return pd.DataFrame({"x": grid_x, "y": grid_y, "lon": lons, "lat": lats})


def extract_features_at_grid(
    grid_df: pd.DataFrame,
    dataset: xr.Dataset,
) -> pd.DataFrame:
    """
    Sample xr.Dataset values at each grid point using nearest-neighbour.
    Adds dataset variables as feature columns to grid_df.
    """
    grid_out = grid_df.copy()
    for var in dataset.data_vars:
        da = dataset[var]
        vals = []
        for _, row in grid_df.iterrows():
            try:
                v = float(da.sel({"x": row["x"], "y": row["y"]}, method="nearest").values)
            except Exception:
                v = np.nan
            vals.append(v)
        grid_out[var] = vals
    return grid_out


# ─────────────────────────────────────────────────────────────────
# Prospectivity Mapping Agent
# ─────────────────────────────────────────────────────────────────

class ProspectivityMappingAgent:
    """
    Generates prospectivity maps and ranked drill targets.
    Integrates with ProspectivityMLEngine for predictions.
    """

    def __init__(
        self,
        bbox: BoundingBox = MOUNT_WOODS_BBOX,
        output_dir: Optional[Path] = None,
    ):
        self.bbox = bbox
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_full_pipeline(self, ml_engine=None) -> Dict[str, Any]:
        """
        End-to-end prospectivity pipeline.
        1. Assemble feature datacube
        2. Generate prediction grid
        3. Extract features
        4. ML prediction (proba + uncertainty)
        5. Generate rasters
        6. Extract ranked targets
        7. Create maps and exports
        """
        results: Dict[str, Any] = {}
        log.info("=== Prospectivity Mapping Pipeline START ===")

        # Step 1: Datacube
        ds = assemble_feature_datacube(self.bbox)
        if ds is None or len(ds.data_vars) == 0:
            log.warning(f"No datacube available for {self.bbox.name}. Expansion required.")
            return {
                "error": f"No geophysical data found for the selected region ({self.bbox.name}).",
                "status": "failed",
                "suggestion": "Expand the boundary polygon or ensure you are targeting a valid Australian mineral province."
            }

        # Step 2: Prediction grid
        log.info("Generating prediction grid …")
        grid_df = generate_prediction_grid(self.bbox)
        log.info(f"  Grid: {len(grid_df):,} cells")

        # Step 3: Extract features
        log.info("Extracting features at grid points …")
        feat_df = extract_features_at_grid(grid_df, ds)

        # Step 4: Predictions
        if ml_engine is None:
            try:
                from agents.ml_engine_agent import ProspectivityMLEngine
                ml_engine = ProspectivityMLEngine.load()
            except Exception as e:
                log.error(f"Failed to load ML models: {e}. No predictions can be made.")
                raise RuntimeError("ML model engine is not initialized. Cannot perform 'Drill or Drop' analysis. Run training first.")

        log.info("Running ML predictions …")
        proba, uncertainty = ml_engine.predict_with_uncertainty(feat_df)
        grid_df["prospectivity"] = proba
        grid_df["uncertainty"] = uncertainty
        
        # Calculate SHAP explainability per pixel/target
        try:
            shap_vals = ml_engine.compute_shap_values(feat_df)
            if shap_vals is not None:
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                feature_names = ml_engine.feature_cols
                top_indices = np.argmax(np.abs(shap_vals), axis=1)
                grid_df["primary_driver"] = [feature_names[i] for i in top_indices]
            else:
                grid_df["primary_driver"] = "Unknown"
        except Exception as e:
            log.warning(f"SHAP explanation failed: {e}")
            grid_df["primary_driver"] = "Unknown"

        # Step 5: Rasterize
        prosp_raster = self._grid_to_raster(grid_df, "prospectivity", "prospectivity_map.tif")
        uncert_raster = self._grid_to_raster(grid_df, "uncertainty", "uncertainty_map.tif")
        results["prospectivity_raster"] = str(prosp_raster)
        results["uncertainty_raster"] = str(uncert_raster)

        # Step 6: Extract ranked targets
        targets_gdf = self._extract_targets(grid_df)
        targets_csv = self.output_dir / "ranked_targets.csv"
        targets_gpkg = self.output_dir / "ranked_targets.gpkg"
        targets_kml = self.output_dir / "ranked_targets.kml"
        targets_gdf.to_csv(targets_csv, index=False)
        targets_gdf.to_file(targets_gpkg, driver="GPKG")
        try:
            targets_gdf.to_crs(WGS84_CRS).to_file(targets_kml, driver="KML")
        except Exception as e:
            log.warning(f"KML export failed: {e}")
        results["targets_csv"] = str(targets_csv)
        results["targets_gpkg"] = str(targets_gpkg)
        results["n_targets"] = len(targets_gdf)

        # Step 7: Folium interactive map
        map_path = self._create_folium_map(grid_df, targets_gdf)
        results["interactive_map"] = str(map_path)

        log.info("=== Prospectivity Mapping Pipeline COMPLETE ===")
        log.info(f"  Top targets: {len(targets_gdf)}")
        return results

    def _grid_to_raster(
        self, grid_df: pd.DataFrame, value_col: str, filename: str
    ) -> Path:
        """Convert a flat grid DataFrame column to a GeoTIFF raster."""
        from pyproj import Transformer
        gdf = gpd.GeoDataFrame(
            grid_df[[value_col, "x", "y"]],
            geometry=gpd.points_from_xy(grid_df["x"], grid_df["y"]),
            crs=TARGET_CRS,
        )
        bounds = gdf.total_bounds
        resolution = TARGET_RESOLUTION_M
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)
        transform = from_bounds(*bounds, width=width, height=height)

        raster = np.full((height, width), RASTER_NODATA, dtype=np.float32)
        for _, row in gdf.iterrows():
            col_idx = int((row.geometry.x - bounds[0]) / resolution)
            row_idx = height - 1 - int((row.geometry.y - bounds[1]) / resolution)
            if 0 <= row_idx < height and 0 <= col_idx < width:
                val = row[value_col]
                raster[row_idx, col_idx] = RASTER_NODATA if np.isnan(val) else val

        out_path = self.output_dir / filename
        with rasterio.open(
            out_path, "w",
            driver="GTiff", height=height, width=width,
            count=1, dtype="float32", crs=TARGET_CRS,
            transform=transform, nodata=RASTER_NODATA,
        ) as dst:
            dst.write(raster, 1)
        log.info(f"Raster saved → {out_path}")
        return out_path

    def _extract_targets(
        self, grid_df: pd.DataFrame, top_n: int = 50, threshold: float = 0.6
    ) -> gpd.GeoDataFrame:
        """Extract and rank top-N high-prospectivity locations as drill targets."""
        high = grid_df[grid_df["prospectivity"] >= threshold].copy()
        if len(high) == 0:
            high = grid_df.nlargest(top_n, "prospectivity")
        else:
            high = high.nlargest(top_n, "prospectivity")

        high = high.reset_index(drop=True)
        high["rank"] = high.index + 1
        high["confidence_score"] = (high["prospectivity"] * 100).round(1)
        high["confidence_category"] = pd.cut(
            high["prospectivity"],
            bins=[-0.1, 0.5, 0.7, 0.85, 1.1],
            labels=["Low", "Medium", "High", "Very High"],
        )
        high["target_label"] = [f"JR-T{i+1:03d}" for i in range(len(high))]
        
        # Ensure we have SHAP explanation if missing
        if "primary_driver" not in high.columns:
            high["primary_driver"] = "Unknown Target Source"

        geometry = [Point(row["lon"], row["lat"]) for _, row in high.iterrows()]
        gdf = gpd.GeoDataFrame(high, geometry=geometry, crs=WGS84_CRS)
        log.info(f"Extracted {len(gdf)} ranked targets (top prospectivity)")
        return gdf

    def _create_folium_map(
        self,
        grid_df: pd.DataFrame,
        targets_gdf: gpd.GeoDataFrame,
    ) -> Path:
        """Create an interactive folium map showing targets and prospectivity."""
        centre_lat, centre_lon = self.bbox.centre

        m = folium.Map(
            location=[centre_lat, centre_lon],
            zoom_start=9,
            tiles="CartoDB dark_matter",
        )

        # Title control
        title_html = f"""
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                    z-index: 1000; background: rgba(0,0,0,0.8); color: #FFD700;
                    padding: 10px 20px; border-radius: 8px; font-family: Arial;
                    font-size: 14px; font-weight: bold; border: 1px solid #FFD700;">
            🏆 {BRAND_NAME} – {self.bbox.name}<br>
            <span style="color: #ccc; font-size: 11px;">{BRAND_HEADER}</span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        # Add target markers
        cluster = MarkerCluster(name="Drill Targets").add_to(m)
        for _, row in targets_gdf.iterrows():
            cat = str(row.get("confidence_category", "Unknown"))
            colour = {"Very High": "red", "High": "orange", "Medium": "yellow", "Low": "gray"}.get(cat, "blue")
            popup_html = f"""
            <b>{row['target_label']}</b><br>
            Rank: {row['rank']}<br>
            Prospectivity Score: <b>{row['confidence_score']:.1f}%</b><br>
            Category: <b>{cat}</b><br>
            Primary ML Driver (SHAP): <b>{row.get('primary_driver', 'Unknown').upper()}</b><br>
            Uncertainty: {row.get('uncertainty', 0):.3f}<br>
            Lat/Lon: {row['lat']:.4f}, {row['lon']:.4f}<br>
            <hr><i>{BRAND_FOOTER}</i>
            """
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=colour, icon="star" if cat == "Very High" else "circle", prefix="fa"),
                tooltip=f"{row['target_label']} – {row['confidence_score']:.0f}%",
            ).add_to(cluster)

        # Layer control
        folium.LayerControl().add_to(m)

        # Footer
        footer_html = f"""
        <div style="position: fixed; bottom: 20px; right: 10px;
                    z-index: 1000; background: rgba(0,0,0,0.7); color: #ccc;
                    padding: 5px 10px; border-radius: 5px; font-size: 10px;">
            {BRAND_FOOTER}
        </div>
        """
        m.get_root().html.add_child(folium.Element(footer_html))

        out_path = self.output_dir / "jr_prospectivity_map.html"
        m.save(str(out_path))
        log.info(f"Interactive map saved → {out_path}")
        return out_path
