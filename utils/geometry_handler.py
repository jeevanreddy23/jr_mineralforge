"""
JR MineralForge – Self-Correction Geometry Handler
==================================================
Validates user-uploaded geospatial files and auto-corrects point/line inputs 
into area-based polygons via intelligent buffering.
"""

import logging
from typing import Tuple, Optional
from pathlib import Path

from config.settings import BoundingBox, WGS84_CRS
from utils.logging_utils import get_logger

log = get_logger(__name__)

# Resilient Imports
try:
    import geopandas as gpd
    from shapely.geometry import box, Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    log.warning("geopandas/shapely not found. Geometry auto-correction will be disabled.")

def validate_and_fix_geometry(file_path: str, buffer_km: float = 20.0) -> Optional[BoundingBox]:
    """
    Detects geometry type of a geospatial file. 
    If Points/Lines are detected, auto-buffers into a Polygon area.
    Returns: BoundingBox object.
    """
    if not HAS_SPATIAL:
        log.error("Safe Geometry Logic failed: geopandas/shapely missing.")
        return None

    try:
        log.info(f"Validating geometry for {file_path}")
        gdf = gpd.read_file(file_path)
        
        if gdf.empty:
            log.warning("Empty geospatial file uploaded.")
            return None

        # Ensure WGS84
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            log.info("Reprojecting to WGS84...")
            gdf = gdf.to_crs(epsg=4326)

        # Step 2: Complexity Guard (Mega-KML Handler)
        # If the file has too many features, focus on the centroid of the most dense region
        if len(gdf) > 50:
            log.info(f"Large dataset detected ({len(gdf)} features). Focusing analysis on geographic centroid.")
            centroid = gdf.geometry.unary_union.centroid
            gdf = gpd.GeoDataFrame({"geometry": [centroid]}, crs=gdf.crs)
        
        # Step 3: Ensure geometry has sufficient area for geophysical context
        # Winners' Tactic: Minimum 10km x 10km exploration area for regional context
        bounds = gdf.total_bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # If the bounding box is smaller than 2km (point or small line), apply a 10km buffer
        if width < 2000 or height < 2000:
            log.info("Geometry too small for regional analysis; applying 10km exploration buffer.")
            # Reproject to meters for accurate buffering
            gdf_m = gdf.to_crs(epsg=3857) # Web Mercator for planar distance
            gdf_m["geometry"] = gdf_m.geometry.buffer(10000) # 10km buffer
            gdf = gdf_m.to_crs(epsg=4326) # Return to WGS84

        # Check if geometry is purely Point/Line
        all_points = all(isinstance(g, (Point, MultiPoint)) for g in gdf.geometry)
        all_lines = all(isinstance(g, (LineString, MultiLineString)) for g in gdf.geometry)
        
        if all_points or all_lines:
            log.info(f"Point/Line geometry detected. Applying {buffer_km}km buffer to stabilize pipeline.")
            
            # Project to a metric CRS for safe buffering (GDA2020 / MGA zone 53 is decent for SA)
            # Or use dynamic projection based on centroid
            centroid = gdf.centroid.iloc[0]
            utm_crs = gdf.estimate_utm_crs()
            gdf_m = gdf.to_crs(utm_crs)
            
            # Buffer by kilometers
            gdf_buffered = gdf_m.buffer(buffer_km * 1000)
            gdf = gdf_buffered.to_crs(epsg=4326)
            log.info("Successfully converted Points/Lines to Area Polygon.")

        minx, miny, maxx, maxy = gdf.total_bounds
        
        # Anti-Zero Area Check
        if minx == maxx or miny == maxy:
            log.warning("Zero-area detected after buffering. Forcing 5km default.")
            minx -= 0.05
            maxx += 0.05
            miny -= 0.05
            maxy += 0.05

        return BoundingBox(
            name=f"Fixed_{Path(file_path).stem}",
            min_lon=minx, max_lon=maxx,
            min_lat=miny, max_lat=maxy,
            description="Auto-corrected exploration area"
        )

    except Exception as e:
        log.error(f"Geometry validation failed: {e}")
        return None
