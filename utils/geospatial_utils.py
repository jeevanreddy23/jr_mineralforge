"""
JR MineralForge – Resilient Geospatial Utilities
================================================
Helper functions for CRS reprojection, raster I/O, and spatial transforms.
Modified to be resilient to missing drivers (rasterio, geopandas) on Windows.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from config.settings import (
    TARGET_CRS, WGS84_CRS, TARGET_RESOLUTION_M, RASTER_NODATA,
    MOUNT_WOODS_BBOX, BoundingBox
)
from utils.logging_utils import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────
# Resilient Imports
# ─────────────────────────────────────────────────────────────────

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask as rio_mask
    HAS_RASTERIO = True
except ImportError:
    log.warning("rasterio not found – spatial raster processing will be disabled.")
    HAS_RASTERIO = False

try:
    import geopandas as gpd
    from shapely.geometry import box, mapping
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    log.warning("geopandas/shapely not found – vector processing and bbox transforms will be disabled.")
    HAS_GEOPANDAS = False

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    log.warning("xarray not found – datacube stacking will be disabled.")
    HAS_XARRAY = False


# ─────────────────────────────────────────────────────────────────
# Bounding Box Helpers
# ─────────────────────────────────────────────────────────────────

def bbox_to_geodataframe(bbox: BoundingBox) -> Any:
    """Convert a BoundingBox to a GeoDataFrame in WGS84."""
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for bbox_to_geodataframe")
    from shapely.geometry import box
    geom = box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
    return gpd.GeoDataFrame({"geometry": [geom]}, crs=WGS84_CRS)


def bbox_to_projected(bbox: BoundingBox, target_crs: str = TARGET_CRS) -> Any:
    """Return a GeoDataFrame of the bbox reprojected to target_crs."""
    gdf = bbox_to_geodataframe(bbox)
    return gdf.to_crs(target_crs)


def wgs84_bbox_to_projected_bounds(bbox: BoundingBox, target_crs: str = TARGET_CRS) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) in projected CRS."""
    if not HAS_GEOPANDAS:
         # Fallback approximate math if needed, but safer to error
        raise ImportError("geopandas is required for spatial bounds projection")
    gdf = bbox_to_projected(bbox, target_crs)
    return tuple(gdf.total_bounds)  # type: ignore


# ─────────────────────────────────────────────────────────────────
# Raster I/O
# ─────────────────────────────────────────────────────────────────

def reproject_raster(
    src_path: Path,
    dst_path: Path,
    target_crs: str = TARGET_CRS,
    resolution: float = TARGET_RESOLUTION_M,
    resampling: Any = None,
) -> Path:
    """Reproject and resample a raster to target_crs and resolution."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for reproject_raster")
    
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    if resampling is None:
        resampling = Resampling.bilinear

    log.info(f"Reprojecting {src_path.name} -> {target_crs} @ {resolution}m")
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds,
            resolution=(resolution, resolution)
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": RASTER_NODATA,
            "driver": "GTiff",
        })
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )
    log.info(f"  -> saved to {dst_path}")
    return dst_path


def clip_raster_to_bbox(
    raster_path: Path,
    bbox: BoundingBox,
    out_path: Optional[Path] = None,
) -> Tuple[np.ndarray, dict]:
    """Clip a raster to bbox, return (array, meta). Optionally write to disk."""
    if not HAS_RASTERIO or not HAS_GEOPANDAS:
        raise ImportError("rasterio and geopandas are required for clip_raster_to_bbox")
    
    from shapely.geometry import box, mapping
    geom = [mapping(box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat))]
    with rasterio.open(raster_path) as src:
        bbox_gdf = bbox_to_geodataframe(bbox).to_crs(src.crs.to_string())
        clip_geom = [mapping(bbox_gdf.geometry.iloc[0])]
        out_image, out_transform = rio_mask(src, clip_geom, crop=True, nodata=RASTER_NODATA)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": RASTER_NODATA,
        })
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)
        log.info(f"Clipped raster saved -> {out_path}")
    return out_image, out_meta


def raster_to_xarray(raster_path: Path, name: str) -> Any:
    """Load a single-band raster as an xarray DataArray with x/y coords."""
    if not HAS_RASTERIO or not HAS_XARRAY:
        raise ImportError("rasterio and xarray are required for raster_to_xarray")
    
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else RASTER_NODATA
        data = np.where(data == nodata, np.nan, data)
        height, width = data.shape
        # Use 1D coordinates for standard grid alignment
        t = src.transform
        xs = np.array([t[2] + (i + 0.5) * t[0] for i in range(width)])
        ys = np.array([t[5] + (i + 0.5) * t[4] for i in range(height)])
        
        da = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"y": ys, "x": xs},
            name=name,
            attrs={"crs": str(src.crs), "transform": list(src.transform)},
        )
    return da


def stack_rasters_to_datacube(
    raster_paths: Dict[str, Path],
    bbox: BoundingBox,
    target_crs: str = TARGET_CRS,
    resolution: float = TARGET_RESOLUTION_M,
    tmp_dir: Optional[Path] = None,
) -> Any:
    """Reproject, clip, and stack multiple rasters into a single xr.Dataset."""
    if not HAS_XARRAY:
        raise ImportError("xarray is required for stacking")
        
    arrays: Dict[str, xr.DataArray] = {}
    if tmp_dir is None:
        from config.settings import PROCESSED_DIR
        tmp_dir = PROCESSED_DIR / "reprojected"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for name, path in raster_paths.items():
        try:
            log.info(f"Processing layer: {name}")
            reproj_path = tmp_dir / f"{name}_reproj.tif"
            reproject_raster(path, reproj_path, target_crs=target_crs, resolution=resolution)
            clip_path = tmp_dir / f"{name}_clip.tif"
            clip_raster_to_bbox(reproj_path, bbox, out_path=clip_path)
            arrays[name] = raster_to_xarray(clip_path, name)
        except Exception as e:
            log.error(f"Failed to process layer {name}: {e}")

    if not arrays:
        return xr.Dataset()
        
    names = list(arrays.keys())
    template = arrays[names[0]]
    aligned = {names[0]: template}
    for n in names[1:]:
        try:
            aligned[n], _ = xr.align(template, arrays[n], join="left", fill_value=np.nan)
        except Exception:
            aligned[n] = arrays[n]

    return xr.Dataset(aligned)


# ─────────────────────────────────────────────────────────────────
# Geophysics Filtering (Anti-Noise)
# ─────────────────────────────────────────────────────────────────

def median_filter_grid(array: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply 2D median filter to geophysics grid for noise suppression."""
    try:
        from scipy.ndimage import generic_filter
    except ImportError:
        log.warning("scipy not found – skipping median filter")
        return array
        
    nan_mask = np.isnan(array)
    filled = np.where(nan_mask, 0.0, array)
    filtered = generic_filter(filled, np.nanmedian, size=window)
    filtered[nan_mask] = np.nan
    return filtered.astype(np.float32)


def wavelet_denoise_grid(array: np.ndarray, level: int = 3, wavelet: str = "db4") -> np.ndarray:
    """Wavelet soft-threshold denoising for 2D geophysical grids."""
    try:
        import pywt
    except ImportError:
        log.warning("PyWavelets not installed; skipping wavelet denoising")
        return array

    nan_mask = np.isnan(array)
    filled = np.where(nan_mask, np.nanmean(array) if not np.all(nan_mask) else 0.0, array)
    coeffs = pywt.wavedec2(filled, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(filled.size))
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(tuple(pywt.threshold(sub, threshold, mode="soft") for sub in c))
    denoised = pywt.waverec2(new_coeffs, wavelet)
    denoised = denoised[: array.shape[0], : array.shape[1]]
    denoised[nan_mask] = np.nan
    return denoised.astype(np.float32)
