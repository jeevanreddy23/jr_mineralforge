import geopandas as gpd
import os
import logging
from shapely.geometry import box

logger = logging.getLogger("GeometryAnalyzer")

class GeometryAnalyzerAgent:
    """
    Team JR Swarm Agent: Geometry Analyzer.
    Parses KML/GeoJSON/SHP files and calculates optimized Bounding Boxes.
    Specifically designed to handle point-dense KMLs like 'mindep_commod_all_con.kml'.
    """
    def __init__(self, buffer_padding: float = 0.2):
        self.buffer_padding = buffer_padding

    def analyze(self, file_path: str, iteration_buffer: float = 0.0) -> dict:
        logger.info(f"Analyzing geometry: {file_path}")
        try:
            # Handle the file upload
            if not os.path.exists(file_path):
                return {"status": "error", "message": f"File not found: {file_path}"}

            gdf = gpd.read_file(file_path)
            if gdf.empty:
                return {"status": "error", "message": "Geometry file is empty."}

            # Ensure EPSG:4326 (WGS84)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)

            # Get Extent
            minx, miny, maxx, maxy = gdf.total_bounds
            
            # If it's a single point or very small, force a minimum span (approx 10km)
            if abs(maxx - minx) < 0.01:
                minx -= 0.05
                maxx += 0.05
            if abs(maxy - miny) < 0.01:
                miny -= 0.05
                maxy += 0.05

            # Apply buffer (base + recursive expansion)
            total_buffer = self.buffer_padding + iteration_buffer
            width = maxx - minx
            height = maxy - miny
            
            bbox = {
                "min_lon": minx - (width * total_buffer),
                "max_lon": maxx + (width * total_buffer),
                "min_lat": miny - (height * total_buffer),
                "max_lat": maxy + (height * total_buffer)
            }

            return {
                "status": "success",
                "bbox": bbox,
                "feature_count": len(gdf),
                "type": "vector_layer",
                "resonance_weight": 0.9
            }
        except Exception as e:
            logger.error(f"Geometry analysis failed: {str(e)}")
            return {"status": "error", "message": str(e)}
