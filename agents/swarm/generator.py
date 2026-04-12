try:
    import gstools as gs
except ImportError:
    gs = None
    
import numpy as np
import xarray as xr
import logging

logger = logging.getLogger("FallbackGenerator")

class FallbackGeneratorAgent:
    """
    Team JR Swarm Agent: Fallback / Simulation.
    Uses geostatistical Gaussian Process (via gstools) to generate meaningful
    synthetic geophysics if live data fails to resonate.
    """
    
    def generate(self, bbox: dict, resolution: float = 0.01) -> xr.Dataset:
        logger.info(f"Generating synthetic field for {bbox}")
        
        lons = np.arange(bbox['min_lon'], bbox['max_lon'], resolution)
        lats = np.arange(bbox['min_lat'], bbox['max_lat'], resolution)
        
        # Ensure we have a grid
        if len(lons) < 2: lons = np.array([bbox['min_lon'], bbox['max_lon']])
        if len(lats) < 2: lats = np.array([bbox['min_lat'], bbox['max_lat']])

        # Use gstools for a structured Gaussian field (if available)
        if gs:
            model = gs.Exponential(dim=2, var=100.0, len_scale=0.1)
            srf = gs.SRF(model, seed=42)
            # Call srf directly with coordinates for structured grid
            # Note: gstools expects (x, y) order corresponding to the model dim
            grid_z = srf((lons, lats), mesh_type='structured')
            
            # Normalize to realistic magnetic nT values
            mag_data = grid_z.T * 50 + 50000 
            grav_data = grid_z.T * 5 + 10
        else:
            # Fallback to simple random field if gstools is missing
            grid_z = np.random.normal(0, 1, size=(len(lats), len(lons)))
            mag_data = grid_z * 50 + 50000
            grav_data = grid_z * 5 + 10

        ds = xr.Dataset(
            data_vars={
                "magnetics": (["lat", "lon"], mag_data),
                "gravity": (["lat", "lon"], grav_data)
            },
            coords={
                "lat": lats,
                "lon": lons
            },
            attrs={"source": "Synthetic_Geostat_v2.1", "is_synthetic": True}
        )
        
        return ds
