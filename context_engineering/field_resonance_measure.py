import numpy as np
import xarray as xr
from typing import Dict, Any, List

class FieldResonanceMeasure:
    """
    Team JR v2.1: Measures the alignment and quality of ingested geospatial data.
    Uses 'Resonance' as a metric for decision-making in the Swarm.
    """
    
    @staticmethod
    def calculate_resonance(datacube: xr.Dataset, context: Dict[str, Any]) -> float:
        """
        Calculates resonance (0.0 - 1.0).
        Criteria:
        - Layer presence (Critical: Mag, Grav)
        - Data density (No-NaN ratio)
        - Spatial alignment with target geometry
        - Recursive decay
        """
        if datacube is None or len(datacube.data_vars) == 0:
            return 0.0
            
        # 1. Coverage Score
        coverage = []
        for var in datacube.data_vars:
            filled = float(datacube[var].notnull().mean())
            coverage.append(filled)
        avg_coverage = np.mean(coverage) if coverage else 0.0
        
        # 2. Priority Check
        has_essential = all(layer in datacube.data_vars for layer in ['magnetics', 'gravity'])
        priority_mult = 1.0 if has_essential else 0.6
        
        # 3. Recursive Decay (Ant-noise logic)
        iteration = context.get('iteration', 0)
        decay = context.get('decay_rate', 0.05)
        
        resonance = (avg_coverage * priority_mult) - (iteration * decay)
        return float(np.clip(resonance, 0.0, 1.0))

    @staticmethod
    def get_status_label(score: float) -> str:
        if score >= 0.85: return "HIGH RESONANCE (Production Ready)"
        if score >= 0.70: return "STABLE (Slight Noise/Missingness)"
        if score >= 0.40: return "LOW (Synthetic Required)"
        return "DISSONANCE (Retry Loop Triggered)"
