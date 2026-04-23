
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from agents.ml_engine_agent import ProspectivityMLEngine
from config.settings import ML_CONFIG

def initialize():
    print("Initializing JR MineralForge ML Pipeline...")
    
    # Create a synthetic dataset for bootstrap initialization
    # In a real scenario, this would load from a database or processed rasters
    n_samples = 100
    features = ["tmi", "gravity", "total_dose", "elevation"]
    
    data = {}
    for f in features:
        data[f] = np.random.randn(n_samples)
    
    # Add coordinates for spatial CV
    data["x"] = np.random.uniform(0, 10000, n_samples)
    data["y"] = np.random.uniform(0, 10000, n_samples)
    
    # Labels (0: non-mine, 1: mine)
    data["label"] = np.random.randint(0, 2, n_samples)
    
    df = pd.DataFrame(data)
    coord_df = df[["x", "y"]]
    
    engine = ProspectivityMLEngine(config=ML_CONFIG)
    
    print("Fitting model on initial dataset...")
    engine.fit(df, target_col="label", coord_df=coord_df)
    
    print("ML Engine Initialized and Models Saved.")

if __name__ == "__main__":
    initialize()
