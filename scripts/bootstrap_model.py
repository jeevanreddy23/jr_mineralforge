import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

# Add root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from agents.ml_engine_agent import ProspectivityMLEngine
from config.settings import MODELS_DIR

def bootstrap():
    print("Bootstrap: Initializing JR MineralForge ML Engine...")
    
    # Create fake training data
    n_samples = 100
    features = ["magnetics_tmi", "gravity_bouguer", "radiometrics_k", "elevation"]
    
    X = pd.DataFrame(
        np.random.randn(n_samples, len(features)),
        columns=features
    )
    # Target: 1 if high mag & high gravity (proxy for IOCG)
    y = (X["magnetics_tmi"] > 0.5) & (X["gravity_bouguer"] > 0.5)
    X["label"] = y.astype(int)
    
    engine = ProspectivityMLEngine()
    print(f"Training base model on {n_samples} synthetic samples...")
    engine.fit(X, target_col="label")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    engine.save()
    print("Success: ml_engine.pkl has been created and saved.")

if __name__ == "__main__":
    bootstrap()
