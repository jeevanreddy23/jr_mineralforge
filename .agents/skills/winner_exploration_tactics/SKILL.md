# Winner Exploration Tactics - Mineral Prospectivity Mapping (MPM)
================================================================

This skill outlines the high-fidelity features and validation strategies used by winners of the **OZ Minerals Explorer Challenge** and other industry-standard prospectivity competitions.

## 🏆 Key Tactics for Drill Target Success

### 1. Geophysical Feature Engineering (The "Guru" Pattern)
Don't just use raw TMI (Magnetics) or Bouguer Gravity. Targets are often at **gradients** or **structural intersections**.

*   **1st Vertical Derivative (1VD)**: Sharpens shallow features and highlights structural edges. Calculated as the vertical gradient of the magnetic field.
*   **Analytical Signal (AS)**: The amplitude of the total gradient. Independent of magnetization direction; extremely useful for mapping the boundaries of large magnetic bodies.
*   **Residual Gravity**: Subtract the regional field to isolate local density anomalies (e.g., hematite-rich IOCG pipes).
*   **Distance-to-Conduits**: Pre-calculate Euclidian distance to mapped faults or "magnetic worms" (multi-scale edges).

### 2. Multi-Agent Ensemble (The "DeepSightX" Pattern)
Single models (like a plain Random Forest) overfit on spatial noise.
*   **Voting Ensemble**: Combine Balanced Random Forest (bias toward sparse deposits), XGBoost (high precision), and LightGBM (fast iteration).
*   **Calibration**: Use Platt Scaling or Isotonic Regression to ensure output probabilities represent "Drill Confidence".

### 3. Spatial Integrity (Anti-Leakage)
*   **Block Cross-Validation**: Divide the tenement into geographic blocks (e.g., 5km x 5km). Never allow a training pixel to be adjacent to a testing pixel.
*   **Buffer Zones**: Implement a 500m-2km exclusion buffer between training and testing folds to prevent "near-neighbor" leakage.

## 🛠️ Implementation Snippets

### 1st Vertical Derivative (1VD) Approximation
```python
import numpy as np
from scipy.ndimage import gaussian_filter

def compute_1vd(grid, resolution=50):
    """
    Computes 1st Vertical Derivative (1VD) approximation.
    """
    dy, dx = np.gradient(grid, resolution)
    return np.sqrt(dx**2 + dy**2)
```

### Analytical Signal (AS)
```python
def analytical_signal(grid):
    # Pseudo-AS using horizontal gradients
    dy, dx = np.gradient(grid)
    return np.sqrt(dx**2 + dy**2)
```
