# MineralForge

Edge-deployed, interpretable rock-burst risk detection for underground mines.

MineralForge is a field-ready prototype for the pipeline mining companies
actually care about:

```text
INMP441 + ADXL345 -> edge features -> XGBoost/RF model -> SHAP drivers -> TARP action
```

The point is not to stream raw sensor noise into a black box. The point is to
extract the right geotechnical features on a Raspberry Pi class device, score
risk locally, explain the cause, and map that prediction to an operational
Trigger Action Response Plan.

## Why this matters

Most mining AI demos stop at "we trained a model." MineralForge is structured as
a decision system:

| Layer | What it does |
| --- | --- |
| Edge acquisition | Reads acoustic emission and vibration windows from low-cost sensors |
| Feature extraction | Computes RMS, peak frequency, spectral entropy, event rate, PPV, dominant frequency, energy, and cumulative energy |
| ML risk engine | Uses XGBoost when available, with Random Forest fallback |
| Imbalance roadmap | Supports the project path from SMOTE to SVMSMOTE to GAN-style synthetic seismic event generation |
| Explainability | Produces SHAP-style top drivers for every assessment |
| TARP integration | Converts risk into a clear field action |

The core predictor is cumulative energy:

```text
E_cum = sum(E_i)
```

It is treated as a first-class feature because rapid energy accumulation is a
known precursor pattern in rock-burst and seismic hazard monitoring.

## Quickstart

```bash
python -m pip install -r requirements.txt
python main.py --zone "Stope 3" --stress 2.8
```

JSON output for integration:

```bash
python main.py --zone "Stope 3" --stress 2.8 --json
```

Model validation report on synthetic proxy data:

```bash
python main.py --train-report
```

Batch prediction from the included sample CSV:

```bash
python predict.py data/sample_input.csv
```

Run the lightweight report app:

```bash
python app.py
```

Run the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

Run the FastAPI inference service:

```bash
uvicorn api:app --reload
```

## Example output

```text
MineralForge Edge Assessment
================================
Zone: Stope 3
Risk: HIGH (82.4%)

Top drivers:
- cumulative_energy: 7981.219
- acoustic_event_rate: 35.500
- vibration_ppv: 2.724

TARP action:
Evacuate Stope 3 immediately and suspend work until geotechnical clearance.
Notify geotechnical engineer, shift supervisor, and control room.
Restrict all non-essential access and establish exclusion barricades.
```

## Repository layout

```text
mineralforge/
  features.py        Edge feature extraction
  models.py          XGBoost/RF training and SHAP-style explanation
  pipeline.py        Risk assessment orchestration
  tarp.py            Trigger Action Response Plan mapping
  synthetic.py       Synthetic microseismic proxy data
  edge_simulator.py  Raspberry Pi style demo loop
tests/
  test_features.py
  test_pipeline.py
```

## Hardware target

Recommended prototype hardware:

- Raspberry Pi 5
- INMP441 I2S MEMS microphone for acoustic emission proxies
- ADXL345 accelerometer for vibration/seismic proxies

The edge device should compute features locally and transmit only compact event
records. Raw streams can be logged during calibration, but they should not be
the production data path.

## Technical Background

Blast vibration risk depends on both source energy and propagation path. Two
industry-standard scaled-distance terms are included:

```text
SD_sqrt = distance / sqrt(charge mass)
SD_cuberoot = distance / charge mass^(1/3)
```

These features help normalize blasts of different sizes and distances. PPV is
estimated with a site-calibrated relation of the form `PPV = k * SD^b`, then
adjusted with a soil attenuation factor. Competent rock, fractured rock, sand,
clay, and fill can attenuate or amplify vibration differently, so `soil_type`
is encoded during preprocessing.

Frequency content matters because structures respond differently to low and high
frequency vibration. MineralForge includes FFT utilities for dominant frequency
and frequency-band energy so a field engineer can distinguish a high-amplitude
short-duration event from a lower-amplitude event that lands in a sensitive
structural frequency band.

## Professional Interfaces

- `dashboard.py` gives managers and field engineers a Streamlit review surface
  for CSV upload, risk summaries, timelines, and driver counts.
- `api.py` exposes `/predict` for Raspberry Pi or geophone gateways to post
  feature records for live inference.
- `mineralforge/training.py` includes class-weighted Random Forest tuning,
  optional SMOTE, and optional MLflow tracking.
- `data/sample_input.csv` lets users test the project immediately.

## Roadmap

1. Edge sensor drivers for INMP441 and ADXL345
2. SVMSMOTE training mode for tighter rare-event boundary learning
3. SHAP dashboard with driver trends over the last 30, 60, and 90 minutes
4. Multi-node spatial risk map across stopes and drives
5. GAN-based synthetic event generation for rare high-risk scenarios
