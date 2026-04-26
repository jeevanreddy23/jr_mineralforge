# Ground Vibration Seismic Detection Pipeline

## 1. Project Overview

1.1. This project trains a machine learning classifier for the included ground vibration dataset:

```text
data/ground_vibration_dataset.csv
```

1.2. The pipeline predicts `Vibration_Level` (`Low`, `Medium`, `High`) from blast, ground, and seismic sensor features.

1.3. The system also adds geotechnical engineering features, model artifacts, prediction tooling, visualizations, a Streamlit dashboard, and a FastAPI inference endpoint.

## 2. Repository Contents

2.1. `train_pipeline.py` trains and saves the best model.

2.2. `predict.py` loads the saved model and writes predictions for a CSV file.

2.3. `visualize_dataset.py` creates exploratory charts for the dataset.

2.4. `dashboard.py` provides a Streamlit field-review dashboard.

2.5. `api.py` provides a FastAPI endpoint for feature-based edge inference.

2.6. `data/ground_vibration_dataset.csv` contains the main vibration seismic detection dataset.

2.7. `data/sample_input.csv` contains a small example input file for quick testing.

2.8. `requirements.txt` lists the Python packages required to run the project.

2.9. `artifacts/` contains model outputs after training:

1. `vibration_detection_pipeline.joblib`: The trained reusable model.
2. `metrics.json`: Accuracy, macro F1, classification report, and confusion matrix.
3. `feature_importance.csv`: The strongest predictors when the selected model supports feature importance.
4. `predictions.csv`: The latest generated prediction output.

## 3. Train The Model

3.1. Run the default training pipeline:

```powershell
python train_pipeline.py
```

3.2. Run training with custom input and output paths:

```powershell
python train_pipeline.py --data data\ground_vibration_dataset.csv --output-dir artifacts
```

## 4. Generate Predictions

4.1. Score the included dataset and write predictions to the artifacts folder:

```powershell
python predict.py --input data\ground_vibration_dataset.csv --output artifacts\predictions.csv
```

4.2. Score the sample input file:

```powershell
python predict.py --input data\sample_input.csv --output artifacts\sample_predictions.csv
```

## 5. Launch The Dashboard

5.1. Start the Streamlit dashboard:

```powershell
streamlit run dashboard.py
```

5.2. The dashboard loads the trained pipeline, scores uploaded CSV files, shows PPV timelines, plots PPV against frequency, and displays feature importance from `artifacts/feature_importance.csv`.

## 6. Run The API

6.1. Start the FastAPI service:

```powershell
uvicorn api:app --reload
```

6.2. The API exposes a lightweight `/predict` endpoint for edge devices that already compute feature windows from acoustic and vibration sensors.

## 7. Visualize The Dataset

7.1. Generate exploratory visualizations:

```powershell
python visualize_dataset.py
```

7.2. The generated charts are written to the `visualizations/` folder.

## 8. Pipeline Workflow

8.1. The pipeline loads the CSV dataset.

8.2. The pipeline converts `Timestamp` into `hour`, `dayofweek`, and `month`.

8.3. The pipeline drops `Blast_ID` because it is an identifier, not a reusable physical predictor.

8.4. The pipeline adds geotechnical engineering features such as scaled distance and estimated PPV.

8.5. The pipeline imputes missing numeric and categorical values.

8.6. The pipeline scales numeric features.

8.7. The pipeline one-hot encodes categorical features such as `Soil_Type`.

8.8. The pipeline compares Random Forest, Logistic Regression, and SVM models using 5-fold stratified cross-validation.

8.9. The pipeline saves the best model to `artifacts/vibration_detection_pipeline.joblib`.

8.10. The pipeline saves accuracy, macro F1, classification report, and confusion matrix to `artifacts/metrics.json`.

8.11. The pipeline saves feature importance to `artifacts/feature_importance.csv` when available.

## 9. Technical Background

9.1. MineralForge adds geotechnical blast-vibration features before training.

9.2. `Effective_Distance(m)` is calculated from burden and spacing.

9.3. `Scaled_Distance_Sqrt` is calculated as `distance / sqrt(charge weight)`.

9.4. `Scaled_Distance_Cuberoot` is calculated as `distance / charge weight^(1/3)`.

9.5. `Estimated_PPV(mm/s)` is calculated using a site-calibrated scaled-distance relation.

9.6. `Acceleration_Resultant(m/s²)` is calculated from triaxial sensor channels.

9.7. `PPV_Frequency_Product` supports structural response screening.

9.8. Scaled distance is a standard blast-vibration normalization method because it captures how charge mass and distance jointly control vibration intensity.

9.9. FFT utilities are included in `mineralforge/fft.py` for waveform workflows where dominant frequency and band energy matter as much as amplitude.

## 10. Engineering Interfaces

10.1. `mineralforge/geotech.py` provides scaled distance and PPV calculations.

10.2. `mineralforge/fft.py` provides dominant frequency and band-energy analysis.

10.3. `mineralforge/features.py` provides acoustic and vibration feature extraction.

10.4. `mineralforge/training.py` provides optional SMOTE, class weighting, hyperparameter tuning, and MLflow tracking helpers.

10.5. `dashboard.py` provides a Streamlit field-review dashboard.

10.6. `api.py` provides a FastAPI edge-inference endpoint.

10.7. `tests/` contains domain, feature, pipeline, and preprocessing tests.
