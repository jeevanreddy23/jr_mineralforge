# Ground Vibration Seismic Detection Pipeline

This project trains a machine learning classifier for the included dataset at:

```text
data/ground_vibration_dataset.csv
```

The pipeline predicts `Vibration_Level` (`Low`, `Medium`, `High`) from blast, ground, and seismic sensor features.

## Files

- `train_pipeline.py` trains and saves the best model.
- `predict.py` loads the saved model and writes predictions for a CSV.
- `visualize_dataset.py` creates exploratory charts for the dataset.
- `data/ground_vibration_dataset.csv` is the vibration seismic detection dataset.
- `requirements.txt` lists the Python packages needed.
- `artifacts/` is created after training and contains the model plus metrics.
  - `vibration_detection_pipeline.joblib`: trained reusable model.
  - `metrics.json`: accuracy, macro F1, classification report, and confusion matrix.
  - `feature_importance.csv`: strongest predictors when the selected model supports it.

## Train

```powershell
python train_pipeline.py
```

Optional custom paths:

```powershell
python train_pipeline.py --data data\ground_vibration_dataset.csv --output-dir artifacts
```

## Predict

```powershell
python predict.py --input data\ground_vibration_dataset.csv --output artifacts\predictions.csv
```

## Visualize

```powershell
python visualize_dataset.py
```

## What The Pipeline Does

1. Loads the CSV.
2. Converts `Timestamp` into `hour`, `dayofweek`, and `month`.
3. Drops `Blast_ID` because it is an identifier, not a reusable physical predictor.
4. Imputes missing numeric and categorical values.
5. Scales numeric features.
6. One-hot encodes categorical features such as `Soil_Type`.
7. Compares Random Forest, Logistic Regression, and SVM using 5-fold stratified cross-validation.
8. Saves the best model to `artifacts/vibration_detection_pipeline.joblib`.
9. Saves accuracy, macro F1, classification report, and confusion matrix to `artifacts/metrics.json`.
10. Saves feature importance to `artifacts/feature_importance.csv` when available.
