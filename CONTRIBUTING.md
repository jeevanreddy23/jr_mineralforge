# Contributing

MineralForge is focused on one use case: blast vibration risk prediction for mining/geotechnical review.

Good contributions should improve:

1. Blast vibration feature engineering.
2. Model training, tuning, and high-risk recall.
3. Data QA, model scoring, and geotechnical recommendation workflows.
4. FastAPI or Streamlit usability for field review.
5. Documentation that clearly separates demo data from validated mine deployment.

## Development

```bash
python -m pip install -r requirements.txt
python train_pipeline.py --tuner grid
python predict.py --input data/ground_vibration_dataset.csv --output artifacts/predictions.csv
python -m pytest
```

## Pull Request Checklist

1. Keep the repository focused on blast vibration risk prediction.
2. Do not add unrelated mineral exploration, RAG, or generic agent content.
3. Add or update tests for feature engineering, model behavior, prediction workflow, API, or dashboard changes.
4. Do not overclaim validation. Site-specific calibration is required before operational use.
