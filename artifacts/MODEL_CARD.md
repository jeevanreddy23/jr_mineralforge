# Model Card: Blast Vibration Risk Predictor

## Intended Use
Predict Low, Medium, or High blast vibration risk from charge weight, burden, spacing, PPV, frequency, and soil/rock type.

## Validation Status
This model is trained on the included CSV dataset. It is an MVP and is not validated for production mine safety decisions.

## Key Metrics
- Macro F1: 1.000
- Test Accuracy: 1.000
- High Risk Recall: 1.000

## Output
The model outputs a vibration risk class. The application also reports estimated PPV and an explanation/recommendation layer.
