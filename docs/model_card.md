# Model Card: Blast Vibration Risk Predictor

## Intended Use

This model predicts `Low`, `Medium`, or `High` blast vibration risk from blast design and vibration monitoring inputs.

## Inputs

- Charge weight.
- Burden.
- Spacing.
- PPV.
- Frequency.
- Soil or rock type.

## Outputs

- Vibration risk class.
- Estimated PPV.
- Class probabilities when the selected model supports them.
- Field-review recommendation.

## Validation Status

The repository is an MVP trained on the included CSV dataset. It is not validated for production mine safety decisions.

## Safety Note

Use this project as decision support only. Site-specific calibration and professional geotechnical review are required before operational use.
