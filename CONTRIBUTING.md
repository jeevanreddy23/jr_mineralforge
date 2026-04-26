# Contributing

MineralForge is now focused on edge-deployed geotechnical risk detection. Good
contributions should improve one of four things:

1. Feature extraction from acoustic or vibration sensors.
2. Robust model training for rare high-risk events.
3. Clear SHAP-style explanations for field users.
4. TARP mappings that turn model output into operational actions.

## Development

```bash
python -m pip install -r requirements.txt
python main.py --train-report
python -m pytest tests
```

The demo must stay runnable without physical sensors. Hardware integrations
should use interfaces that can be backed by synthetic signals in tests.

## Pull request checklist

- Keep raw sensor streams out of committed data.
- Add or update tests for feature extraction, model behavior, or TARP mapping.
- Explain how the change affects underground deployment or decision quality.
- Prefer interpretable models and measurable reliability over novelty.
