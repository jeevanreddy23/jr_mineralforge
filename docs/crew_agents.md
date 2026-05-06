# CrewAI Multi-Agent Design

## Agents

1. Blast Data QA Agent checks input fields such as charge weight, burden, spacing, PPV, frequency, and soil type.
2. Vibration Risk Model Agent scores the event with the trained blast vibration classifier.
3. Geotechnical TARP Agent converts the risk level into a practical field-review recommendation.

## Design Choice

The MVP keeps prediction deterministic. CrewAI provides the review structure, while the trained model remains the auditable scoring component.
