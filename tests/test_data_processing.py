import pandas as pd

from mineralforge.data_processing import preprocess_events


def test_preprocess_events_handles_null_soil_type_and_adds_scaled_distance():
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-26T08:15:00",
                "soil_type": None,
                "charge_mass_kg": 100,
                "distance_m": 50,
                "risk_event": 1,
            }
        ]
    )
    processed = preprocess_events(frame)
    assert processed.loc[0, "soil_type"] == "rock"
    assert processed.loc[0, "scaled_distance_sqrt"] == 5.0
    assert "soil_type_rock" in processed.columns
