import pandas as pd

from mineralforge.data.processing import split_features_target
from mineralforge.features.engineering import prepare_training_frame
from mineralforge.utils.paths import TARGET_COLUMN


def test_prepare_training_frame_adds_blast_features_and_time_features():
    frame = pd.DataFrame(
        [
            {
                "Timestamp": "2026-04-26T08:15:00",
                "Blast_ID": "B1",
                "Charge_Weight(kg)": 100,
                "Burden(m)": 3,
                "Spacing(m)": 4,
                "Soil_Type": "Hard",
                "PPV(mm/s)": 2,
                "Frequency(Hz)": 40,
                TARGET_COLUMN: "Low",
            }
        ]
    )
    processed = prepare_training_frame(frame)
    x, y = split_features_target(processed)
    assert "hour" in processed.columns
    assert "Scaled_Distance_Sqrt" in processed.columns
    assert "Blast_ID" not in x.columns
    assert y.iloc[0] == "Low"
