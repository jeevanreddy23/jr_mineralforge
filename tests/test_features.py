import pandas as pd

from mineralforge.features.engineering import add_blast_engineering_features


def test_blast_engineering_features_add_scaled_distance_and_ppv_product():
    frame = pd.DataFrame(
        {
            "Charge_Weight(kg)": [100],
            "Burden(m)": [3],
            "Spacing(m)": [4],
            "Soil_Type": ["Hard"],
            "PPV(mm/s)": [2.5],
            "Frequency(Hz)": [40],
        }
    )
    result = add_blast_engineering_features(frame)
    assert result.loc[0, "Effective_Distance(m)"] == 5.0
    assert result.loc[0, "Scaled_Distance_Sqrt"] == 0.5
    assert result.loc[0, "PPV_Frequency_Product"] == 100.0
