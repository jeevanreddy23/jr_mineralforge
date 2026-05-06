from pathlib import Path

import pandas as pd

from mineralforge.agents.crew import build_crewai_team, run_blast_vibration_crew


def test_crewai_team_is_available_or_optional():
    agents = build_crewai_team()
    assert agents is None or len(agents) == 3


def test_crew_workflow_scores_csv(tmp_path):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    pd.DataFrame(
        [
            {
                "Timestamp": "2024-01-01 00:00:00",
                "Blast_ID": "B1",
                "Charge_Weight(kg)": 100,
                "Burden(m)": 3,
                "Spacing(m)": 4,
                "Delay(ms)": 20,
                "Soil_Type": "Medium",
                "Temperature(°C)": 20,
                "Wind_Speed(m/s)": 1,
                "Seismometer(m/s²)": 0.5,
                "Geophone(mm/s)": 1.2,
                "Acc_X(m/s²)": 0.1,
                "Acc_Y(m/s²)": 0.1,
                "Acc_Z(m/s²)": 0.1,
                "PSD_Value": 0.2,
                "PPV(mm/s)": 3.0,
                "Frequency(Hz)": 35,
            }
        ]
    ).to_csv(input_csv, index=False)
    if not Path("artifacts/vibration_detection_pipeline.joblib").exists():
        return
    result = run_blast_vibration_crew(input_csv, output_csv)
    assert result.rows_scored == 1
    assert output_csv.exists()
