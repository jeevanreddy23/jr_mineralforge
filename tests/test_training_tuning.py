import pandas as pd

from train_pipeline import add_engineering_features, run_grid_search


def test_grid_search_returns_tuning_summary():
    frame = pd.DataFrame(
        {
            "Charge_Weight(kg)": [100, 120, 300, 320, 80, 90, 450, 470, 200, 220, 510, 530],
            "Burden(m)": [5, 5.2, 2, 2.1, 7, 7.1, 1.5, 1.6, 4, 4.1, 1.2, 1.3],
            "Spacing(m)": [5, 5.2, 2, 2.1, 7, 7.1, 1.5, 1.6, 4, 4.1, 1.2, 1.3],
            "Soil_Type": ["Hard", "Hard", "Soft", "Soft", "Medium", "Medium"] * 2,
            "PPV(mm/s)": [1, 1.1, 8, 8.2, 3, 3.1, 12, 12.4, 5, 5.1, 15, 15.3],
            "Frequency(Hz)": [30, 31, 45, 46, 25, 26, 50, 51, 35, 36, 55, 56],
        }
    )
    x = add_engineering_features(frame)
    y = pd.Series(["Low", "Low", "High", "High", "Medium", "Medium"] * 2)
    _, summary = run_grid_search(x, y)
    assert summary["method"] == "GridSearchCV"
    assert summary["trials"] > 0
    assert summary["best_score"] >= 0
