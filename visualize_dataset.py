from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATA_PATH = Path("data/ground_vibration_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("visualizations")
TARGET_COLUMN = "Vibration_Level"
CLASS_ORDER = ["Low", "Medium", "High"]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.replace("Â", "", regex=False)
        .str.replace("²", "2", regex=False)
        .str.strip()
    )
    return df


def save_current(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_class_balance(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    ax = sns.countplot(data=df, x=TARGET_COLUMN, order=CLASS_ORDER, palette="viridis", hue=TARGET_COLUMN, legend=False)
    ax.set_title("Vibration Level Class Balance")
    ax.set_xlabel("Vibration level")
    ax.set_ylabel("Number of records")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    save_current(output_dir / "01_class_balance.png")


def plot_time_series(df: pd.DataFrame, output_dir: Path) -> None:
    if "Timestamp" not in df.columns:
        return

    time_df = df.copy()
    time_df["Timestamp"] = pd.to_datetime(time_df["Timestamp"], errors="coerce")
    time_df = time_df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    sensor_columns = [column for column in ["PPV(mm/s)", "Frequency(Hz)", "Seismometer(m/s2)", "Geophone(mm/s)"] if column in time_df.columns]
    if not sensor_columns:
        return

    fig, axes = plt.subplots(len(sensor_columns), 1, figsize=(11, 2.6 * len(sensor_columns)), sharex=True)
    if len(sensor_columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, sensor_columns):
        sns.lineplot(data=time_df, x="Timestamp", y=column, hue=TARGET_COLUMN, hue_order=CLASS_ORDER, ax=ax, linewidth=1.2)
        ax.set_title(f"{column} Over Time")
        ax.set_xlabel("")
        ax.legend(title=TARGET_COLUMN, loc="upper right")

    save_current(output_dir / "02_sensor_time_series.png")


def plot_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        column
        for column in [
            "Charge_Weight(kg)",
            "Burden(m)",
            "Spacing(m)",
            "PPV(mm/s)",
            "Frequency(Hz)",
            "PSD_Value",
            "Seismometer(m/s2)",
            "Geophone(mm/s)",
        ]
        if column in df.columns
    ]

    melted = df.melt(id_vars=TARGET_COLUMN, value_vars=columns, var_name="Feature", value_name="Value")
    grid = sns.FacetGrid(melted, col="Feature", col_wrap=4, sharex=False, sharey=False, height=3)
    grid.map_dataframe(sns.histplot, x="Value", hue=TARGET_COLUMN, hue_order=CLASS_ORDER, multiple="stack", palette="viridis")
    grid.set_titles("{col_name}")
    grid.fig.suptitle("Feature Distributions By Vibration Level", y=1.03)
    grid.savefig(output_dir / "03_feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(grid.fig)


def plot_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    columns = [column for column in ["PPV(mm/s)", "Frequency(Hz)", "PSD_Value", "Seismometer(m/s2)", "Geophone(mm/s)"] if column in df.columns]
    melted = df.melt(id_vars=TARGET_COLUMN, value_vars=columns, var_name="Feature", value_name="Value")

    plt.figure(figsize=(11, 6))
    ax = sns.boxplot(data=melted, x="Feature", y="Value", hue=TARGET_COLUMN, hue_order=CLASS_ORDER, palette="viridis")
    ax.set_title("Sensor Feature Ranges By Vibration Level")
    ax.set_xlabel("")
    ax.set_ylabel("Measured value")
    ax.tick_params(axis="x", rotation=20)
    save_current(output_dir / "04_sensor_boxplots.png")


def plot_relationships(df: pd.DataFrame, output_dir: Path) -> None:
    x_col = "PPV(mm/s)"
    y_col = "Frequency(Hz)"
    size_col = "Charge_Weight(kg)"
    if not {x_col, y_col, size_col}.issubset(df.columns):
        return

    plt.figure(figsize=(9, 6))
    ax = sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=TARGET_COLUMN,
        hue_order=CLASS_ORDER,
        size=size_col,
        sizes=(25, 220),
        alpha=0.75,
        palette="viridis",
    )
    ax.set_title("PPV vs Frequency, Sized By Charge Weight")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    save_current(output_dir / "05_ppv_frequency_scatter.png")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return

    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(numeric_df.corr(), cmap="vlag", center=0, annot=False, square=True, linewidths=0.3)
    ax.set_title("Numeric Feature Correlation Heatmap")
    save_current(output_dir / "06_correlation_heatmap.png")


def plot_soil_type(df: pd.DataFrame, output_dir: Path) -> None:
    if "Soil_Type" not in df.columns:
        return

    counts = df.groupby(["Soil_Type", TARGET_COLUMN]).size().reset_index(name="Count")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=counts, x="Soil_Type", y="Count", hue=TARGET_COLUMN, hue_order=CLASS_ORDER, palette="viridis")
    ax.set_title("Vibration Level By Soil Type")
    ax.set_xlabel("Soil type")
    ax.set_ylabel("Number of records")
    save_current(output_dir / "07_soil_type_by_vibration.png")


def write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_summary = df.describe(include="number").round(3).T
    class_counts = df[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER)
    soil_counts = df["Soil_Type"].value_counts() if "Soil_Type" in df.columns else pd.Series(dtype=int)

    summary = [
        "# Dataset Visualization Summary",
        "",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        "",
        "## Class Counts",
        "",
        class_counts.to_markdown(),
        "",
        "## Soil Type Counts",
        "",
        soil_counts.to_markdown(),
        "",
        "## Numeric Summary",
        "",
        numeric_summary.to_markdown(),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")


def create_visualizations(csv_path: Path, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = clean_column_names(pd.read_csv(csv_path))
    plot_class_balance(df, output_dir)
    plot_time_series(df, output_dir)
    plot_distributions(df, output_dir)
    plot_boxplots(df, output_dir)
    plot_relationships(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_soil_type(df, output_dir)
    write_summary(df, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visualizations for the ground vibration dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the dataset CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for chart outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_visualizations(args.data, args.output_dir)
    print(f"Saved visualizations to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
