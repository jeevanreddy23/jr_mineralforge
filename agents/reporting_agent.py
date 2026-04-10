"""
JR MineralForge – Explanation, Interpretability & Reporting Agent
==================================================================
Produces branded reports, SHAP explanation diagrams, ranked target summaries,
and human-readable geological interpretations for Team JR outputs.
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Optional matplotlib for SHAP plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from langchain.tools import tool

from config.settings import (
    REPORTS_DIR, BRAND_NAME, TEAM_NAME, BRAND_HEADER, BRAND_FOOTER,
    WINNERS_CONTEXT, MOUNT_WOODS_BBOX, ML_CONFIG,
)
from utils.logging_utils import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────────────────────────

REPORT_TEMPLATE = """\
================================================================================
{brand_name} – Mineral Prospectivity Report
{header}
Generated: {timestamp}
Area of Interest: {aoi}
================================================================================

EXECUTIVE SUMMARY
-----------------
{executive_summary}

TOP DRILL TARGETS
-----------------
{targets_table}

MACHINE LEARNING PERFORMANCE
----------------------------
{ml_summary}

GEOLOGICAL INTERPRETATION
--------------------------
{geo_interpretation}

HOW JR MINERALFORGE IMPROVES ON PREVIOUS CHALLENGE WINNERS
-----------------------------------------------------------
{winners_comparison}

DATA SOURCES USED
-----------------
{data_sources}

DISCLAIMER
----------
This prospectivity analysis is generated from open Australian government data
(SARIG, Geoscience Australia) using AI/ML methods. Results are probabilistic,
not deterministic, and should be validated by qualified geoscientists before
any exploration decision is made.

================================================================================
{brand_footer}
================================================================================
"""

def generate_text_report(
    targets_df: Optional[pd.DataFrame] = None,
    cv_results: Optional[Dict] = None,
    feature_importance: Optional[Dict] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Generate a full text report for Team JR mineral prospectivity results."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S AEST")

    # Executive summary
    n_targets = len(targets_df) if targets_df is not None else 0
    exec_summary = (
        f"JR MineralForge has analysed the {MOUNT_WOODS_BBOX.name} area using "
        f"open data from SARIG and Geoscience Australia, applying a multi-agent "
        f"AI workflow with spatial cross-validation, ensemble ML, and SHAP "
        f"interpretability. A total of {n_targets} ranked drill targets were "
        f"identified, incorporating anti-noise geophysical filtering and "
        f"bootstrap uncertainty quantification."
    )

    # Targets table
    if targets_df is not None and len(targets_df) > 0:
        display_cols = [c for c in ["target_label", "rank", "confidence_score",
                                     "confidence_category", "lat", "lon", "uncertainty"]
                        if c in targets_df.columns]
        targets_table = targets_df[display_cols].head(20).to_string(index=False)
    else:
        targets_table = "No targets generated yet. Run the prospectivity pipeline."

    # ML summary
    if cv_results:
        ml_summary = (
            f"Spatial Cross-Validation AUC: {cv_results.get('mean_auc', 0):.4f} "
            f"± {cv_results.get('std_auc', 0):.4f}\n"
            f"Folds: {cv_results.get('fold_aucs', [])}\n"
            f"Ensemble: Balanced Random Forest + XGBoost + LightGBM (calibrated)\n"
            f"Anti-overfitting: Spatial block CV, L1/L2 regularisation, SHAP feature selection\n"
            f"Anti-noise: Wavelet denoising, median filtering, Isolation Forest"
        )
    else:
        ml_summary = "ML pipeline not yet run. Execute the full workflow to see results."

    # Geological interpretation
    geo_interpretation = """
The Mount Woods / Prominent Hill region sits within the Olympic IOCG Province
on the Gawler Craton, South Australia. The area is prospective for:
  • Iron Oxide Copper-Gold (IOCG) deposits (e.g., Olympic Dam type)
  • Magnetite-hematite breccia systems
  • Uranium-REE associations with hematite alteration

Key targeting criteria applied:
  1. Circular/elliptical negative TMI anomalies (magnetite destruction)
  2. Associated gravity lows (low-density breccia body)
  3. Elevated K% radiometrics (potassic alteration halo)
  4. Spatial proximity to Hiltaba Suite granite contacts
  5. Second-order fault intersections within Olympic-scale fault corridors
"""

    # Winners comparison
    winners_comparison = f"""
{WINNERS_CONTEXT}

Building on the 2019 OZ Minerals Explorer Challenge:
  • Team Guru (1st): JR MineralForge extends their mineral systems framework with
    automated open-data ingestion from SARIG/GA and LangChain RAG.
  • DeepSightX (2nd): JR improves deep learning uncertainty (MC dropout → bootstrap)
    and adds spatial cross-validation to prevent leakage.
  • deCODES (3rd): JR uses SHAP instead of Bayesian belief networks for more
    transparent, geological feature-level interpretability.
  • SRK Consulting (Fusion): JR automates the multi-dataset fusion via xarray
    datacube stacking and standardised CRS/resolution alignment.
  • OreFox: JR's LangChain RAG replaces manual NLP with a persistent, updatable
    knowledge base over all open reports and exploration data.
"""

    data_sources = """
  1. SARIG (catalog.sarig.sa.gov.au):
     - SA Surface Geology 1:250,000
     - Mineral Occurrences (MINOCC)
     - Drillhole Collars
     - Geochemistry (soils)
     - WFS services (bbox-filtered)

  2. Geoscience Australia (portal.ga.gov.au / dap.nci.org.au):
     - National Magnetic Compilation (TMI)
     - Australian National Gravity Compilation (Bouguer)
     - Radiometric Map of Australia (K, eTh, eU, Total Dose)
     - 1 Second SRTM Digital Elevation Model
     - OZMIN Mineral Occurrences (WFS)
"""

    report = REPORT_TEMPLATE.format(
        brand_name=BRAND_NAME,
        header=BRAND_HEADER,
        timestamp=timestamp,
        aoi=MOUNT_WOODS_BBOX.name,
        executive_summary=exec_summary,
        targets_table=targets_table,
        ml_summary=ml_summary,
        geo_interpretation=geo_interpretation,
        winners_comparison=winners_comparison,
        data_sources=data_sources,
        brand_footer=BRAND_FOOTER,
    )

    if output_path is None:
        output_path = REPORTS_DIR / "jr_mineralforge_report.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    log.info(f"Report saved → {output_path}")
    return report


# ─────────────────────────────────────────────────────────────────
# SHAP Summary Plot
# ─────────────────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Generate a SHAP beeswarm summary plot for Team JR report branding."""
    if not HAS_SHAP or not HAS_MPL:
        log.warning("SHAP or matplotlib not available — skipping SHAP plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#0a0a1a")
    fig.patch.set_facecolor("#0a0a1a")

    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1][:20]  # top 20
    names = [feature_names[i] for i in sorted_idx]
    vals = mean_abs[sorted_idx]

    colours = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(names)))
    ax.barh(range(len(names)), vals[::-1], color=colours[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], color="white", fontsize=10)
    ax.set_xlabel("Mean |SHAP Value|", color="white")
    ax.set_title(
        f"{BRAND_NAME} – Feature Importance (SHAP)\n{BRAND_HEADER}",
        color="#FFD700", fontsize=11, pad=15,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.text(
        0.01, -0.08, BRAND_FOOTER,
        transform=ax.transAxes, fontsize=7, color="#888",
        ha="left",
    )
    plt.tight_layout()

    if output_path is None:
        output_path = REPORTS_DIR / "shap_importance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"SHAP plot saved → {output_path}")
    return output_path


def plot_target_confidence(
    targets_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Bar chart of top-ranked target confidence scores."""
    if not HAS_MPL or targets_df is None or len(targets_df) == 0:
        return None

    df = targets_df.head(20).copy()
    if "confidence_score" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    labels = df.get("target_label", pd.Series(range(len(df)))).values
    scores = df["confidence_score"].values
    uncertainties = df.get("uncertainty", pd.Series(np.zeros(len(df)))).values * 100

    colours = ["#ff4444" if s > 85 else "#ff9900" if s > 70 else "#44cc44" for s in scores]
    bars = ax.barh(range(len(labels)), scores, color=colours, alpha=0.85)
    if uncertainties.sum() > 0:
        ax.barh(
            range(len(labels)), uncertainties, left=scores,
            color="#888888", alpha=0.4, label="Uncertainty (±)"
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color="white", fontsize=9)
    ax.set_xlabel("Confidence Score (%)", color="white")
    ax.set_title(
        f"{BRAND_NAME} – Ranked Drill Targets\n{BRAND_HEADER}",
        color="#FFD700", fontsize=11, pad=10,
    )
    ax.set_xlim(0, 110)
    ax.axvline(x=70, color="#ff9900", linestyle="--", alpha=0.5, label="High threshold")
    ax.axvline(x=85, color="#ff4444", linestyle="--", alpha=0.5, label="Very High threshold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.text(
        0.01, -0.08, BRAND_FOOTER,
        transform=ax.transAxes, fontsize=7, color="#888", ha="left"
    )
    plt.tight_layout()

    if output_path is None:
        output_path = REPORTS_DIR / "target_confidence.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Target confidence chart saved → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────
# LangChain Tools
# ─────────────────────────────────────────────────────────────────

@tool
def generate_full_report(targets_json: str = "{}") -> str:
    """
    Generate a complete JR MineralForge prospectivity report.
    targets_json: JSON string with target data (optional; uses last pipeline run data).
    Returns the full report text and saves it to the reports directory.
    """
    try:
        targets_data = json.loads(targets_json) if targets_json.strip() not in ("{}", "") else {}
        targets_df = pd.DataFrame(targets_data) if targets_data else None
    except Exception:
        targets_df = None

    report = generate_text_report(targets_df=targets_df)
    return f"Report generated successfully!\n\nPreview (first 1000 chars):\n{report[:1000]}…"


@tool
def explain_target(target_label: str, features_json: str = "{}") -> str:
    """
    Provide a geological explanation for a specific ranked drill target.
    target_label: e.g. 'JR-T001'
    features_json: JSON dict of feature values at the target location (optional).
    """
    try:
        features = json.loads(features_json) if features_json.strip() not in ("{}", "") else {}
    except Exception:
        features = {}

    lines = [
        f"[{BRAND_NAME}] Geological Explanation for {target_label}",
        "=" * 55,
    ]

    # Build feature narrative
    if features:
        lines.append("\nFeature Values at Target Location:")
        for feat, val in features.items():
            lines.append(f"  {feat}: {val}")
        lines.append("")

        # Geological interpretation
        if "tmi" in features and features.get("tmi", 0) < -100:
            lines.append("📍 Strong negative TMI anomaly → possible magnetite destruction / hematite breccia (IOCG proxy)")
        if "gravity" in features and features.get("gravity", 0) < -10:
            lines.append("📍 Gravity low → low-density breccia body or granitic intrusion at depth (IOCG proxy)")
        if "k_pct" in features and features.get("k_pct", 0) > 3.0:
            lines.append("📍 Elevated K% → potassic alteration halo (strong IOCG proxy)")
        if "k_eth_ratio" in features and features.get("k_eth_ratio", 0) > 0.5:
            lines.append("📍 High K/eTh ratio → confirms K enrichment is alteration-related, not lithological")
    else:
        lines.append("No feature data provided. Query the prospectivity pipeline for feature extraction.")

    lines.append(f"\n{BRAND_FOOTER}")
    return "\n".join(lines)
