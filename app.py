"""
JR MineralForge – Gradio Web UI
==================================
Interactive web interface for Team JR with:
  - RAG chat for geological queries and winner analysis
  - One-click data ingestion from SARIG and GA
  - Prospectivity pipeline trigger
  - Report generation with download
  - Embedded interactive map
"""

from __future__ import annotations

import json
import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

# Self-Healing Pre-flight Setup
from utils.dependency_manager import check_and_install_dependencies
check_and_install_dependencies()

import gradio as gr

from config.settings import (
    BRAND_NAME, TEAM_NAME, BRAND_HEADER, BRAND_FOOTER,
    REPORTS_DIR, LLM_PROVIDER, LLM_MODEL,
)
from utils.logging_utils import get_logger

log = get_logger(__name__)

# ─── Global state ────────────────────────────────────────────────
_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from main import JRMineralForgeOrchestrator
        from utils.llm_factory import get_llm
        try:
            llm = get_llm()
        except Exception as e:
            log.warning(f"LLM init failed: {e}")
            llm = None
        _orchestrator = JRMineralForgeOrchestrator(llm=llm, verbose=False)
    return _orchestrator


# ─── Actions ─────────────────────────────────────────────────────

def get_supervisor():
    from main import RigidMultiAgentSupervisor
    return RigidMultiAgentSupervisor()

def chat_supervisor_nli(message: str, history: list) -> tuple[str, list]:
    """Execute simple Supervisor NLP target routing."""
    if not message.strip():
        return "", history
        
    sup = get_supervisor()
    try:
        # We only hit Agent 1 (NLI Processor) explicitly
        params = sup.run_nli_extraction(message)
        response = f"✅ NLI Extracted Constraints: \nCommodity: {params.get('commodity', 'Cu-Au')}\nRegion: {params.get('province_id', 'mount_woods')}\nReady for Execution."
    except Exception as e:
        response = f"❌ Error extracting constraints: {str(e)}"
        
    history.append({"role": "user", "content": str(message)})
    history.append({"role": "assistant", "content": str(response)})
    
    return "", history


def run_prospectivity(province_id: str = "mount_woods", file_obj=None) -> tuple[str, str, object, str]:
    try:
        from agents.prospectivity_agent import ProspectivityMappingAgent
        from config.settings import AUSTRALIAN_PROVINCES, BoundingBox
        import geopandas as gpd
        import pandas as pd
        from pathlib import Path
        import json

        bbox = AUSTRALIAN_PROVINCES.get(province_id, AUSTRALIAN_PROVINCES["mount_woods"])
        
        # 1. Parse Tenement Boundary
        status_logs = []
        if file_obj is not None:
            from utils.geometry_handler import validate_and_fix_geometry
            file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
            status_logs.append(f"🔍 Analyzing uploaded file: {Path(file_path).name}…")
            fixed_bbox = validate_and_fix_geometry(file_path)
            if fixed_bbox:
                bbox = fixed_bbox
                status_logs.append(f"🛠️ Self-Correction Applied: {bbox.description}")
            else:
                status_logs.append(f"⚠️ Geometry fix failed, falling back to {province_id}")
        else:
            status_logs.append(f"📍 Using default region: {province_id}")
                
        # 2. Force Dynamic Ingestion
        from agents.data_ingestion_agent import SARIGIngestionAgent
        status_logs.append(f"🛰️ Pulling live OGC data for {bbox.name}…")
        ingestor = SARIGIngestionAgent(bbox=bbox)
        ingestor.ingest_all()
        status_logs.append("✅ Data Ingestion Synchronized.")

        # 3. Target Generation
        status_logs.append("🧠 Initializing ML Prospectivity Pipeline…")
        agent = ProspectivityMappingAgent(bbox=bbox)
        results = agent.run_full_pipeline()
        
        if "error" in results:
             raise ValueError(results["error"])

        status_logs.append(f"🎯 Generated {results.get('n_targets', 0)} high-confidence drill targets.")
        
        # 4. Map Extraction
        map_path = results.get("interactive_map", "")
        if map_path and Path(map_path).exists():
            with open(map_path, "r", encoding="utf-8") as f:
                map_html = f.read()
        else:
            map_html = "<p>Map not generated.</p>"
            
        # 5. Targets Dataframe Extraction & KPIs
        csv_path = results.get("targets_csv", "")
        target_count = 0
        max_conf = 0.0
        
        if csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            target_count = len(df)
            max_conf = float(df["confidence_score"].max()) if "confidence_score" in df.columns and not df.empty else 0.0
            
            df_out = pd.DataFrame()
            df_out["Target ID"] = df.get("target_label", [f"T-{i}" for i in range(len(df))])
            df_out["Confidence"] = df.get("confidence_category", "Unknown").astype(str) + " (" + df.get("confidence_score", 0).astype(str) + "%)"
            df_out["Primary Driver (SHAP)"] = df.get("primary_driver", "Unknown").astype(str).str.upper()
            df_out["Easting"] = df.get("lon", 0).round(2)
            df_out["Northing"] = df.get("lat", 0).round(2)
            df_out = df_out.head(10)
        else:
            df_out = pd.DataFrame(columns=["Target ID", "Confidence", "Primary Driver (SHAP)", "Easting", "Northing"])

        kpi_metrics = f"""
        <div class="kpi-container">
            <div class="kpi-box"><div class="kpi-title">Data Status</div><div class="kpi-value" id="kpi-data">Live</div></div>
            <div class="kpi-box"><div class="kpi-title">ML Samples</div><div class="kpi-value" id="kpi-features">{len(df) if csv_path else 'None'}</div></div>
            <div class="kpi-box"><div class="kpi-title">Max Confidence</div><div class="kpi-value" id="kpi-conf">{max_conf:.1f}%</div></div>
            <div class="kpi-box"><div class="kpi-title">Targets</div><div class="kpi-value" id="kpi-targets">{target_count}</div></div>
        </div>
        """

        full_logs = "\n".join(status_logs)
        return f"✅ Pipeline Complete\n\n{full_logs}", map_html, df_out, kpi_metrics
    except Exception as e:
        import traceback
        log.error(f"Pipeline Crash: {e}\n{traceback.format_exc()}")
        fallback_kpi = f"""
        <div class="kpi-container">
            <div class="kpi-box"><div class="kpi-title">Execution</div><div class="kpi-value" style="color: #ff4a4a;">FAILED</div></div>
            <div class="kpi-box"><div class="kpi-title">Error Type</div><div class="kpi-value" style="color: #ff4a4a; font-size: 0.9em;">{type(e).__name__}</div></div>
        </div>
        """
        err_msg = f"❌ Error: {str(e)}\n\n💡 Try expanding the search region or checking if the uploaded file contains valid Australian locations."
        return err_msg, f"<div style='color:red; padding:20px;'><h3>Pipeline Error</h3><p>{str(e)}</p></div>", pd.DataFrame(), fallback_kpi


def run_national_sweep() -> str:
    try:
        orc = get_orchestrator()
        return orc.run("run_national_sweep")
    except Exception as e:
        return f"❌ Error: {e}"


def run_check_data() -> str:
    try:
        from agents.data_ingestion_agent import check_data_availability
        return check_data_availability.invoke({})
    except Exception as e:
        return f"❌ Error: {e}"


def generate_report() -> tuple[str, str]:
    try:
        from agents.reporting_agent import generate_text_report
        report = generate_text_report()
        report_path = str(REPORTS_DIR / "jr_mineralforge_report.txt")
        return report[:3000] + "\n…(truncated)", report_path
    except Exception as e:
        return f"❌ Error: {e}", ""


def list_winners() -> str:
    try:
        from agents.knowledge_agent import list_all_winners
        return list_all_winners.invoke({})
    except Exception as e:
        return f"❌ Error: {e}"


def analyse_winner(team: str) -> str:
    try:
        from agents.knowledge_agent import analyse_winner_strategy
        return analyse_winner_strategy.invoke({"team_name": team})
    except Exception as e:
        return f"❌ Error: {e}"


def rebuild_kb() -> str:
    try:
        from agents.knowledge_agent import rebuild_knowledge_base
        return rebuild_knowledge_base.invoke({})
    except Exception as e:
        return f"❌ Error: {e}"


# ─── Custom CSS ──────────────────────────────────────────────────

CUSTOM_CSS = """
:root {
    --brand-gold: #FFD700;
    --brand-dark: #0d1117;
    --brand-blue: #1a2035;
    --brand-accent: #4a9eff;
    --brand-text: #e6edf3;
}

body, .gradio-container {
    background: var(--brand-dark) !important;
    color: var(--brand-text) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.brand-header {
    background: linear-gradient(135deg, #0d1117 0%, #1a2035 50%, #0d1117 100%);
    border: 1px solid var(--brand-gold);
    border-radius: 12px;
    padding: 20px 30px;
    text-align: center;
    margin-bottom: 20px;
}

.brand-header h1 {
    color: var(--brand-gold) !important;
    font-size: 2em;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 0 20px rgba(255,215,0,0.4);
}

.brand-header p {
    color: #aac4ff;
    font-size: 0.95em;
    margin: 8px 0 0 0;
}

.tab-nav button {
    background: #1a2035 !important;
    color: #aac4ff !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    font-weight: 600;
}

.tab-nav button.selected {
    background: var(--brand-gold) !important;
    color: #0d1117 !important;
    border-color: var(--brand-gold) !important;
}

.gr-button.primary {
    background: linear-gradient(135deg, #FFD700, #ff9500) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}

.gr-button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(255,215,0,0.4) !important;
}

.gr-button.secondary {
    background: #1a2035 !important;
    color: var(--brand-gold) !important;
    border: 1px solid var(--brand-gold) !important;
    border-radius: 8px !important;
}

.kpi-container {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    margin-bottom: 20px;
}

.kpi-box {
    background: #161b22;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    flex: 1;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}

.kpi-title {
    color: #8b949e;
    font-size: 0.9em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.kpi-value {
    color: var(--brand-gold);
    font-size: 2.2em;
    font-weight: 800;
    margin: 0;
}

.gr-textbox, .gr-chatbot {
    background: #161b22 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
    color: var(--brand-text) !important;
}

.gr-chatbot .message.user {
    background: #1a2035 !important;
    border-left: 3px solid var(--brand-gold) !important;
}

.gr-chatbot .message.bot {
    background: #1e2d4d !important;
    border-left: 3px solid var(--brand-accent) !important;
}

footer { display: none !important; }
"""


# ─── Build UI ────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    demo = gr.Blocks(title=f"{BRAND_NAME} | {TEAM_NAME}")
    with demo:
        # Header
        gr.HTML(f"""
        <div class="brand-header">
            <h1>🏔️ {BRAND_NAME} | Executive Dashboard</h1>
            <p>Commercial "Drill or Drop" Pipeline (Mount Woods Base)</p>
            <p style="color: #666; font-size: 0.8em; margin-top: 4px;">
                IOCG · Copper-Gold · Gawler Craton · ML Targeted
            </p>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ── Tab 1: Executive Dashboard ────────────────────────────────────
            with gr.Tab("🎯 Drill or Drop Dashboard"):
                kpi_html_display = gr.HTML("""
                <div class="kpi-container">
                    <div class="kpi-box"><div class="kpi-title">Data Processed</div><div class="kpi-value" id="kpi-data">-- GB</div></div>
                    <div class="kpi-box"><div class="kpi-title">Features Built</div><div class="kpi-value" id="kpi-features">--</div></div>
                    <div class="kpi-box"><div class="kpi-title">Max Confidence</div><div class="kpi-value" id="kpi-conf">--%</div></div>
                    <div class="kpi-box"><div class="kpi-title">Targets Generated</div><div class="kpi-value" id="kpi-targets">--</div></div>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        tenement_upload = gr.File(label="Upload Tenement Boundary (SHP/KML/GeoJSON)", file_types=[".shp", ".kml", ".geojson", ".zip"])
                        province_select = gr.Dropdown(
                            choices=["mount_woods", "yilgarn_craton", "mt_isa_inlier", "lachlan_fold_belt"],
                            value="mount_woods",
                            label="Target Province (Fallback Region)"
                        )
                        run_btn = gr.Button("▶ Initialize ML Pipeline", variant="primary")
                        pipeline_status = gr.Textbox(label="Pipeline Execution Status", lines=12)

                    with gr.Column(scale=2):
                        map_display = gr.HTML(label="High-Resolution Interactive Targeting Map")
                
                gr.Markdown("### Ranked Drill Targets (Exportable)")
                drill_targets_grid = gr.Dataframe(
                    headers=["Target ID", "Confidence", "Primary Driver (SHAP)", "Easting", "Northing"],
                    datatype=["str", "str", "str", "number", "number"],
                    row_count=5,
                    column_count=(5, "fixed"),
                )
                
                with gr.Accordion("LLM Reporting Sandbox (Analyst Chat)", open=False):
                    chatbot = gr.Chatbot(label="JR MineralForge Analyst", height=300)
                    chat_input = gr.Textbox(placeholder='e.g. Provide a business case for targeting JR-T001', label="Analyst Query")
                    chat_input.submit(chat_supervisor_nli, [chat_input, chatbot], [chat_input, chatbot])

                run_btn.click(
                    run_prospectivity, 
                    inputs=[province_select, tenement_upload], 
                    outputs=[pipeline_status, map_display, drill_targets_grid, kpi_html_display]
                )

            # ── Tab 2: Data & Engine Settings ─────────────────────────
            with gr.Tab("⚙️ Engine Architecture Config"):
                gr.Markdown("""
                ### Dynamic OGC Endpoint Routing
                Manage data source ingestion priorities and WFS clipping layers.
                """)
                
                with gr.Row():
                    sarig_btn = gr.Button("🦘 Force SARIG Clip (South Australia)", variant="primary")
                    ga_btn = gr.Button("🌏 Force GA Sync (National)", variant="secondary")

                ingest_output = gr.Textbox(label="Ingestion Trace Logs", lines=15)
                sarig_btn.click(lambda x: "SARIG Explicit Sync is now managed by Supervisor NLI.", inputs=province_select, outputs=ingest_output)
                ga_btn.click(lambda x: "GA Explicit Sync is now managed by Supervisor NLI.", inputs=province_select, outputs=ingest_output)

        # Footer
        gr.HTML(f"""
        <div style="text-align: center; padding: 16px; color: #555;
                    font-size: 0.85em; border-top: 1px solid #2d3748; margin-top: 20px;">
            {BRAND_FOOTER} | Commercial Evaluation System
        </div>
        """)

    return demo


# ─── Launch ──────────────────────────────────────────────────────

def main():
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False,
        favicon_path=None,
        show_error=True,
    )


if __name__ == "__main__":
    main()
