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

def chat_with_jr(message: str, history: list) -> tuple[str, list]:
    """Handle RAG chat queries."""
    if not message.strip():
        return "", history
    orc = get_orchestrator()
    try:
        response = orc.run(message)
    except Exception as e:
        response = f"Error: {e}"
    history.append((message, response))
    return "", history


def run_tasmania_ingest(province_id: str = "tasmania_mount_read") -> str:
    try:
        from agents.data_ingestion_agent import TasmaniaIngestionAgent
        from config.settings import AUSTRALIAN_PROVINCES
        bbox = AUSTRALIAN_PROVINCES.get(province_id, AUSTRALIAN_PROVINCES["tasmania_mount_read"])
        agent = TasmaniaIngestionAgent(bbox=bbox)
        results = agent.ingest_all()
        return f"✅ MRT (Tasmania) Ingestion Complete for {bbox.name}\n\n{json.dumps(results, indent=2)}"
    except Exception as e:
        return f"❌ Error: {e}"


def run_ga_ingest(province_id: str = "mount_woods") -> str:
    try:
        from agents.data_ingestion_agent import GAIngestionAgent
        from config.settings import AUSTRALIAN_PROVINCES
        bbox = AUSTRALIAN_PROVINCES.get(province_id, AUSTRALIAN_PROVINCES["mount_woods"])
        agent = GAIngestionAgent(bbox=bbox)
        results = agent.ingest_all()
        return f"✅ GA Ingestion Complete for {bbox.name}\n\n{json.dumps(results, indent=2)}"
    except Exception as e:
        return f"❌ Error: {e}"


def run_state_ingest(state_code, layer, province_id) -> str:
    try:
        from agents.data_ingestion_agent import StateSurveyIngestionAgent
        from config.settings import AUSTRALIAN_PROVINCES
        bbox = AUSTRALIAN_PROVINCES.get(province_id, AUSTRALIAN_PROVINCES["mount_woods"])
        agent = StateSurveyIngestionAgent(bbox=bbox)
        res = agent.ingest_from_portal(state_code, layer)
        return f"✅ State Ingestion Started: {res}"
    except Exception as e:
        return f"❌ Error: {e}"


def run_prospectivity(province_id: str = "mount_woods") -> tuple[str, str]:
    try:
        from agents.prospectivity_agent import ProspectivityMappingAgent
        from config.settings import AUSTRALIAN_PROVINCES
        bbox = AUSTRALIAN_PROVINCES.get(province_id, AUSTRALIAN_PROVINCES["mount_woods"])
        agent = ProspectivityMappingAgent(bbox=bbox)
        results = agent.run_full_pipeline()
        map_path = results.get("interactive_map", "")
        if map_path and Path(map_path).exists():
            with open(map_path, "r", encoding="utf-8") as f:
                map_html = f.read()
        else:
            map_html = "<p>Map not generated yet.</p>"
        return f"✅ Pipeline Complete for {bbox.name}\n{json.dumps(results, indent=2)}", map_html
    except Exception as e:
        return f"❌ Error: {e}", "<p>Error generating map.</p>"


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
    with gr.Blocks(
        title=f"{BRAND_NAME} | {TEAM_NAME}",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="yellow",
            secondary_hue="blue",
            neutral_hue="slate",
            font=["Inter", "sans-serif"],
        ),
    ) as demo:

        # Header
        gr.HTML(f"""
        <div class="brand-header">
            <h1>🏔️ {BRAND_NAME}</h1>
            <p>{BRAND_HEADER}</p>
            <p style="color: #666; font-size: 0.8em; margin-top: 4px;">
                IOCG · Copper-Gold · Mount Woods / Prominent Hill · Open Australian Geodata
            </p>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ── Tab 1: Chat ────────────────────────────────────
            with gr.Tab("💬 Geological Chat"):
                gr.Markdown("""
**Ask JR MineralForge anything about:**
- IOCG mineral systems and Gawler Craton geology
- OZ Minerals Explorer Challenge winners and how Team JR improves upon them
- SARIG/GA open data and what features to target
- Specific drill targets: *"Show top copper-gold targets in Mount Woods"*
                """)
                chatbot = gr.Chatbot(label="JR MineralForge Assistant", height=450)
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder='e.g. "How did Team Guru win and how does JR MineralForge improve upon their strategy?"',
                        label="Your question",
                        lines=2,
                        scale=5,
                    )
                    send_btn = gr.Button("Send →", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        "Show top copper-gold targets in Mount Woods from SARIG data",
                        "How did Team Guru win and how does JR MineralForge improve it with GA geophysics?",
                        "What geophysical features indicate IOCG mineralisation in the Gawler Craton?",
                        "Compare DeepSightX strategy to Team JR approach for prospectivity mapping",
                        "What radiometric proxies should I use for potassic alteration targeting?",
                        "List all OZ Minerals Explorer Challenge winners",
                    ],
                    inputs=chat_input,
                )
                send_btn.click(chat_with_jr, [chat_input, chatbot], [chat_input, chatbot])
                chat_input.submit(chat_with_jr, [chat_input, chatbot], [chat_input, chatbot])

            # ── Tab 2: Data Ingestion ─────────────────────────
            with gr.Tab("📥 Data Ingestion"):
                gr.Markdown("""
### Automated Open Data Download
Downloads from **SARIG**, **Geoscience Australia**, and **State Geological Surveys** across Australia.
                """)
                
                with gr.Row():
                    province_select = gr.Dropdown(
                        choices=["mount_woods", "yilgarn_craton", "mt_isa_inlier", "lachlan_fold_belt", "pilbara_craton", "pine_creek"],
                        value="mount_woods",
                        label="Discovery Province / Region"
                    )

                with gr.Row():
                    check_btn = gr.Button("🔍 Check Available Data", variant="secondary")
                    tasmania_btn = gr.Button("🇦🇺 Ingest MRT (Tasmania)", variant="primary")
                    ga_btn = gr.Button("🌏 Ingest GA Data (National)", variant="primary")

                with gr.Accordion("Advanced: State Survey WFS", open=False):
                    with gr.Row():
                        state_code = gr.Dropdown(choices=["WA", "QLD", "NSW"], label="State", value="WA")
                        layer_input = gr.Textbox(label="WFS Layer Name", placeholder="e.g. Geology:SurfaceGeology")
                        state_btn = gr.Button("Fetch State Data")

                ingest_output = gr.Textbox(
                    label="Ingestion Status", lines=20, max_lines=30
                )

                check_btn.click(run_check_data, outputs=ingest_output)
                tasmania_btn.click(run_tasmania_ingest, inputs=province_select, outputs=ingest_output)
                ga_btn.click(run_ga_ingest, inputs=province_select, outputs=ingest_output)
                state_btn.click(run_state_ingest, inputs=[state_code, layer_input, province_select], outputs=ingest_output)

            # ── Tab 3: Prospectivity ──────────────────────────
            with gr.Tab("🗺️ Prospectivity Map"):
                gr.Markdown("""
### Mineral Prospectivity Pipeline
Generates ranked IOCG drill targets using ensemble ML with national awareness.
                """)
                
                with gr.Row():
                    province_map_select = gr.Dropdown(
                        choices=["mount_woods", "yilgarn_craton", "mt_isa_inlier", "lachlan_fold_belt", "pilbara_craton", "pine_creek"],
                        value="mount_woods",
                        label="Discovery Province"
                    )
                    run_sweep_btn = gr.Button("🚀 Run National Sweep (Experimental)", variant="secondary")

                run_btn = gr.Button("▶ Run Full Prospectivity Pipeline", variant="primary")
                pipeline_status = gr.Textbox(label="Pipeline Status", lines=10)
                map_display = gr.HTML(label="Interactive Prospectivity Map")

                run_btn.click(run_prospectivity, inputs=province_map_select, outputs=[pipeline_status, map_display])
                run_sweep_btn.click(run_national_sweep, outputs=pipeline_status)

            # ── Tab 4: Winners Analysis ───────────────────────
            with gr.Tab("🏆 Winner Analysis"):
                gr.Markdown("""
### OZ Minerals Explorer Challenge 2019 – Winner Strategy Analysis
Understand how Team JR's JR MineralForge improves upon each prize-winning strategy.
                """)

                list_btn = gr.Button("📋 List All Winners", variant="secondary")
                winners_display = gr.Textbox(label="Winners Overview", lines=15, max_lines=20)
                list_btn.click(list_winners, outputs=winners_display)

                gr.Markdown("---\n**Analyse a Specific Winner:**")
                with gr.Row():
                    winner_input = gr.Dropdown(
                        choices=["Team Guru", "DeepSightX", "deCODES", "SRK Consulting", "OreFox"],
                        label="Select Team",
                        value="Team Guru",
                    )
                    analyse_btn = gr.Button("Analyse & Compare", variant="primary")

                winner_analysis = gr.Textbox(label="Analysis & JR Improvements", lines=20, max_lines=30)
                analyse_btn.click(analyse_winner, inputs=winner_input, outputs=winner_analysis)

            # ── Tab 5: Reports ────────────────────────────────
            with gr.Tab("📄 Reports"):
                gr.Markdown("""
### JR MineralForge Reporting
Generate branded reports, SHAP importance plots, and target confidence charts.
All outputs are labelled: **Produced by JR MineralForge for Team JR**.
                """)

                with gr.Row():
                    report_btn = gr.Button("📊 Generate Full Report", variant="primary")
                    kb_btn = gr.Button("🔄 Rebuild Knowledge Base", variant="secondary")

                report_preview = gr.Textbox(label="Report Preview", lines=20, max_lines=30)
                report_file = gr.File(label="Download Report")
                report_btn.click(generate_report, outputs=[report_preview, report_file])
                kb_btn.click(rebuild_kb, outputs=report_preview)

        # Footer
        gr.HTML(f"""
        <div style="text-align: center; padding: 16px; color: #555;
                    font-size: 0.85em; border-top: 1px solid #2d3748; margin-top: 20px;">
            {BRAND_FOOTER} |
            Open Data: <a href="https://map.sarig.sa.gov.au" target="_blank"
               style="color: #4a9eff;">SARIG</a> ·
            <a href="https://portal.ga.gov.au" target="_blank"
               style="color: #4a9eff;">Geoscience Australia</a>
        </div>
        """)

    return demo


# ─── Launch ──────────────────────────────────────────────────────

def main():
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=None,
        show_error=True,
    )


if __name__ == "__main__":
    main()
