"""
JR MineralForge – Commercial Drill-or-Drop Orchestrator
===========================================================
Re-architected Multi-Agent Supervisor: Retains agent capabilities but enforces
a strict, deterministic execution pipeline to guarantee stable commercial mapping.
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

# Self-Healing Pre-flight Setup
from utils.dependency_manager import check_and_install_dependencies
check_and_install_dependencies()

from langchain_core.messages import HumanMessage
from config.settings import BRAND_NAME, TEAM_NAME, BRAND_HEADER, BRAND_FOOTER
from utils.logging_utils import get_logger
from utils.llm_factory import get_llm

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────
# 5-Component Supervised Flow
# ─────────────────────────────────────────────────────────────────

class RigidMultiAgentSupervisor:
    """Supervisor Agent that strictly forces 5 discrete pipeline components to execute."""

    def __init__(self, llm=None, verbose: bool = True):
        self.llm = llm or get_llm()
        self.verbose = verbose
        log.info(f"{BRAND_NAME} Commercial Supervisor Agent online.")

    def run_nli_extraction(self, query: str) -> dict:
        """
        Agent 1: NLI Processor.
        Takes business string -> outputs strict JSON execution limits.
        """
        log.info("Agent 1 [NLI Supervisor] parsing constraints...")
        prompt = (
            f"You are a geospatial parameter extraction agent. "
            f"Extract variables from this mining request: '{query}'. "
            f"Output ONLY valid JSON in format: "
            f'{{"province_id": "mount_woods", "commodity": "Cu-Au", "weights": {{"tmi": 1.0, "gravity": 1.0, "k": 1.0}}}} '
            f"(Do not use markdown blocks, just raw JSON)."
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, 'content') else str(response)
        
        try:
            # Clean possible markdown format
            cleaned = content.replace("```json", "").replace("```", "").strip()
            params = json.loads(cleaned)
            return params
        except json.JSONDecodeError:
            log.warning("Agent 1 failed parsing, defaulting to Mount Woods Base.")
            return {"province_id": "mount_woods", "commodity": "Cu-Au", "weights": {}}

    def fetch_data_layers(self, params: dict):
        """Agent 2: Dynamic Data Ingestion from SARIG/GA."""
        log.info("Agent 2 [Ingestion Worker] connecting to WFS nodes...")
        from agents.data_ingestion_agent import SARIGIngestionAgent
        from config.settings import AUSTRALIAN_PROVINCES
        
        bbox = AUSTRALIAN_PROVINCES.get(params.get("province_id", "mount_woods"), AUSTRALIAN_PROVINCES["mount_woods"])
        agent = SARIGIngestionAgent(bbox=bbox)
        return agent.ingest_all()

    def run_ml_pipeline(self, params: dict):
        """Agent 3 & 4: Isolation Forest & Explainer Target Ranking."""
        log.info("Agents 3 & 4 [ML Engine & Ranker] generating targets...")
        from agents.prospectivity_agent import ProspectivityMappingAgent
        from config.settings import AUSTRALIAN_PROVINCES
        
        bbox = AUSTRALIAN_PROVINCES.get(params.get("province_id", "mount_woods"), AUSTRALIAN_PROVINCES["mount_woods"])
        agent = ProspectivityMappingAgent(bbox=bbox)
        return agent.run_full_pipeline()

    def commercial_run(self, query: str) -> dict:
        """Strict Supervisor Loop enforcing commercial validity."""
        print(f"\n{BRAND_HEADER}\nTriggering Multi-Agent Drill-or-Drop Pipeline...\n")
        
        # 1. NLI
        params = self.run_nli_extraction(query)
        print(f"✅ NLI Processor mapped parameters: {params['commodity']} targeting in {params['province_id']}")

        # 2. Ingestion
        try:
            data = self.fetch_data_layers(params)
            print(f"✅ OGC Ingestion Complete: {len(data)} sub-systems downloaded via bounding box.")
        except Exception as e:
            print(f"❌ Ingestion Failed: {e}")
            return {}

        # 3 & 4. Compute & Rank
        try:
            results = self.run_ml_pipeline(params)
            print(f"✅ ML Target Grid extracted! Generating Top 10 Anomalies...")
        except Exception as e:
            print(f"❌ Compute Failed: {e}")
            return {}
            
        print(f"\n✅ Supervisor execution sequence successful. System outputs prepared.")
        return results


def main():
    parser = argparse.ArgumentParser(description=f"{BRAND_NAME} Supervisor MultiAgent Flow")
    parser.add_argument("query", type=str, nargs='?', default="Find top copper-gold targets in Mount Woods", help="NLI Query")
    
    args = parser.parse_args()
    supervisor = RigidMultiAgentSupervisor()
    supervisor.commercial_run(args.query)


if __name__ == "__main__":
    main()

# Alias for app.py compatibility
JRMineralForgeOrchestrator = RigidMultiAgentSupervisor
