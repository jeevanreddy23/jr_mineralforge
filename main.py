"""
JR MineralForge – Main Orchestrator (Python 3.14 Compatible)
===========================================================
Wires together all agents into a robust tool-calling system.
Avoids legacy langchain.agents for compatibility with newer Python versions.
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from config.settings import BRAND_NAME, TEAM_NAME, BRAND_HEADER, BRAND_FOOTER
from utils.logging_utils import get_logger
from utils.llm_factory import get_llm

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# All Tools Registration
# ─────────────────────────────────────────────────────────────────

def _load_all_tools() -> list:
    """Import and return all LangChain tools from all agents."""
    tools = []

    # Data Ingestion tools
    from agents.data_ingestion_agent import (
        run_sarig_ingestion, run_ga_ingestion, check_data_availability,
        run_tasmania_ingestion
    )
    tools.extend([run_sarig_ingestion, run_ga_ingestion, check_data_availability, 
                 run_tasmania_ingestion])

    # Knowledge / RAG tools
    from agents.knowledge_agent import (
        search_geological_knowledge, analyse_winner_strategy,
        list_all_winners, rebuild_knowledge_base
    )
    tools.extend([search_geological_knowledge, analyse_winner_strategy,
                  list_all_winners, rebuild_knowledge_base])

    # Reporting tools
    from agents.reporting_agent import generate_full_report, explain_target
    tools.extend([generate_full_report, explain_target])

    # Prospectivity / Mapping tools
    tools.append(run_prospectivity_pipeline)
    tools.append(run_national_sweep)

    return tools


@tool
def run_prospectivity_pipeline(province_id: str = "mount_woods") -> str:
    """
    Run the full JR MineralForge prospectivity mapping pipeline for a specific province.
    Assembles the feature datacube, runs ML predictions, generates rasters,
    extracts ranked targets, and creates the interactive folium map.
    province_id: one of ['mount_woods', 'yilgarn_craton', 'mt_isa_inlier', 'lachlan_fold_belt', 'pilbara_craton', 'pine_creek']
    """
    from agents.prospectivity_agent import ProspectivityMappingAgent
    from config.settings import AUSTRALIAN_PROVINCES
    bbox = AUSTRALIAN_PROVINCES.get(province_id, AUSTRALIAN_PROVINCES["mount_woods"])
    agent = ProspectivityMappingAgent(bbox=bbox)
    results = agent.run_full_pipeline()
    return f"Prospectivity pipeline complete for {bbox.name}!\n{json.dumps(results, indent=2)}"


@tool
def run_national_sweep() -> str:
    """
    Experimental: Sweep multiple major Australian mineral provinces for IOCG/Cu-Au potential.
    Iterates through known high-potential cratons and runs target extraction.
    """
    from config.settings import AUSTRALIAN_PROVINCES
    summary = []
    for pid in ["yilgarn_craton", "mt_isa_inlier", "lachlan_fold_belt"]:
        res = run_prospectivity_pipeline.invoke({"province_id": pid})
        summary.append(res)
    return "\n\n".join(summary)


@tool
def run_state_ingestion(state_code: str, layer: str, province_id: str = "mount_woods") -> str:
    """
    Fetch specific state-level geological data (WA, QLD, NSW).
    state_code: 'WA', 'QLD', or 'NSW'
    layer: WFS layer name
    """
    from agents.data_ingestion_agent import StateSurveyIngestionAgent
    from config.settings import AUSTRALIAN_PROVINCES
    bbox = AUSTRALIAN_PROVINCES.get(province_id, AUSTRALIAN_PROVINCES["mount_woods"])
    agent = StateSurveyIngestionAgent(bbox=bbox)
    res = agent.ingest_from_portal(state_code, layer)
    return str(res)


# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

class JRMineralForgeOrchestrator:
    """Main multi-agent orchestrator for JR MineralForge."""

    def __init__(self, llm=None, verbose: bool = True):
        self.llm = llm or get_llm()
        self.tools = _load_all_tools()
        self.tool_map = {t.name: t for t in self.tools}
        self.verbose = verbose
        
        # Bind tools to LLM if supported
        if hasattr(self.llm, "bind_tools"):
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm
            
        log.info(f"{BRAND_NAME} orchestrator initialised with {len(self.tools)} tools")

    def run(self, query: str) -> str:
        """Execute a query through a tool-calling loop."""
        log.info(f"Query: {query[:100]}…")
        
        messages = [HumanMessage(content=query)]
        
        # Simple loop for max 5 iterations
        for i in range(5):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            if not response.tool_calls:
                return response.content
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                if tool_name not in self.tool_map:
                    observation = f"Error: Tool {tool_name} not found."
                else:
                    if self.verbose:
                        print(f"\n🛠️  Calling Tool: {tool_name} with {tool_args}")
                    try:
                        observation = self.tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        observation = f"Error executing tool: {e}"
                
                if self.verbose:
                    print(f"👁️  Observation: {str(observation)[:200]}…")
                    
                messages.append(ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"]
                ))
        
        return "I've completed my analysis. Unfortunately, I reached my iteration limit."

    def run_full_workflow(self) -> None:
        """Run the complete JR MineralForge workflow end-to-end."""
        print(f"\n{'='*70}")
        print(f"  🏔️  {BRAND_NAME} – Full Workflow")
        print(f"  {BRAND_HEADER}")
        print(f"{'='*70}\n")

        steps = [
            ("Checking data availability", "check_data_availability"),
            ("Ingesting SARIG open data", "run_sarig_ingestion"),
            ("Ingesting Geoscience Australia open data", "run_ga_ingestion"),
            ("Listing OZ Minerals Explorer Challenge winners", "list_all_winners"),
            ("Running prospectivity pipeline", "run_prospectivity_pipeline"),
            ("Generating full report", "generate_full_report"),
        ]

        for step_name, action in steps:
            print(f"\n📍 Step: {step_name}")
            print("-" * 50)
            result = self.run(step_name)
            print(result[:500] if len(result) > 500 else result)

        print(f"\n{'='*70}")
        print(f"  ✅ {BRAND_NAME} workflow complete!")
        print(f"  {BRAND_FOOTER}")
        print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=f"{BRAND_NAME} CLI – Team JR Mineral Discovery AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"\n{BRAND_FOOTER}",
    )
    parser.add_argument(
        "command",
        choices=["chat", "workflow", "ingest", "report", "rebuild-kb"],
        help="Command to run",
    )
    parser.add_argument("--query", "-q", type=str, default=None, help="Query for chat mode")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider (ollama/openai/anthropic)")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose agent output")

    args = parser.parse_args()

    # Print banner
    print(f"\n{'#'*70}")
    print(f"  {BRAND_NAME} for {TEAM_NAME}")
    print(f"  {BRAND_HEADER}")
    print(f"{'#'*70}\n")

    # Initialise LLM
    from config.settings import LLM_PROVIDER as default_provider, LLM_MODEL as default_model
    provider = args.provider or default_provider
    model = args.model or default_model

    try:
        llm = get_llm(provider=provider, model=model)
    except Exception as e:
        print(f"⚠️  LLM initialisation failed: {e}")
        llm = None

    orchestrator = JRMineralForgeOrchestrator(llm=llm, verbose=args.verbose)

    if args.command == "chat":
        if args.query:
            result = orchestrator.run(args.query)
            print(f"\n{result}")
        else:
            print("Interactive chat mode. Type 'exit' to quit.\n")
            while True:
                try:
                    query = input(f"[{TEAM_NAME}] → ").strip()
                    if query.lower() in ("exit", "quit", "q"):
                        break
                    if not query:
                        continue
                    result = orchestrator.run(query)
                    print(f"\n{result}\n")
                except KeyboardInterrupt:
                    break
            print(f"\n{BRAND_FOOTER}\n")

    elif args.command == "workflow":
        orchestrator.run_full_workflow()

    elif args.command == "ingest":
        print("Running SARIG + GA data ingestion…")
        for q in ["run_sarig_ingestion", "run_ga_ingestion"]:
            print(f"\n{orchestrator.run(q)}")

    elif args.command == "report":
        print("Generating full prospectivity report…")
        print(orchestrator.run("generate_full_report"))

    elif args.command == "rebuild-kb":
        print("Rebuilding knowledge base…")
        print(orchestrator.run("rebuild_knowledge_base"))


if __name__ == "__main__":
    main()
