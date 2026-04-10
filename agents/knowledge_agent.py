"""
JR MineralForge – Geological Knowledge & Winners Analysis Agent
================================================================
RAG-powered agent for:
  - Answering geological questions using the FAISS knowledge base
  - Comparing Team JR's approach vs OZ Minerals Challenge winners
  - Retrieving winner tactics and adapting them to current analysis
  - IOCG mineral systems knowledge retrieval
"""

from __future__ import annotations

import json
from typing import Optional, Any

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import (
    OZ_MINERALS_CHALLENGE_WINNERS,
    BRAND_NAME, TEAM_NAME, BRAND_HEADER, BRAND_FOOTER, WINNERS_CONTEXT,
)
from utils.logging_utils import get_logger
from rag.rag_setup import JRVectorStore, build_rag_chain

log = get_logger(__name__)

# Shared vector store instance (lazy initialised)
_vector_store: Optional[JRVectorStore] = None


def get_vector_store(force_rebuild: bool = False) -> JRVectorStore:
    global _vector_store
    if _vector_store is None or force_rebuild:
        _vector_store = JRVectorStore().load_or_create(force_rebuild=force_rebuild)
    return _vector_store


# ─────────────────────────────────────────────────────────────────
# LangChain Tools
# ─────────────────────────────────────────────────────────────────

@tool
def search_geological_knowledge(query: str) -> str:
    """
    Search the JR MineralForge geological RAG knowledge base.
    Retrieves relevant information about IOCG mineral systems,
    Gawler Craton geology, geophysical signatures, and targeting proxies.
    Anti-noise: only returns chunks above similarity score threshold.
    """
    vs = get_vector_store()
    results = vs.similarity_search_with_scores(query, k=5)
    if not results:
        return "No sufficiently relevant information found in the knowledge base for this query."

    lines = [f"[{BRAND_NAME}] Geological Knowledge Search: '{query}'\n"]
    for i, (doc, score) in enumerate(results, 1):
        lines.append(f"--- Result {i} (similarity: {score:.3f}) ---")
        lines.append(f"Source: {doc.metadata.get('source', 'unknown')}")
        if "title" in doc.metadata:
            lines.append(f"Title: {doc.metadata['title']}")
        lines.append(doc.page_content[:500] + ("…" if len(doc.page_content) > 500 else ""))
        lines.append("")
    return "\n".join(lines)


@tool
def analyse_winner_strategy(team_name: str) -> str:
    """
    Retrieve and analyse the strategy of a specific OZ Minerals Explorer Challenge winner.
    Then explain how Team JR's JR MineralForge improves upon their approach.
    team_name: e.g. 'Team Guru', 'DeepSightX', 'deCODES', 'SRK Consulting', 'OreFox'
    """
    winner = None
    for w in OZ_MINERALS_CHALLENGE_WINNERS:
        if team_name.lower() in w["team"].lower():
            winner = w
            break

    if winner is None:
        available = [w["team"] for w in OZ_MINERALS_CHALLENGE_WINNERS]
        return f"Winner '{team_name}' not found. Available: {available}"

    tactics = "\n".join(f"  • {t}" for t in winner["key_tactics"])
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║         {BRAND_NAME} – Winner Strategy Analysis           ║
║  {BRAND_HEADER[:60]}  ║
╚══════════════════════════════════════════════════════════════════╝

WINNER PROFILE:
  Team: {winner['team']}
  Prize: {winner['prize']} ({winner['year']})
  Challenge: {winner['challenge']}

KEY WINNING TACTICS:
{tactics}

HOW JR MINERALFORGE IMPROVES UPON {winner['team'].upper()}:
{WINNERS_CONTEXT}

Specific improvements by Team JR:
  1. OPEN DATA DEPTH: JR MineralForge ingests the full SARIG and GA open data
     catalogue (magnetics, gravity, radiometrics, drillholes, geochemistry)
     through automated pipelines, giving richer features than previous winners.

  2. ANTI-NOISE PIPELINE: Wavelet denoising + median filtering on geophysical
     grids, plus Isolation Forest outlier removal from training data, reduces
     noise that degraded earlier entries.

  3. SPATIAL CROSS-VALIDATION: Buffered block CV (10 km exclusion buffer)
     prevents the spatial leakage that inflated CV scores in 2019 entries.

  4. ENSEMBLE UNCERTAINTY: Bootstrap uncertainty maps (N=50 models) provide
     drill risk scores that {winner['team']} did not provide.

  5. LANGCHAIN RAG: Full retrieval-augmented answering over geological reports,
     open-access publications, and winner strategies — enabling continuous
     knowledge accumulation that no 2019 team employed.

  6. SHAP INTERPRETABILITY: Feature importance is SHAP-driven, providing
     geological explainability at each target (e.g., why a location has
     potassic alteration signature consistent with IOCG).

{BRAND_FOOTER}
"""
    return report


@tool
def list_all_winners() -> str:
    """List all OZ Minerals Explorer Challenge winners in the knowledge base with their key strategies."""
    lines = [
        f"[{BRAND_NAME}] OZ Minerals Explorer Challenge – Winner Summary\n",
        "=" * 60,
    ]
    for w in OZ_MINERALS_CHALLENGE_WINNERS:
        lines.append(f"\n🏆 {w['team']} – {w['prize']} ({w['year']})")
        for t in w["key_tactics"]:
            lines.append(f"   • {t}")
    lines.append(f"\n{BRAND_FOOTER}")
    return "\n".join(lines)


@tool
def rebuild_knowledge_base() -> str:
    """
    Rebuild the FAISS vector store index from scratch.
    Use this after adding new documents to the data directory.
    """
    log.info("Rebuilding knowledge base …")
    get_vector_store(force_rebuild=True)
    return "Knowledge base rebuilt successfully. New documents have been indexed."


# ─────────────────────────────────────────────────────────────────
# Geological Knowledge Agent (full RAG chain)
# ─────────────────────────────────────────────────────────────────

class GeologicalKnowledgeAgent:
    """
    Full RAG agent for geological queries using LangChain.
    Uses the FAISS vector store and an LLM backend.
    """

    def __init__(self, llm):
        self.llm = llm
        self.vector_store = get_vector_store()
        self.rag_chain = build_rag_chain(llm, self.vector_store)

    def ask(self, question: str) -> str:
        """Answer a geological question using RAG over the knowledge base."""
        log.info(f"RAG query: {question[:80]}…")
        result = self.rag_chain({"query": question})
        answer = result.get("result", "No answer generated")
        sources = result.get("source_documents", [])

        if sources:
            source_strs = []
            for doc in sources:
                src = doc.metadata.get("source", "unknown")
                title = doc.metadata.get("title", doc.metadata.get("team", ""))
                source_strs.append(f"  • {src}" + (f" – {title}" if title else ""))
            answer += "\n\nSources:\n" + "\n".join(source_strs)

        return answer

    def compare_with_winners(self, query: str) -> str:
        """Query focused on winner comparison and Team JR improvements."""
        enhanced_query = (
            f"{query}\n\nPlease specifically address how Team JR's JR MineralForge "
            f"approach compares to and improves upon the OZ Minerals Explorer Challenge "
            f"winners (Team Guru, DeepSightX, deCODES, SRK Consulting) using SARIG "
            f"and Geoscience Australia open data."
        )
        return self.ask(enhanced_query)
