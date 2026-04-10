"""
JR MineralForge – LLM Factory
================================
Creates the appropriate LLM backend based on LLM_PROVIDER setting.
Supports: Ollama (local), OpenAI, Anthropic.
"""

from __future__ import annotations

from config.settings import (
    LLM_PROVIDER, LLM_MODEL, OLLAMA_BASE_URL, OPENAI_API_KEY, ANTHROPIC_API_KEY
)
from utils.logging_utils import get_logger

log = get_logger(__name__)


def get_llm(
    provider: str = LLM_PROVIDER,
    model: str = LLM_MODEL,
    temperature: float = 0.1,
):
    """
    Return a LangChain-compatible LLM instance.
    Priority: OLLAMA (local/free) → OpenAI → Anthropic.
    """
    provider = provider.lower()
    log.info(f"Initialising LLM: provider={provider}, model={model}")

    if provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(base_url=OLLAMA_BASE_URL, model=model, temperature=temperature)

    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=temperature,
                          api_key=OPENAI_API_KEY)

    elif provider in ("anthropic", "claude"):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-3-haiku-20240307",
                             temperature=temperature, api_key=ANTHROPIC_API_KEY)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'ollama', 'openai', or 'anthropic'")
