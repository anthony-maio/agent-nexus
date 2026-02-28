"""LangChain model adapters for the LangGraph orchestrator.

Creates :class:`~langchain_openai.ChatOpenAI` instances that route through
OpenRouter, giving LangGraph nodes native tool-calling and structured-output
support while keeping the same API key and billing as the rest of Nexus.

The existing :class:`~nexus.models.openrouter.OpenRouterClient` (aiohttp)
continues to serve swarm conversation, crosstalk, and consensus -- these
adapters are used exclusively by LangGraph nodes.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/agent-nexus",
    "X-Title": "Agent Nexus",
}


def create_orchestrator_llm(
    api_key: str,
    model: str = "google/gemini-2.5-flash",
) -> ChatOpenAI:
    """Create the dedicated orchestrator LLM via OpenRouter.

    Args:
        api_key: OpenRouter API key.
        model: Model identifier (override via ``ORCHESTRATOR_MODEL`` config).

    Returns:
        A ``ChatOpenAI`` instance configured for structured decision-making.
    """
    log.info("Creating orchestrator LLM: %s", model)
    return ChatOpenAI(
        base_url=_OPENROUTER_BASE_URL,
        api_key=api_key,
        model=model,
        temperature=0.3,
        max_tokens=2048,
        default_headers=_DEFAULT_HEADERS,
    )


def create_agent_llm(
    api_key: str,
    model: str = "z-ai/glm-4.7-flash",
) -> ChatOpenAI:
    """Create a tool-enabled task agent LLM via OpenRouter.

    Args:
        api_key: OpenRouter API key.
        model: Model identifier (override via ``TASK_AGENT_MODEL`` config).

    Returns:
        A ``ChatOpenAI`` instance configured for tool calling.
    """
    log.info("Creating agent LLM: %s", model)
    return ChatOpenAI(
        base_url=_OPENROUTER_BASE_URL,
        api_key=api_key,
        model=model,
        temperature=0.3,
        max_tokens=4096,
        default_headers=_DEFAULT_HEADERS,
    )
