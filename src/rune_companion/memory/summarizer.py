# src/rune_companion/memory/summarizer.py

from __future__ import annotations

"""
Episodic summarizer.

Produces short third-person notes that can be stored as long-term memory items.
Used to keep context compact when dialog history grows.
"""

import logging
from typing import Any

from ..core.state import AppState

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """
You are a summarization module that produces short notes for long-term memory.

Input: a short dialog excerpt between:
- user (role="user")
- assistant (role="assistant")

Task:
- Summarize 1–3 key topics and any stable, useful facts about the user
  (goals, plans, preferences, constraints, recurring themes).

Rules:
- Write in third person ("The user ...", "They discussed ...").
- Do not address the user directly ("you").
- No roleplay, no emojis.
- 1–4 sentences.
- If there is nothing worth remembering, reply exactly:
  No significant facts.
""".strip()


def summarize_dialog_chunk(state: AppState, dialog_messages: list[dict[str, Any]]) -> str | None:
    """
    Compress a dialog chunk into a short text note for memory storage.

    dialog_messages: list of {"role": "user"/"assistant", "content": "..."}.
    Returns: summary string or None on failure.
    """
    if not dialog_messages:
        return None

    s = getattr(state, "settings", None)
    max_msg_chars = int(getattr(s, "memory_summarizer_max_msg_chars", 600))
    max_summary_chars = int(getattr(s, "memory_summarizer_max_summary_chars", 800))

    trimmed: list[dict[str, str]] = []
    for m in dialog_messages:
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        # Guardrail: keep summarizer prompt bounded even if a single message is huge.
        if len(content) > max_msg_chars:
            content = content[:max_msg_chars] + "…"
        trimmed.append({"role": role, "content": content})

    raw = ""
    try:
        for piece in state.llm.stream_chat(trimmed, SUMMARY_SYSTEM_PROMPT):
            raw += piece
    except Exception:
        logger.exception("Summarizer failed.")
        return None

    summary = raw.strip()
    if not summary:
        return None

    if len(summary) > max_summary_chars:
        summary = summary[:max_summary_chars] + "…"

    logger.debug("Episode summary produced len=%d", len(summary))
    return summary
