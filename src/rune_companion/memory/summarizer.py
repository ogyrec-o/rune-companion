# memory_summarizer.py

from __future__ import annotations

import logging
from typing import Any

import openai

from ..config import get_settings
from ..llm.client import stream_chat_chunks
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

    s = get_settings()
    max_msg_chars = int(getattr(s, "memory_summarizer_max_msg_chars", 600))
    max_summary_chars = int(getattr(s, "memory_summarizer_max_summary_chars", 800))

    trimmed: list[dict[str, str]] = []
    for m in dialog_messages:
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        if len(content) > max_msg_chars:
            content = content[:max_msg_chars] + "…"
        trimmed.append({"role": role, "content": content})

    raw = ""
    try:
        for piece in stream_chat_chunks(trimmed, SUMMARY_SYSTEM_PROMPT):
            raw += piece
    except openai.RateLimitError as e:
        logger.warning("Summarizer rate limit: %s", e)
        return None
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
