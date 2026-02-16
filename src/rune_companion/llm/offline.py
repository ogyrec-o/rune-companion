# src/rune_companion/llm/offline.py

from __future__ import annotations

from collections.abc import Iterable

from ..core.ports import ChatMessage


class OfflineLLMClient:
    """
    Offline deterministic LLM client used for demos when no external API is configured.

    Behavior:
    - Memory controller prompts -> returns {"ops": []}
    - Summarizer prompts -> returns "No significant facts."
    - Normal chat -> returns a friendly offline demo response
    """

    def stream_chat(self, messages: list[ChatMessage], system_prompt: str) -> Iterable[str]:
        sp = (system_prompt or "").lower()

        # Planner must output JSON that the controller can parse safely.
        if "memory/task planner" in sp or "memory controller" in sp or "<plan_json>" in sp:
            yield '{"ops":[]}'
            return

        # Summarizer has a strict fallback phrase.
        if "summarization module" in sp or "episodic summarizer" in sp:
            yield "No significant facts."
            return

        # Normal chat: reflect user message, no external calls.
        user_text = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_text = m["content"]
                break

        yield (
            "Offline demo mode: no external LLM is configured.\n"
            "Set RUNE_OPENROUTER_API_KEY (and RUNE_LLM_MODELS) to enable real responses.\n\n"
            f"You said: {user_text}"
        )
