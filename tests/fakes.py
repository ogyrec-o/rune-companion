# tests/fakes.py

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from rune_companion.core.ports import ChatMessage, OutboundMessenger


class FakeLLMClient:
    """
    Deterministic LLM client for unit tests.

    - Captures calls for assertions
    - Yields a predefined text as a single chunk
    """

    def __init__(self, next_text: str = "ok") -> None:
        self.next_text = next_text
        self.calls: list[tuple[list[ChatMessage], str]] = []

    def stream_chat(self, messages: list[ChatMessage], system_prompt: str) -> Iterable[str]:
        self.calls.append((messages, system_prompt))
        yield self.next_text


@dataclass(slots=True)
class SentMessage:
    text: str
    room_id: str | None
    to_user_id: str | None


@dataclass(slots=True)
class FakeMessenger(OutboundMessenger):
    """
    Fake OutboundMessenger used by scheduler tests.
    """

    sent: list[SentMessage] = field(default_factory=list)

    async def send_text(
        self,
        *,
        text: str,
        room_id: str | None = None,
        to_user_id: str | None = None,
    ) -> None:
        self.sent.append(SentMessage(text=text, room_id=room_id, to_user_id=to_user_id))
