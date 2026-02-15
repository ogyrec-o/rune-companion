# src/rune_companion/core/ports.py

from __future__ import annotations

"""
Ports (interfaces) used by the core.

The core depends on Protocols instead of concrete implementations.
This keeps connectors/storage/LLM providers swappable and makes testing easier.
"""

from typing import Any, Awaitable, Iterable, Protocol

ChatMessage = dict[str, str]
# OpenAI-style chat messages: {"role": "...", "content": "..."}.


class LLMClient(Protocol):
    """Streaming chat completion client (OpenAI/OpenRouter-compatible)."""
    def stream_chat(self, messages: list[ChatMessage], system_prompt: str) -> Iterable[str]: ...


class OutboundMessenger(Protocol):
    """
    Connector-side port: how services (tasks scheduler) can send text outward.

    The connector decides how to interpret:
    - room_id (can be None)
    - to_user_id (can be None)
    E.g. Matrix connector may pick a default room if room_id is missing.
    """

    def send_text(
            self,
            *,
            text: str,
            room_id: str | None = None,
            to_user_id: str | None = None,
    ) -> Awaitable[None]: ...


class MemoryRepo(Protocol):
    def global_subject_id(self) -> str: ...

    def add_memory(
            self,
            *,
            subject_type: str,
            subject_id: str,
            text: str,
            importance: float,
            tags: list[str],
            source: str,
            person_ref: str | None,
    ) -> int: ...

    def prune_subject(self, subject_type: str, subject_id: str, max_items: int) -> None: ...

    def query_memory(
            self,
            *,
            subject_type: str,
            subject_id: str,
            limit: int,
            tag: str | None = None,
    ) -> list[Any]: ...

    def update_memory(
            self,
            mem_id: int,
            *,
            text: str | None,
            tags: list[str] | None,
            importance: float | None,
            person_ref: str | None,
    ) -> None: ...

    def delete_memory(self, mem_id: int) -> None: ...
    def count_memories(self) -> int: ...


class TaskRepo(Protocol):
    # Prompt injection / reply-capture API
    def list_open_tasks_for_user(self, user_id: str, limit: int = 16) -> list[Any]: ...
    def find_waiting_ask_task(self, *, to_user_id: str, room_id: str) -> Any | None: ...
    def save_answer_and_mark_received(
            self,
            task_id: int,
            answer_text: str,
            now_ts: float | None = None,
    ) -> None: ...

    # Scheduler API
    def list_runnable_tasks(self, *, now_ts: float, limit: int = 32) -> list[Any]: ...
    def try_claim_task(self, task_id: int, *, expected: Iterable[Any]) -> bool: ...
    def update_task_status(self, task_id: int, new_status: Any) -> None: ...
    def update_task_fields(
            self,
            task_id: int,
            *,
            status: Any | None = None,
            due_at: float | None = None,
            meta: dict[str, Any] | None = None,
            question_text: str | None = None,
            answer_text: str | None = None,
    ) -> None: ...

    # Task creation (memory planner)
    def add_task(
            self,
            *,
            kind: str,
            description: str,
            from_user_id: str | None = None,
            to_user_id: str | None = None,
            reply_to_user_id: str | None = None,
            room_id: str | None = None,
            due_at: float | None = None,
            importance: float = 0.7,
            meta: dict[str, Any] | None = None,
            status: Any = None,  # TaskStatus (kept as Any to avoid import coupling)
            question_text: str | None = None,
            answer_text: str | None = None,
    ) -> int: ...


class TTSEngine(Protocol):
    def speak_sentence(self, text: str) -> None: ...
    def wait_all(self) -> None: ...
    def shutdown(self) -> None: ...
