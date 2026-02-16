# src/rune_companion/core/state.py

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .ports import ChatMessage, LLMClient, MemoryRepo, TaskRepo, TTSEngine


@dataclass(slots=True)
class AppState:
    settings: Any

    llm: LLMClient
    memory: MemoryRepo
    task_store: TaskRepo
    tts_engine: TTSEngine

    tts_enabled: bool
    save_history: bool

    # Used by connectors to serialize access when running multi-threaded (console + Matrix).
    lock: RLock = field(default_factory=RLock)

    conversation: list[ChatMessage] = field(default_factory=list)
    dialog_histories: dict[str, list[ChatMessage]] = field(default_factory=dict)

    episode_counters: dict[str, int] = field(default_factory=dict)
    memory_ctrl_counters: dict[str, int] = field(default_factory=dict)
