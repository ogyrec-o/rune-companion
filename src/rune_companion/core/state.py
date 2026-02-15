# src/rune_companion/core/state.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .ports import ChatMessage, LLMClient, MemoryRepo, TaskRepo, TTSEngine


@dataclass
class AppState:
    settings: Any

    llm: LLMClient
    memory: MemoryRepo
    task_store: TaskRepo
    tts_engine: TTSEngine

    tts_enabled: bool
    save_history: bool

    conversation: List[ChatMessage] = field(default_factory=list)
    dialog_histories: Dict[str, List[ChatMessage]] = field(default_factory=dict)

    # (kept in state for potential future use; currently chat.py uses module-level dicts)
    episode_counters: Dict[str, int] = field(default_factory=dict)
    memory_ctrl_counters: Dict[str, int] = field(default_factory=dict)
