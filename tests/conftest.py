# tests/conftest.py

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rune_companion.core.state import AppState
from rune_companion.memory.store import MemoryStore
from rune_companion.tasks.task_store import TaskStore
from rune_companion.tts.engine import TTSEngine

from .fakes import FakeLLMClient


@pytest.fixture()
def settings(tmp_path: Path) -> SimpleNamespace:
    """
    Minimal settings object compatible with AppState and core modules.

    We intentionally use a SimpleNamespace rather than importing real config,
    to keep unit tests isolated and deterministic.
    """
    return SimpleNamespace(
        # Paths (tmp per test run)
        memory_db_path=tmp_path / "memory.sqlite3",
        tasks_db_path=tmp_path / "tasks.sqlite3",
        dialog_history_path=tmp_path / "dialog_histories.json",
        # Limits used by memory/api.py
        memory_max_user=50,
        memory_max_room=50,
        memory_max_rel=50,
        memory_max_global=50,
        memory_prompt_limit_user=10,
        memory_prompt_limit_room=10,
        memory_prompt_limit_rel=10,
        memory_prompt_limit_global=10,
        memory_prompt_limit_global_userstories=10,
        # Features
        save_history=True,
        tts_mode=False,
    )


@pytest.fixture()
def state(settings: SimpleNamespace) -> AppState:
    """
    AppState wired with deterministic fakes.

    NOTE: We keep real SQLite stores here (MemoryStore/TaskStore) because
    their correctness is part of what we want to test.
    """
    return AppState(
        settings=settings,
        llm=FakeLLMClient(),
        tts_engine=TTSEngine(enabled=False),
        memory=MemoryStore(settings.memory_db_path),
        task_store=TaskStore(settings.tasks_db_path),
        tts_enabled=False,
        save_history=True,
    )
