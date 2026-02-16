# src/rune_companion/cli/bootstrap.py

"""
CLI bootstrap helpers.

This module is the "composition root":
- loads settings once,
- ensures local (gitignored) directories exist,
- wires concrete implementations into AppState (LLM/memory/tasks/TTS),
- persists per-dialog histories as JSON (optional).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Literal, cast

from ..config import get_settings
from ..core.ports import ChatMessage, LLMClient
from ..core.state import AppState
from ..llm.client import OpenRouterLLMClient
from ..llm.offline import OfflineLLMClient
from ..memory.store import MemoryStore
from ..tasks.task_store import TaskStore
from ..tts.engine import TTSEngine

logger = logging.getLogger(__name__)


def _ensure_local_dirs(settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.matrix_store_path.mkdir(parents=True, exist_ok=True)
    settings.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.tasks_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.dialog_history_path.parent.mkdir(parents=True, exist_ok=True)


def create_initial_state(*, settings=None) -> AppState:
    """
    Create AppState from the provided settings.

    Keeping settings injectable makes the app easier to test and avoids hidden global config reads.
    If settings is None, falls back to get_settings().
    """
    if settings is None:
        settings = get_settings()

    _ensure_local_dirs(settings)

    llm_client: LLMClient
    try:
        llm_client = OpenRouterLLMClient(settings)
    except Exception:
        # Fallback for demos / local runs without external services.
        llm_client = OfflineLLMClient()

    state = AppState(
        settings=settings,
        llm=llm_client,
        tts_engine=TTSEngine(enabled=settings.tts_mode, settings=settings),
        memory=MemoryStore(settings.memory_db_path),
        task_store=TaskStore(settings.tasks_db_path),
        tts_enabled=settings.tts_mode,
        save_history=settings.save_history,
    )
    return state


def load_dialog_histories(state: AppState) -> dict[str, list[ChatMessage]]:
    if not state.save_history:
        return {}
    raw_path = getattr(state.settings, "dialog_history_path", None)
    if not raw_path:
        return {}
    path = Path(raw_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text("utf-8"))
        if not isinstance(data, dict):
            return {}
        out: dict[str, list[ChatMessage]] = {}
        for key, msgs in data.items():
            if not isinstance(key, str) or not isinstance(msgs, list):
                continue
            clean: list[ChatMessage] = []
            for m in msgs:
                if isinstance(m, dict):
                    role_any = m.get("role", "user")
                    role_s = role_any if isinstance(role_any, str) else "user"
                    if role_s not in ("system", "user", "assistant"):
                        role_s = "user"
                    role = cast(Literal["system", "user", "assistant"], role_s)

                    clean.append({"role": role, "content": str(m.get("content", ""))})
            if clean:
                out[key] = clean
        logger.info("Loaded dialog histories: %d dialogs from %s", len(out), path)
        return out
    except Exception:
        logger.exception("Failed to load dialog histories from %s", path)
        return {}


def save_dialog_histories(state: AppState) -> None:
    if not state.save_history:
        return
    raw_path = getattr(state.settings, "dialog_history_path", None)
    if not raw_path:
        return
    path = Path(raw_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.dialog_histories, ensure_ascii=False, indent=2), "utf-8")
        os.replace(tmp, path)
        with contextlib.suppress(Exception):
            # Best-effort: history may contain sensitive content, keep the file private on disk.
            os.chmod(path, 0o600)
        logger.info("Saved dialog histories: %d dialogs to %s", len(state.dialog_histories), path)
    except Exception:
        logger.exception("Failed to save dialog histories to %s", path)
