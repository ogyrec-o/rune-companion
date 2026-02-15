# src/rune_companion/cli/bootstrap.py

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from ..config import get_settings
from ..core.state import AppState
from ..llm.client import OpenRouterLLMClient
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
    Create AppState from provided settings.
    If settings is None, falls back to get_settings().
    """
    if settings is None:
        settings = get_settings()

    _ensure_local_dirs(settings)

    state = AppState(
        settings=settings,
        llm=OpenRouterLLMClient(settings),
        tts_engine=TTSEngine(enabled=settings.tts_mode, settings=settings),
        memory=MemoryStore(settings.memory_db_path),
        task_store=TaskStore(settings.tasks_db_path),
        tts_enabled=settings.tts_mode,
        save_history=settings.save_history,
    )
    return state

def load_dialog_histories(state: AppState) -> Dict[str, List[dict[str, str]]]:
    if not state.save_history:
        return {}
    path: Path = state.settings.dialog_history_path  # type: ignore[attr-defined]
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text("utf-8"))
        if not isinstance(data, dict):
            return {}
        out: Dict[str, List[dict[str, str]]] = {}
        for key, msgs in data.items():
            if not isinstance(key, str) or not isinstance(msgs, list):
                continue
            clean = []
            for m in msgs:
                if isinstance(m, dict):
                    clean.append({"role": str(m.get("role", "user")), "content": str(m.get("content", ""))})
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
    path: Path = state.settings.dialog_history_path  # type: ignore[attr-defined]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.dialog_histories, ensure_ascii=False, indent=2), "utf-8")
        os.replace(tmp, path)
        with contextlib.suppress(Exception):
            os.chmod(path, 0o600)
        logger.info("Saved dialog histories: %d dialogs to %s", len(state.dialog_histories), path)
    except Exception:
        logger.exception("Failed to save dialog histories to %s", path)
