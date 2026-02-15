# state.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from config import get_settings
from memory_store import MemoryStore
from tasks.task_store import TaskStore
from tts_engine import TTSEngine


@dataclass
class AppState:
    # Store Settings on the state for easy access in other modules later.
    settings: object

    tts_enabled: bool
    save_history: bool
    tts_engine: TTSEngine
    memory: MemoryStore
    task_store: TaskStore

    conversation: List[Dict[str, str]] = field(default_factory=list)
    dialog_histories: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)


def _ensure_local_dirs(settings) -> None:
    # Everything should live under gitignored local paths.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.matrix_store_path.mkdir(parents=True, exist_ok=True)

    settings.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.tasks_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.dialog_history_path.parent.mkdir(parents=True, exist_ok=True)


def create_initial_state() -> AppState:
    """Create initial application state (no secrets required)."""
    settings = get_settings()

    _ensure_local_dirs(settings)

    tts = TTSEngine(enabled=settings.tts_mode)

    memory = MemoryStore(settings.memory_db_path)
    task_store = TaskStore(settings.tasks_db_path)

    state = AppState(
        settings=settings,
        tts_enabled=settings.tts_mode,
        save_history=settings.save_history,
        tts_engine=tts,
        memory=memory,
        task_store=task_store,
    )

    return state


def load_dialog_histories(state: AppState) -> Dict[str, List[Dict[str, str]]]:
    """Load per-user/room dialog histories from JSON (best-effort)."""
    if not state.save_history:
        return {}

    settings = state.settings
    path: Path = settings.dialog_history_path  # type: ignore[attr-defined]

    if not path.exists():
        return {}

    try:
        raw = path.read_text("utf-8")
        data = json.loads(raw)
        out: Dict[str, List[Dict[str, str]]] = {}

        if isinstance(data, dict):
            for key, msgs in data.items():
                if not isinstance(key, str) or not isinstance(msgs, list):
                    continue

                clean: List[Dict[str, str]] = []
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    role = str(m.get("role", "user"))
                    content = str(m.get("content", ""))
                    clean.append({"role": role, "content": content})

                if clean:
                    out[key] = clean

        print(f"[STATE] Loaded dialog histories: {len(out)} dialogs from {path}")
        return out
    except Exception as e:
        print("[STATE][WARN] Failed to load dialog histories:", e)
        return {}


def save_dialog_histories(state: AppState) -> None:
    """Save per-user/room dialog histories to JSON (best-effort)."""
    if not state.save_history:
        return

    settings = state.settings
    path: Path = settings.dialog_history_path  # type: ignore[attr-defined]

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        json_str = json.dumps(state.dialog_histories, ensure_ascii=False, indent=2)

        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json_str, "utf-8")
        os.replace(tmp_path, path)

        # Best-effort: make it private (contains PII)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass

        print(f"[STATE] Saved dialog histories: {len(state.dialog_histories)} dialogs to {path}")
    except Exception as e:
        print("[STATE][WARN] Failed to save dialog histories:", e)
