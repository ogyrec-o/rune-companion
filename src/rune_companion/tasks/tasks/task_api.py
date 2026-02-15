# tasks/task_api.py

from __future__ import annotations

import logging
import time

from state import AppState

from .task_store import TaskStore

logger = logging.getLogger(__name__)


def _default_tasks_db_path() -> str:
    """
    Place tasks DB next to the memory DB in the local gitignored data dir.

    Prefers Settings (get_settings). Falls back to legacy config.MEMORY_DB_PATH if needed.
    """
    try:
        from config import get_settings  # type: ignore

        s = get_settings()
        mem_db = getattr(s, "memory_db_path", None)
        if mem_db:
            return str(mem_db.with_name("tasks.sqlite3"))
    except Exception:
        pass

    try:
        from config import MEMORY_DB_PATH  # type: ignore

        return str(MEMORY_DB_PATH.with_name("tasks.sqlite3"))
    except Exception:
        return "tasks.sqlite3"


def ensure_task_store(state: AppState, db_path: str | None = None) -> TaskStore:
    """
    Lazy init TaskStore in AppState.
    If db_path is None -> place it next to the memory DB (in .local by default).
    """
    ts = getattr(state, "task_store", None)
    if ts is not None:
        return ts

    path = db_path or _default_tasks_db_path()
    ts = TaskStore(db_path=path)
    setattr(state, "task_store", ts)
    logger.info("TaskStore initialized db=%s", path)
    return ts


def schedule_simple_message(
        state: AppState,
        *,
        description: str,
        to_user_id: str | None,
        room_id: str | None,
        run_after_minutes: int = 0,
        from_user_id: str | None = None,
        importance: float = 0.7,
) -> int:
    """
    Convenience helper: schedule a simple "send a message" task.

    This is intended for manual task creation (outside the LLM planner).
    """
    ts = ensure_task_store(state)
    now_ts = time.time()
    due_at = now_ts + max(0, int(run_after_minutes)) * 60

    task_id = ts.add_task(
        kind="message",
        description=description,
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        room_id=room_id,
        due_at=due_at,
        importance=importance,
        meta={},
    )
    logger.debug("schedule_simple_message task_id=%s", task_id)
    return task_id


def maybe_handle_reply(state: AppState, user_id: str | None, room_id: str | None, message_text: str) -> None:
    """
    Interpret an incoming message as a reply to a previously created ask_user* task.

    Logic:
    - find a task kind LIKE 'ask_user%' with status=waiting_answer
      for (to_user_id == user_id, room_id == room_id)
    - store the answer_text
    - transition task -> answer_received and set due_at=now
      so the scheduler can run phase-2 ("reply back") immediately
    """
    if not user_id or not room_id:
        return

    ts = ensure_task_store(state)

    try:
        task = ts.find_waiting_ask_task(to_user_id=user_id, room_id=room_id)
    except Exception:
        logger.exception("find_waiting_ask_task failed user=%s room=%s", user_id, room_id)
        return

    if not task:
        return

    now_ts = time.time()
    try:
        ts.save_answer_and_mark_received(task.id, message_text, now_ts)
        logger.info("Captured answer for task_id=%s user=%s room=%s", task.id, user_id, room_id)
    except Exception:
        logger.exception("save_answer_and_mark_received failed task_id=%s", task.id)
