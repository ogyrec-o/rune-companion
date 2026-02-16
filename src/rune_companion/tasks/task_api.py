# src/rune_companion/tasks/task_api.py

from __future__ import annotations

import logging
import time

from ..core.state import AppState

logger = logging.getLogger(__name__)


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
    Uses state.task_store (already constructed in bootstrap).
    """
    now_ts = time.time()
    due_at = now_ts + max(0, int(run_after_minutes)) * 60

    return state.task_store.add_task(
        kind="message",
        description=description,
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        room_id=room_id,
        due_at=due_at,
        importance=importance,
        meta={},
    )


def maybe_handle_reply(
    state: AppState, user_id: str | None, room_id: str | None, message_text: str
) -> None:
    """
    Compatibility helper (if some connector still calls it).
    NOTE: core/chat.py уже делает capture reply сам.
    """
    if not user_id or not room_id:
        return

    try:
        task = state.task_store.find_waiting_ask_task(to_user_id=user_id, room_id=room_id)
    except Exception:
        logger.exception("find_waiting_ask_task failed user=%s room=%s", user_id, room_id)
        return

    if not task:
        return

    now_ts = time.time()
    try:
        state.task_store.save_answer_and_mark_received(task.id, message_text, now_ts)
        logger.info("Captured answer for task_id=%s user=%s room=%s", task.id, user_id, room_id)
    except Exception:
        logger.exception("save_answer_and_mark_received failed task_id=%s", task.id)
