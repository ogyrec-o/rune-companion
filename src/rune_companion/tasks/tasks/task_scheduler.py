# tasks/task_scheduler.py

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable

from .task_models import Task, TaskStatus
from .task_store import TaskStore

logger = logging.getLogger(__name__)


class DispatchPhase(str, Enum):
    ASK = "ask"
    REPLY_BACK = "reply_back"
    MESSAGE = "message"


@dataclass(slots=True, frozen=True)
class TaskDispatch:
    """
    What the scheduler wants to send.

    The scheduler decides:
    - whether the task is runnable
    - which phase it is in (ask vs reply_back vs message)
    - the suggested text to send

    The connector decides:
    - how to map (to_user_id, room_id) into an actual send operation
      (e.g., choose a default room if missing)
    """

    task: Task
    phase: DispatchPhase
    text: str
    to_user_id: str | None
    room_id: str | None
    reply_to_user_id: str | None


def build_dispatch(task: Task) -> TaskDispatch | None:
    """
    Convert a stored Task into a dispatchable message.

    Two-phase tasks (ask_user*):
    - Phase 1: status=pending -> send question_text to to_user_id, then WAITING_ANSWER
    - Phase 2: status=answer_received -> send answer summary to reply_to_user_id, then DONE
    """
    kind = (task.kind or "").strip()
    status = task.status

    if not kind:
        return None

    if kind.startswith("ask_user"):
        if status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
            text = (task.question_text or task.description or "").strip()
            if not text:
                return None
            return TaskDispatch(
                task=task,
                phase=DispatchPhase.ASK,
                text=text,
                to_user_id=task.to_user_id,
                room_id=task.room_id,
                reply_to_user_id=task.reply_to_user_id,
            )

        if kind == "ask_user_and_reply_back" and status == TaskStatus.ANSWER_RECEIVED:
            ans = (task.answer_text or "").strip()
            text = f"Answer: {ans}" if ans else (task.description or "").strip()
            if not text:
                return None

            # Optional: allow storing a dedicated reply room in meta.
            reply_room = None
            try:
                reply_room = (task.meta or {}).get("reply_room_id")
            except Exception:
                reply_room = None

            return TaskDispatch(
                task=task,
                phase=DispatchPhase.REPLY_BACK,
                text=text,
                to_user_id=task.reply_to_user_id,
                room_id=reply_room or task.room_id,
                reply_to_user_id=task.reply_to_user_id,
            )

        # Unknown ask_user combination: fallback to "message" semantics.
        text = (task.description or "").strip()
        if not text:
            return None
        return TaskDispatch(
            task=task,
            phase=DispatchPhase.MESSAGE,
            text=text,
            to_user_id=task.to_user_id,
            room_id=task.room_id,
            reply_to_user_id=task.reply_to_user_id,
        )

    # One-shot tasks
    text = (task.description or "").strip()
    if not text:
        return None
    return TaskDispatch(
        task=task,
        phase=DispatchPhase.MESSAGE,
        text=text,
        to_user_id=task.to_user_id,
        room_id=task.room_id,
        reply_to_user_id=task.reply_to_user_id,
    )


async def run_task_scheduler(
        task_store: TaskStore,
        send_func: Callable[[TaskDispatch], Awaitable[None]],
        *,
        interval_seconds: float = 15.0,
        retry_delay_seconds: float = 60.0,
        batch_limit: int = 32,
) -> None:
    """
    Simple polling scheduler.

    Every interval_seconds:
    - fetch runnable tasks (pending / answer_received, due_at <= now or due_at is NULL)
    - claim the task (best-effort) to avoid duplicates
    - build dispatch payload (TaskDispatch)
    - call send_func(dispatch)
    - advance status:
        * ASK        -> WAITING_ANSWER
        * REPLY_BACK -> DONE
        * MESSAGE    -> DONE
      On failure:
        - revert to PENDING (or ANSWER_RECEIVED for phase-2 tasks)
        - push due_at forward by retry_delay_seconds
    """
    sleep_s = max(0.5, float(interval_seconds))
    retry_s = max(1.0, float(retry_delay_seconds))

    while True:
        now_ts = time.time()

        try:
            tasks = task_store.list_runnable_tasks(now_ts=now_ts, limit=int(batch_limit))
        except Exception:
            logger.exception("list_runnable_tasks failed")
            tasks = []

        for task in tasks:
            # Claim to avoid duplicate dispatch.
            expected = [TaskStatus.PENDING, TaskStatus.ANSWER_RECEIVED]
            if not task_store.try_claim_task(task.id, expected=expected):
                continue

            # Re-read is optional; we work with the snapshot.
            dispatch = build_dispatch(task)
            if dispatch is None:
                # Nothing to send -> cancel (or mark done). Here: cancel to keep visibility.
                logger.warning("Task %s is not dispatchable; cancelling", task.id)
                task_store.update_task_status(task.id, TaskStatus.CANCELLED)
                continue

            try:
                await send_func(dispatch)

                if dispatch.phase == DispatchPhase.ASK:
                    task_store.update_task_status(task.id, TaskStatus.WAITING_ANSWER)
                    logger.info("Task %s -> waiting_answer", task.id)
                else:
                    task_store.update_task_status(task.id, TaskStatus.DONE)
                    logger.info("Task %s -> done", task.id)

            except Exception:
                logger.exception("send_func failed task_id=%s phase=%s", task.id, dispatch.phase.value)

                # Reschedule with a backoff.
                due_at = time.time() + retry_s

                # If it was phase-2, preserve ANSWER_RECEIVED so we retry phase-2 later.
                if dispatch.phase == DispatchPhase.REPLY_BACK:
                    task_store.update_task_fields(task.id, status=TaskStatus.ANSWER_RECEIVED, due_at=due_at)
                else:
                    task_store.update_task_fields(task.id, status=TaskStatus.PENDING, due_at=due_at)

        await asyncio.sleep(sleep_s)
