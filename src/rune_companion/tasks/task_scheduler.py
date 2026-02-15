# src/rune_companion/tasks/task_scheduler.py

from __future__ import annotations

"""
Task scheduler.

A small polling loop that:
- fetches runnable tasks,
- claims them (best-effort),
- sends outbound messages via an injected messenger port,
- advances task status or reschedules on failure.

Transport routing (room selection, formatting) belongs to the connector, not the scheduler.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum

from ..core.ports import OutboundMessenger, TaskRepo
from .task_models import Task, TaskStatus

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
    - where/how to actually send it (room selection, formatting, transport, etc.)
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
    - Phase 1: status=pending/in_progress -> send question_text to to_user_id, then WAITING_ANSWER
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
            text = f"Answer received: {ans}" if ans else (task.description or "").strip()
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
                room_id=(reply_room or task.room_id),
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
        task_store: TaskRepo,
        messenger: OutboundMessenger,
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
    - send via messenger.send_text(...)
    - advance status:
        * ASK        -> WAITING_ANSWER
        * REPLY_BACK -> DONE
        * MESSAGE    -> DONE
      On failure:
        - revert to PENDING (or ANSWER_RECEIVED for phase-2 tasks)
        - push due_at forward by retry_delay_seconds

    To stop the scheduler, cancel the coroutine/task.
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

        for task_any in tasks:
            # We expect Task objects from TaskStore; keep it defensive.
            task = task_any
            try:
                task_id = int(getattr(task, "id"))
            except Exception:
                continue

            # Claim to avoid duplicate dispatch.
            expected = [TaskStatus.PENDING, TaskStatus.ANSWER_RECEIVED]
            try:
                claimed = task_store.try_claim_task(task_id, expected=expected)
            except Exception:
                logger.exception("try_claim_task failed task_id=%s", task_id)
                continue

            if not claimed:
                continue

            dispatch = build_dispatch(task)
            if dispatch is None:
                logger.warning("Task %s is not dispatchable; cancelling", task_id)
                try:
                    task_store.update_task_status(task_id, TaskStatus.CANCELLED)
                except Exception:
                    logger.exception("update_task_status(cancelled) failed task_id=%s", task_id)
                continue

            try:
                await messenger.send_text(
                    text=dispatch.text,
                    room_id=dispatch.room_id,
                    to_user_id=dispatch.to_user_id,
                )

                if dispatch.phase == DispatchPhase.ASK:
                    task_store.update_task_status(task_id, TaskStatus.WAITING_ANSWER)
                    logger.info("Task %s -> waiting_answer", task_id)
                else:
                    task_store.update_task_status(task_id, TaskStatus.DONE)
                    logger.info("Task %s -> done", task_id)

            except Exception:
                logger.exception("dispatch send failed task_id=%s phase=%s", task_id, dispatch.phase.value)

                # Reschedule with a backoff.
                due_at = time.time() + retry_s

                try:
                    # If it was phase-2, preserve ANSWER_RECEIVED so we retry phase-2 later.
                    if dispatch.phase == DispatchPhase.REPLY_BACK:
                        task_store.update_task_fields(task_id, status=TaskStatus.ANSWER_RECEIVED, due_at=due_at)
                    else:
                        task_store.update_task_fields(task_id, status=TaskStatus.PENDING, due_at=due_at)
                except Exception:
                    logger.exception("update_task_fields(backoff) failed task_id=%s", task_id)

        await asyncio.sleep(sleep_s)
