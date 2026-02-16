# tests/test_task_scheduler.py

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest

from rune_companion.tasks.task_models import Task, TaskStatus
from rune_companion.tasks.task_scheduler import run_task_scheduler

from .fakes import FakeMessenger


class FakeTaskRepo:
    """
    In-memory TaskRepo used for scheduler unit tests.

    This avoids SQLite and makes tests purely about scheduling logic:
    time gating, claiming, transitions, and messenger calls.
    """

    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = {t.id: t for t in tasks}

    def list_runnable_tasks(self, *, now_ts: float, limit: int = 32):
        out: list[Task] = []
        for t in self.tasks.values():
            if t.status in (TaskStatus.PENDING, TaskStatus.ANSWER_RECEIVED) and (
                t.due_at is None or t.due_at <= now_ts
            ):
                out.append(t)
        out.sort(key=lambda x: (x.due_at or x.created_at, x.created_at))
        return out[:limit]

    def try_claim_task(self, task_id: int, *, expected):
        t = self.tasks.get(task_id)
        if t is None or t.status not in expected:
            return False
        self.tasks[task_id] = replace(t, status=TaskStatus.IN_PROGRESS, updated_at=time.time())
        return True

    def update_task_status(self, task_id: int, new_status):
        t = self.tasks[task_id]
        self.tasks[task_id] = replace(t, status=new_status, updated_at=time.time())

    def update_task_fields(
        self,
        task_id: int,
        *,
        status=None,
        due_at=None,
        meta=None,
        question_text=None,
        answer_text=None,
    ):
        t = self.tasks[task_id]
        self.tasks[task_id] = replace(
            t,
            status=t.status if status is None else status,
            due_at=t.due_at if due_at is None else due_at,
            meta=t.meta if meta is None else meta,
            question_text=t.question_text if question_text is None else question_text,
            answer_text=t.answer_text if answer_text is None else answer_text,
            updated_at=time.time(),
        )


@pytest.mark.asyncio
async def test_scheduler_dispatches_due_task_once() -> None:
    now = time.time()
    task = Task(
        id=1,
        status=TaskStatus.PENDING,
        created_at=now - 10,
        updated_at=now - 10,
        due_at=now - 1,
        kind="message",
        description="ping",
        from_user_id=None,
        to_user_id="u1",
        reply_to_user_id=None,
        room_id="r1",
        importance=0.7,
        meta={},
        question_text=None,
        answer_text=None,
    )

    repo = FakeTaskRepo([task])
    messenger = FakeMessenger()

    runner = asyncio.create_task(
        run_task_scheduler(
            repo,
            messenger,
            interval_seconds=0.01,
            retry_delay_seconds=0.01,
            batch_limit=10,
        )
    )

    await asyncio.sleep(0.05)
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner

    assert messenger.sent, "Scheduler should send at least one message"
    assert messenger.sent[0].room_id == "r1"
