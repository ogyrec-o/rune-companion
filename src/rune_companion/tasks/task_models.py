# tasks/task_models.py

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    """
    Task lifecycle status.

    Notes:
    - "in_progress" is kept for backward compatibility and as an optional claim-lock
      to avoid duplicate dispatch when multiple schedulers run.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"  # legacy / internal claim
    WAITING_ANSWER = "waiting_answer"
    ANSWER_RECEIVED = "answer_received"
    DONE = "done"
    CANCELLED = "cancelled"

    @classmethod
    def from_db(cls, raw: str | None) -> TaskStatus:
        if not raw:
            return cls.PENDING
        try:
            return cls(raw)
        except Exception:
            return cls.PENDING


@dataclass(slots=True)
class Task:
    id: int
    status: TaskStatus
    created_at: float
    updated_at: float
    due_at: float | None

    kind: str
    description: str

    from_user_id: str | None
    to_user_id: str | None
    reply_to_user_id: str | None
    room_id: str | None

    importance: float
    meta: dict[str, Any]

    question_text: str | None = None
    answer_text: str | None = None
