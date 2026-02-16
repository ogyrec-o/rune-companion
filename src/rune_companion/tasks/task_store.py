# tasks/task_store.py

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .task_models import Task, TaskStatus

logger = logging.getLogger(__name__)


class TaskStore:
    """
    SQLite task store.

    The schema is intentionally simple and migration-safe:
    - create table if missing
    - use PRAGMA table_info to detect missing columns
    - add columns with ALTER TABLE only when needed

    Thread-safety:
    - each method opens its own SQLite connection
    """

    def __init__(self, db_path: str | Path = "tasks.sqlite3") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        try:
            total = self.count_tasks()
        except Exception:
            total = -1
        logger.info("TaskStore ready db=%s total=%s", self._db_path, total)

    def close(self) -> None:
        """Compatibility hook for shutdown (no persistent connections to close)."""
        return

    # ---- low-level helpers ----

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        self._configure_conn(conn)
        return conn

    @staticmethod
    def _configure_conn(conn: sqlite3.Connection) -> None:
        with contextlib.suppress(Exception):
            conn.execute("PRAGMA journal_mode=WAL")

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    due_at REAL,
                    kind TEXT NOT NULL,
                    description TEXT NOT NULL,
                    from_user_id TEXT,
                    to_user_id TEXT,
                    reply_to_user_id TEXT,
                    room_id TEXT,
                    importance REAL NOT NULL DEFAULT 0.7,
                    meta TEXT NOT NULL DEFAULT '{}',
                    question_text TEXT,
                    answer_text TEXT
                )
                """
            )

            # Migrations (safe): add missing columns.
            cur.execute("PRAGMA table_info(tasks)")
            cols = {row["name"] for row in cur.fetchall()}

            def add_col(name: str, decl: str) -> None:
                if name in cols:
                    return
                cur.execute(f"ALTER TABLE tasks ADD COLUMN {name} {decl}")
                logger.info("TaskStore migration: added column %s", name)

            # If any old DB is missing any of these, we add them.
            add_col("status", "TEXT NOT NULL DEFAULT 'pending'")
            add_col("created_at", "REAL NOT NULL DEFAULT 0")
            add_col("updated_at", "REAL NOT NULL DEFAULT 0")
            add_col("due_at", "REAL")
            add_col("kind", "TEXT NOT NULL DEFAULT 'generic'")
            add_col("description", "TEXT NOT NULL DEFAULT ''")
            add_col("from_user_id", "TEXT")
            add_col("to_user_id", "TEXT")
            add_col("reply_to_user_id", "TEXT")
            add_col("room_id", "TEXT")
            add_col("importance", "REAL NOT NULL DEFAULT 0.7")
            add_col("meta", "TEXT NOT NULL DEFAULT '{}'")
            add_col("question_text", "TEXT")
            add_col("answer_text", "TEXT")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_tasks_due_status ON tasks(status, due_at)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_participants "
                "ON tasks(to_user_id, from_user_id, reply_to_user_id)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tasks_room ON tasks(room_id)")

            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _meta_to_str(meta: dict[str, Any] | None) -> str:
        if not meta:
            return "{}"
        try:
            return json.dumps(meta, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to JSON-encode meta; storing {}.")
            return "{}"

    @staticmethod
    def _str_to_meta(s: str | None) -> dict[str, Any]:
        if not s:
            return {}
        try:
            val = json.loads(s)
            return val if isinstance(val, dict) else {}
        except Exception:
            return {}

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        return Task(
            id=int(row["id"]),
            status=TaskStatus.from_db(row["status"]),
            created_at=float(row["created_at"] or 0.0),
            updated_at=float(row["updated_at"] or 0.0),
            due_at=float(row["due_at"]) if row["due_at"] is not None else None,
            kind=str(row["kind"] or "generic"),
            description=str(row["description"] or ""),
            from_user_id=row["from_user_id"],
            to_user_id=row["to_user_id"],
            reply_to_user_id=row["reply_to_user_id"],
            room_id=row["room_id"],
            importance=float(row["importance"] or 0.7),
            meta=self._str_to_meta(row["meta"]),
            question_text=row["question_text"],
            answer_text=row["answer_text"],
        )

    # ---- public API ----

    def count_tasks(self) -> int:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM tasks")
            (n,) = cur.fetchone()
            return int(n)
        finally:
            conn.close()

    def add_task(
        self,
        *,
        kind: str,
        description: str,
        from_user_id: str | None = None,
        to_user_id: str | None = None,
        reply_to_user_id: str | None = None,
        room_id: str | None = None,
        due_at: float | None = None,
        importance: float = 0.7,
        meta: dict[str, Any] | None = None,
        status: TaskStatus = TaskStatus.PENDING,
        question_text: str | None = None,
        answer_text: str | None = None,
    ) -> int:
        if not kind or not kind.strip():
            raise ValueError("kind is required")
        if not description or not description.strip():
            raise ValueError("description is required")

        now = time.time()
        meta_str = self._meta_to_str(meta)

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO tasks(
                    status, created_at, updated_at, due_at,
                    kind, description,
                    from_user_id, to_user_id, reply_to_user_id, room_id,
                    importance, meta, question_text, answer_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    status.value,
                    now,
                    now,
                    due_at,
                    kind.strip(),
                    description.strip(),
                    from_user_id,
                    to_user_id,
                    reply_to_user_id,
                    room_id,
                    float(max(0.0, min(1.0, importance))),
                    meta_str,
                    question_text,
                    answer_text,
                ),
            )
            conn.commit()
            rowid = cur.lastrowid
            if rowid is None:
                raise RuntimeError("SQLite did not return lastrowid for memories insert")
            task_id = int(rowid)
            logger.debug(
                "Task added id=%s kind=%s status=%s due_at=%s",
                task_id,
                kind,
                status.value,
                due_at,
            )
            return task_id
        finally:
            conn.close()

    def list_runnable_tasks(self, *, now_ts: float, limit: int = 32) -> list[Task]:
        """
        Return tasks that are ready to be dispatched.

        Runnable statuses:
        - pending
        - answer_received

        A task is runnable if:
        - due_at IS NULL (meaning "any time"), OR
        - due_at <= now_ts
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT *
                FROM tasks
                WHERE status IN ('pending','answer_received')
                  AND (due_at IS NULL OR due_at <= ?)
                ORDER BY COALESCE(due_at, created_at) ASC, created_at ASC
                    LIMIT ?
                """,
                (float(now_ts), int(limit)),
            )
            rows = cur.fetchall()
            return [self._row_to_task(r) for r in rows]
        finally:
            conn.close()

    def try_claim_task(self, task_id: int, *, expected: Iterable[TaskStatus]) -> bool:
        """
        Best-effort claim to avoid duplicate dispatch.

        Atomically transitions:
          status IN expected  -> status = in_progress

        Returns True if the row was claimed by this caller.
        """
        exp = [e.value for e in expected]
        if not exp:
            return False

        now = time.time()
        conn = self._get_conn()
        try:
            placeholders = ",".join("?" for _ in exp)
            cur = conn.cursor()
            cur.execute(
                f"""
                UPDATE tasks
                SET status = 'in_progress', updated_at = ?
                WHERE id = ?
                  AND status IN ({placeholders})
                """,
                (now, int(task_id), *exp),
            )
            conn.commit()
            return cur.rowcount == 1
        finally:
            conn.close()

    def update_task_status(self, task_id: int, new_status: TaskStatus) -> None:
        now = time.time()
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                (new_status.value, now, int(task_id)),
            )
            conn.commit()
        finally:
            conn.close()

    def update_task_fields(
        self,
        task_id: int,
        *,
        status: TaskStatus | None = None,
        due_at: float | None = None,
        meta: dict[str, Any] | None = None,
        question_text: str | None = None,
        answer_text: str | None = None,
    ) -> None:
        fields: list[str] = []
        params: list[Any] = []

        if status is not None:
            fields.append("status = ?")
            params.append(status.value)

        if due_at is not None:
            fields.append("due_at = ?")
            params.append(float(due_at))

        if meta is not None:
            fields.append("meta = ?")
            params.append(self._meta_to_str(meta))

        if question_text is not None:
            fields.append("question_text = ?")
            params.append(question_text)

        if answer_text is not None:
            fields.append("answer_text = ?")
            params.append(answer_text)

        if not fields:
            return

        fields.append("updated_at = ?")
        params.append(time.time())
        params.append(int(task_id))

        sql = f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?"

        conn = self._get_conn()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def list_open_tasks_for_user(self, user_id: str, limit: int = 16) -> list[Task]:
        """
        Open tasks that involve the given user as a participant.

        Used for prompt injection ("Open tasks / promises related to this user").
        """
        if not user_id:
            return []

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT *
                FROM tasks
                WHERE status NOT IN ('done','cancelled')
                  AND (
                    to_user_id = ?
                        OR from_user_id = ?
                        OR reply_to_user_id = ?
                    )
                ORDER BY COALESCE(due_at, created_at) ASC
                    LIMIT ?
                """,
                (user_id, user_id, user_id, int(limit)),
            )
            return [self._row_to_task(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def find_waiting_ask_task(self, *, to_user_id: str, room_id: str) -> Task | None:
        """
        Find the oldest ask_user* task waiting for an answer
        from (to_user_id) in (room_id).
        """
        if not to_user_id or not room_id:
            return None

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT *
                FROM tasks
                WHERE kind LIKE 'ask_user%%'
                  AND status = 'waiting_answer'
                  AND to_user_id = ?
                  AND room_id = ?
                ORDER BY created_at ASC
                    LIMIT 1
                """,
                (to_user_id, room_id),
            )
            row = cur.fetchone()
            return self._row_to_task(row) if row else None
        finally:
            conn.close()

    def save_answer_and_mark_received(
        self, task_id: int, answer_text: str, now_ts: float | None = None
    ) -> None:
        """
        Record the answer and move the task into phase-2:
          status -> answer_received
          due_at  -> now (so scheduler can pick it immediately)
        """
        if now_ts is None:
            now_ts = time.time()

        conn = self._get_conn()
        try:
            conn.execute(
                """
                UPDATE tasks
                SET answer_text = ?,
                    status = 'answer_received',
                    due_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (answer_text, float(now_ts), float(now_ts), int(task_id)),
            )
            conn.commit()
        finally:
            conn.close()
