# memory_store.py

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_GLOBAL_SUBJECT_ID = "__GLOBAL__"
_UNSET: Any = object()


@dataclass(frozen=True, slots=True)
class MemoryItem:
    id: int
    subject_type: str  # "user" | "room" | "relationship" | "global" | ...
    subject_id: str  # mxid / room_id / reserved "__GLOBAL__"
    text: str  # the stored fact / note
    tags: list[str]
    importance: float  # 0.0..1.0
    last_updated: float  # UNIX timestamp
    source: str  # "auto" | "manual" | "system" | ...
    person_ref: str | None  # "user:@mxid" | "anon:xxx" | None


class MemoryStore:
    """
    SQLite-backed memory store.

    Thread-safety:
    - Each operation opens its own SQLite connection (no shared cursors).
    - This is sufficient for a simple local app with occasional concurrent calls.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

        try:
            total = self.count_memories()
        except Exception:
            total = -1
        logger.info("MemoryStore ready db=%s total=%s", self._db_path, total)

    def close(self) -> None:
        """Compatibility hook for state shutdown (no persistent connections to close)."""
        return

    # ---- low-level helpers ----

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        self._configure_conn(conn)
        return conn

    @staticmethod
    def _configure_conn(conn: sqlite3.Connection) -> None:
        # Best-effort pragmas. Failures should not stop the app.
        try:
            conn.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass

    @staticmethod
    def _tags_to_str(tags: Iterable[str] | None) -> str:
        if not tags:
            return ""
        cleaned = {str(t).strip() for t in tags if str(t).strip()}
        return ",".join(sorted(cleaned))

    @staticmethod
    def _str_to_tags(s: str | None) -> list[str]:
        if not s:
            return []
        return [t.strip() for t in s.split(",") if t.strip()]

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                                                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                                                        subject_type  TEXT NOT NULL,
                                                        subject_id    TEXT NOT NULL,
                                                        text          TEXT NOT NULL,
                                                        tags          TEXT NOT NULL DEFAULT '',
                                                        importance    REAL NOT NULL DEFAULT 0.5,
                                                        last_updated  REAL NOT NULL,
                                                        source        TEXT NOT NULL DEFAULT 'auto',
                                                        person_ref    TEXT
                )
                """
            )

            # Migrations: add missing columns safely via PRAGMA table_info.
            cur.execute("PRAGMA table_info(memories)")
            cols = {row["name"] for row in cur.fetchall()}

            def add_col(name: str, decl: str) -> None:
                if name in cols:
                    return
                cur.execute(f"ALTER TABLE memories ADD COLUMN {name} {decl}")
                logger.info("MemoryStore migration: added column %s", name)

            add_col("person_ref", "TEXT")

            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_subject "
                "ON memories(subject_type, subject_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_importance "
                "ON memories(importance, last_updated)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_person_ref "
                "ON memories(person_ref)"
            )

            conn.commit()
        finally:
            conn.close()

    # ---- public API ----

    def add_memory(
            self,
            *,
            subject_type: str,
            subject_id: str,
            text: str,
            tags: list[str] | None = None,
            importance: float = 0.5,
            source: str = "auto",
            person_ref: str | None = None,
    ) -> int:
        """Insert a memory record and return its id."""
        if not subject_type:
            raise ValueError("subject_type is required")
        if not subject_id:
            raise ValueError("subject_id is required")
        if not text or not text.strip():
            raise ValueError("text is required")

        ts = time.time()
        tag_str = self._tags_to_str(tags)
        imp = float(max(0.0, min(1.0, importance)))

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO memories(
                    subject_type,
                    subject_id,
                    text,
                    tags,
                    importance,
                    last_updated,
                    source,
                    person_ref
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (subject_type, subject_id, text.strip(), tag_str, imp, ts, source, person_ref),
            )
            conn.commit()
            mem_id = int(cur.lastrowid)
            logger.debug(
                "MemoryStore add id=%s type=%s subject=%s imp=%.2f tags=%s",
                mem_id,
                subject_type,
                subject_id,
                imp,
                tag_str,
            )
            return mem_id
        finally:
            conn.close()

    def update_memory(
            self,
            mem_id: int,
            *,
            text: str | None = None,
            tags: list[str] | None = None,
            importance: float | None = None,
            person_ref: str | None | Any = _UNSET,
    ) -> None:
        """
        Update a memory record partially.

        Note: person_ref supports three states:
        - omitted (default): do not change
        - None: set to NULL
        - str: set to this value
        """
        if mem_id <= 0:
            raise ValueError("mem_id must be positive")

        fields: list[str] = []
        params: list[Any] = []

        if text is not None:
            fields.append("text = ?")
            params.append(text.strip())

        if tags is not None:
            fields.append("tags = ?")
            params.append(self._tags_to_str(tags))

        if importance is not None:
            imp = float(max(0.0, min(1.0, importance)))
            fields.append("importance = ?")
            params.append(imp)

        if person_ref is not _UNSET:
            fields.append("person_ref = ?")
            params.append(person_ref)

        if not fields:
            return

        fields.append("last_updated = ?")
        params.append(time.time())

        params.append(mem_id)

        sql = f"UPDATE memories SET {', '.join(fields)} WHERE id = ?"

        conn = self._get_conn()
        try:
            conn.execute(sql, params)
            conn.commit()
            logger.debug("MemoryStore update id=%s fields=%s", mem_id, fields)
        finally:
            conn.close()

    def delete_memory(self, mem_id: int) -> None:
        """Delete a memory record by id."""
        if mem_id <= 0:
            return
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            conn.commit()
            logger.debug("MemoryStore delete id=%s", mem_id)
        finally:
            conn.close()

    def count_memories(self) -> int:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM memories")
            (n,) = cur.fetchone()
            return int(n)
        finally:
            conn.close()

    def query_memory(
            self,
            *,
            subject_type: str | None = None,
            subject_id: str | None = None,
            min_importance: float = 0.0,
            limit: int = 20,
            tag: str | None = None,
            person_ref: str | None = None,
    ) -> list[MemoryItem]:
        """
        Query memories ordered by (importance DESC, last_updated DESC).

        Filters:
        - subject_type / subject_id: exact match
        - tag: exact tag match (comma-separated tag list)
        - person_ref: exact match
        """
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list[Any] = []

        if subject_type is not None:
            sql += " AND subject_type = ?"
            params.append(subject_type)

        if subject_id is not None:
            sql += " AND subject_id = ?"
            params.append(subject_id)

        sql += " AND importance >= ?"
        params.append(float(min_importance))

        if tag:
            sql += " AND (',' || tags || ',') LIKE ?"
            params.append(f"%,{tag},%")

        if person_ref is not None:
            sql += " AND person_ref = ?"
            params.append(person_ref)

        sql += " ORDER BY importance DESC, last_updated DESC LIMIT ?"
        params.append(int(max(0, limit)))

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()

            out: list[MemoryItem] = []
            for r in rows:
                out.append(
                    MemoryItem(
                        id=int(r["id"]),
                        subject_type=str(r["subject_type"]),
                        subject_id=str(r["subject_id"]),
                        text=str(r["text"]),
                        tags=self._str_to_tags(r["tags"]),
                        importance=float(r["importance"]),
                        last_updated=float(r["last_updated"]),
                        source=str(r["source"]),
                        person_ref=r["person_ref"],
                    )
                )
            logger.debug("MemoryStore query -> %d rows", len(out))
            return out
        finally:
            conn.close()

    def prune_subject(self, subject_type: str, subject_id: str, max_items: int) -> None:
        """
        Keep only max_items best memories for a given (subject_type, subject_id).

        Keeps the top items by (importance DESC, last_updated DESC).
        Deletes the rest.
        """
        if max_items <= 0:
            return
        if not subject_type or not subject_id:
            return

        conn = self._get_conn()
        try:
            cur = conn.cursor()

            cur.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM memories
                WHERE subject_type = ? AND subject_id = ?
                """,
                (subject_type, subject_id),
            )
            row = cur.fetchone()
            total = int(row["cnt"]) if row is not None else 0
            if total <= max_items:
                return

            # Identify ids to remove: everything after the first max_items.
            to_remove = total - max_items
            cur.execute(
                """
                SELECT id
                FROM memories
                WHERE subject_type = ? AND subject_id = ?
                ORDER BY importance DESC, last_updated DESC
                    LIMIT ? OFFSET ?
                """,
                (subject_type, subject_id, to_remove, max_items),
            )
            ids = [int(r["id"]) for r in cur.fetchall()]
            if not ids:
                return

            placeholders = ",".join("?" for _ in ids)
            cur.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
            conn.commit()
            logger.info(
                "MemoryStore pruned type=%s subject=%s removed=%d (max=%d)",
                subject_type,
                subject_id,
                len(ids),
                max_items,
            )
        finally:
            conn.close()

    @staticmethod
    def global_subject_id() -> str:
        return _GLOBAL_SUBJECT_ID
