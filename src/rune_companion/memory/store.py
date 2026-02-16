# src/rune_companion/memory/store.py

from __future__ import annotations

import contextlib
import json
import logging
import re
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GLOBAL_SUBJECT_ID = "__GLOBAL__"
_UNSET: Any = object()

_SOURCE_PRIORITY = {
    "manual": 3,
    "system": 2,
    "explicit": 2,
    "auto": 1,
}

_FACT_FORMAT_TEXT = "text"
_FACT_FORMAT_JSON = "json"


def _source_pri(src: str) -> int:
    s = (src or "").strip().lower() or "auto"
    return _SOURCE_PRIORITY.get(s, 1)


@dataclass(frozen=True, slots=True)
class MemoryItem:
    id: int
    subject_type: str
    subject_id: str
    text: str
    tags: list[str]
    importance: float
    last_updated: float
    source: str
    person_ref: str | None

    decay_days: float  # half-life in days
    last_accessed: float  # recall timestamp
    n_reinforced: int  # how many times we reinforced/merged
    pinned: int  # 1/0 (manual/system)


@dataclass(frozen=True, slots=True)
class FactItem:
    id: int
    subject_type: str
    subject_id: str
    key: str
    value: Any
    tags: list[str]
    confidence: float
    last_updated: float
    source: str
    evidence: str
    person_ref: str | None

    decay_days: float
    last_accessed: float
    n_reinforced: int
    pinned: int


class MemoryStore:
    """
    SQLite-backed memory store.

    Thread-safety:
    - Each operation opens its own SQLite connection (no shared cursors).

    "Giants-grade" behaviors:
    - dedupe on (subject_type, subject_id, text) for memories,
    - structured Facts ("slots") with unique (subject_type, subject_id, key),
    - conflict policy for facts (never overwrite higher-priority source with lower),
    - tags merge + timestamp bump.
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
        return

    # ---- low-level helpers ----

    @staticmethod
    def _clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    @staticmethod
    def _clamp_decay_days(x: float) -> float:
        # protect against zero/negative and insane values
        return float(max(0.1, min(36500.0, x)))

    @staticmethod
    def _default_decay_days_for_memory(importance: float, tags: list[str], source: str) -> float:
        src = (source or "auto").strip().lower()
        t = set(tags or [])
        if src in {"manual", "system"} or "pinned" in t:
            return 36500.0  # ~100y

        imp = MemoryStore._clamp01(float(importance))
        # 1..180 days (square => high-importance lasts much longer)
        return 1.0 + (imp * imp) * 179.0

    @staticmethod
    def _default_decay_days_for_fact(confidence: float, tags: list[str], source: str) -> float:
        src = (source or "auto").strip().lower()
        t = set(tags or [])
        if src in {"manual", "system"} or "pinned" in t:
            return 36500.0

        c = MemoryStore._clamp01(float(confidence))
        # 30..3650 days (1 month..10 years)
        return 30.0 + (c * c) * (3650.0 - 30.0)

    @staticmethod
    def _decay_multiplier(age_seconds: float, decay_days: float) -> float:
        hl = MemoryStore._clamp_decay_days(decay_days) * 86400.0
        return float(0.5 ** (max(0.0, float(age_seconds)) / hl))

    @staticmethod
    def _effective_score(
        base: float,
        now_ts: float,
        last_updated: float,
        decay_days: float,
        pinned: int,
    ) -> float:
        if int(pinned or 0) != 0:
            return float(base)
        age = float(now_ts) - float(last_updated)
        return float(base) * MemoryStore._decay_multiplier(age, decay_days)

    def _delete_memories_by_ids(self, ids: list[int]) -> None:
        ids = [int(x) for x in (ids or []) if int(x) > 0]
        if not ids:
            return
        conn = self._get_conn()
        try:
            ph = ",".join("?" for _ in ids)
            conn.execute(f"DELETE FROM memories WHERE id IN ({ph})", ids)
            conn.commit()
        finally:
            conn.close()

    def _touch_memories_by_ids(self, ids: list[int], *, now_ts: float) -> None:
        ids = [int(x) for x in (ids or []) if int(x) > 0]
        if not ids:
            return
        conn = self._get_conn()
        try:
            ph = ",".join("?" for _ in ids)
            conn.execute(
                f"""
                UPDATE memories
                SET last_accessed = ?,
                    n_reinforced = n_reinforced + 1
                WHERE id IN ({ph})
                """,
                [float(now_ts), *ids],
            )
            conn.commit()
        finally:
            conn.close()

    def _delete_facts_by_ids(self, ids: list[int]) -> None:
        ids = [int(x) for x in (ids or []) if int(x) > 0]
        if not ids:
            return
        conn = self._get_conn()
        try:
            ph = ",".join("?" for _ in ids)
            conn.execute(f"DELETE FROM memory_facts WHERE id IN ({ph})", ids)
            conn.commit()
        finally:
            conn.close()

    def _touch_facts_by_ids(self, ids: list[int], *, now_ts: float) -> None:
        ids = [int(x) for x in (ids or []) if int(x) > 0]
        if not ids:
            return
        conn = self._get_conn()
        try:
            ph = ",".join("?" for _ in ids)
            conn.execute(
                f"""
                UPDATE memory_facts
                SET last_accessed = ?,
                    n_reinforced = n_reinforced + 1
                WHERE id IN ({ph})
                """,
                [float(now_ts), *ids],
            )
            conn.commit()
        finally:
            conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        self._configure_conn(conn)
        return conn

    @staticmethod
    def _configure_conn(conn: sqlite3.Connection) -> None:
        with contextlib.suppress(Exception):
            conn.execute("PRAGMA foreign_keys=ON")
        with contextlib.suppress(Exception):
            conn.execute("PRAGMA journal_mode=WAL")

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

    @staticmethod
    def _merge_tags(a: str, b: str) -> str:
        sa = set(MemoryStore._str_to_tags(a))
        sb = set(MemoryStore._str_to_tags(b))
        return MemoryStore._tags_to_str(sorted(sa | sb))

    @staticmethod
    def _pick_source(old: str, new: str) -> str:
        o = str(old or "").strip() or "auto"
        n = str(new or "").strip() or "auto"
        return n if _source_pri(n) >= _source_pri(o) else o

    @staticmethod
    def _normalize_fact_key(key: str) -> str:
        k = (key or "").strip().lower()
        k = re.sub(r"\s+", "_", k)
        k = re.sub(r"[^a-z0-9_]+", "", k)
        k = re.sub(r"_+", "_", k).strip("_")
        return k

    @staticmethod
    def _encode_fact_value(value: Any) -> tuple[str, str]:
        """
        Returns: (value_format, value_str)
        - text: stored as plain string
        - json: stored as JSON string
        """
        if value is None:
            return _FACT_FORMAT_TEXT, ""
        if isinstance(value, (list, dict)):
            return _FACT_FORMAT_JSON, json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        # ints/floats/bools -> string
        return _FACT_FORMAT_TEXT, str(value).strip()

    @staticmethod
    def _decode_fact_value(value_format: str, value_str: str) -> Any:
        fmt = (value_format or _FACT_FORMAT_TEXT).strip().lower()
        if fmt == _FACT_FORMAT_JSON:
            try:
                return json.loads(value_str or "null")
            except Exception:
                # corrupt row; fall back to raw text
                return value_str
        return value_str

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # ---- memories (unstructured notes) ----
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_type  TEXT NOT NULL,
                    subject_id    TEXT NOT NULL,
                    text          TEXT NOT NULL,
                    tags          TEXT NOT NULL DEFAULT '',
                    importance    REAL NOT NULL DEFAULT 0.5,
                    decay_days    REAL NOT NULL DEFAULT 30,
                    last_accessed REAL NOT NULL DEFAULT 0,
                    n_reinforced  INTEGER NOT NULL DEFAULT 0,
                    pinned        INTEGER NOT NULL DEFAULT 0,
                    last_updated  REAL NOT NULL,
                    source        TEXT NOT NULL DEFAULT 'auto',
                    person_ref    TEXT
                )
                """
            )

            cur.execute("PRAGMA table_info(memories)")
            cols = {row["name"] for row in cur.fetchall()}

            def add_col(table: str, name: str, decl: str) -> None:
                if name in cols:
                    return
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                logger.info("MemoryStore migration: added column %s.%s", table, name)

            add_col("memories", "person_ref", "TEXT")
            add_col("memories", "decay_days", "REAL NOT NULL DEFAULT 30")
            add_col("memories", "last_accessed", "REAL NOT NULL DEFAULT 0")
            add_col("memories", "n_reinforced", "INTEGER NOT NULL DEFAULT 0")
            add_col("memories", "pinned", "INTEGER NOT NULL DEFAULT 0")

            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_subject "
                "ON memories(subject_type, subject_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_importance "
                "ON memories(importance, last_updated)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_person_ref ON memories(person_ref)"
            )

            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed)"
            )

            # ---- facts (structured slots) ----
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_facts (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_type   TEXT NOT NULL,
                    subject_id     TEXT NOT NULL,
                    key            TEXT NOT NULL,
                    value_format   TEXT NOT NULL DEFAULT 'text',
                    value          TEXT NOT NULL,
                    tags           TEXT NOT NULL DEFAULT '',
                    confidence     REAL NOT NULL DEFAULT 0.8,
                    decay_days     REAL NOT NULL DEFAULT 365,
                    last_accessed  REAL NOT NULL DEFAULT 0,
                    n_reinforced   INTEGER NOT NULL DEFAULT 0,
                    pinned         INTEGER NOT NULL DEFAULT 0,
                    last_updated   REAL NOT NULL,
                    source         TEXT NOT NULL DEFAULT 'auto',
                    evidence       TEXT NOT NULL DEFAULT '',
                    person_ref     TEXT
                )
                """
            )

            cur.execute("PRAGMA table_info(memory_facts)")
            fact_cols = {row["name"] for row in cur.fetchall()}

            def add_fact_col(name: str, decl: str) -> None:
                if name in fact_cols:
                    return
                cur.execute(f"ALTER TABLE memory_facts ADD COLUMN {name} {decl}")
                logger.info("MemoryStore migration: added column memory_facts.%s", name)

            add_fact_col("person_ref", "TEXT")
            add_fact_col("evidence", "TEXT NOT NULL DEFAULT ''")
            add_fact_col("value_format", "TEXT NOT NULL DEFAULT 'text'")
            add_fact_col("decay_days", "REAL NOT NULL DEFAULT 365")
            add_fact_col("last_accessed", "REAL NOT NULL DEFAULT 0")
            add_fact_col("n_reinforced", "INTEGER NOT NULL DEFAULT 0")
            add_fact_col("pinned", "INTEGER NOT NULL DEFAULT 0")

            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_unique "
                "ON memory_facts(subject_type, subject_id, key)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_subject "
                "ON memory_facts(subject_type, subject_id, confidence, last_updated)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_person_ref ON memory_facts(person_ref)"
            )

            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_last_accessed ON memory_facts(last_accessed)"
            )

            conn.commit()
        finally:
            conn.close()

    # ----------------------------------------------------------------------------------
    # Unstructured memories API (dedupe + decay)
    # ----------------------------------------------------------------------------------

    def add_memory(
        self,
        *,
        subject_type: str,
        subject_id: str,
        text: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
        decay_days: float | None = None,
        source: str = "auto",
        person_ref: str | None = None,
        pinned: int | None = None,
    ) -> int:
        """
        Insert a memory record and return its id.

        Dedup:
        If a memory with the same (subject_type, subject_id, text) exists,
        we update it (merge tags, bump importance, refresh last_updated)
        and return existing id.
        """
        if not subject_type:
            raise ValueError("subject_type is required")
        if not subject_id:
            raise ValueError("subject_id is required")
        if not text or not text.strip():
            raise ValueError("text is required")

        norm_text = text.strip()
        ts = time.time()

        tags_list = list(tags or [])
        tag_str = self._tags_to_str(tags_list)

        imp = float(max(0.0, min(1.0, importance)))
        src = str(source or "auto").strip() or "auto"

        pin = (
            int(pinned)
            if pinned is not None
            else (1 if (src in {"manual", "system"} or "pinned" in set(tags_list)) else 0)
        )
        dd = self._clamp_decay_days(
            float(decay_days)
            if isinstance(decay_days, (int, float))
            else self._default_decay_days_for_memory(imp, tags_list, src)
        )

        conn = self._get_conn()
        try:
            cur = conn.cursor()

            cur.execute(
                """
                SELECT id, tags, importance, source
                FROM memories
                WHERE subject_type = ? AND subject_id = ? AND text = ?
                    LIMIT 1
                """,
                (subject_type, subject_id, norm_text),
            )
            row = cur.fetchone()
            if row is not None:
                mem_id = int(row["id"])
                old_tags = str(row["tags"] or "")
                old_imp = float(row["importance"] or 0.0)
                old_src = str(row["source"] or "auto")

                merged_tags = self._merge_tags(old_tags, tag_str)
                merged_imp = max(old_imp, imp)
                merged_src = self._pick_source(old_src, src)

                cur.execute(
                    """
                    UPDATE memories
                    SET tags = ?, importance = ?, last_updated = ?, source = ?,
                        person_ref = COALESCE(person_ref, ?),
                        decay_days = MAX(decay_days, ?),
                        last_accessed = ?,
                        n_reinforced = n_reinforced + 1,
                        pinned = MAX(pinned, ?)
                    WHERE id = ?
                    """,
                    (merged_tags, merged_imp, ts, merged_src, person_ref, dd, ts, pin, mem_id),
                )
                conn.commit()

                logger.debug(
                    "MemoryStore dedup update id=%s type=%s subject=%s imp=%.2f tags=%s",
                    mem_id,
                    subject_type,
                    subject_id,
                    merged_imp,
                    merged_tags,
                )
                return mem_id

            cur.execute(
                """
                INSERT INTO memories(
                    subject_type, subject_id, text, tags, importance,
                    decay_days, last_accessed, n_reinforced, pinned,
                    last_updated, source, person_ref
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subject_type,
                    subject_id,
                    norm_text,
                    tag_str,
                    imp,
                    dd,
                    ts,
                    0,
                    pin,
                    ts,
                    src,
                    person_ref,
                ),
            )
            conn.commit()

            rowid = cur.lastrowid
            if rowid is None:
                raise RuntimeError("SQLite did not return lastrowid for memories insert")
            mem_id = int(rowid)

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
        decay_days: float | None = None,
        pinned: int | None = None,
        person_ref: str | None | Any = _UNSET,
    ) -> None:
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

        if decay_days is not None:
            fields.append("decay_days = ?")
            params.append(self._clamp_decay_days(float(decay_days)))

        if pinned is not None:
            fields.append("pinned = ?")
            params.append(int(pinned))

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
        min_effective: float = 0.03,
        overscan: int = 6,
        now_ts: float | None = None,
        limit: int = 20,
        tag: str | None = None,
        person_ref: str | None = None,
        touch: bool = False,
    ) -> list[MemoryItem]:
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

        scan = int(max(0, limit) * max(1, int(overscan)))
        sql += " ORDER BY importance DESC, last_updated DESC LIMIT ?"
        params.append(scan)

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()

            now = float(now_ts) if now_ts is not None else time.time()

            raw_items: list[MemoryItem] = []
            for r in rows:
                lu = float(r["last_updated"] or 0.0)
                la = float(r["last_accessed"] or 0.0) or lu
                raw_items.append(
                    MemoryItem(
                        id=int(r["id"]),
                        subject_type=str(r["subject_type"]),
                        subject_id=str(r["subject_id"]),
                        text=str(r["text"]),
                        tags=self._str_to_tags(r["tags"]),
                        importance=float(r["importance"]),
                        last_updated=lu,
                        source=str(r["source"]),
                        person_ref=r["person_ref"],
                        decay_days=float(r["decay_days"] or 30.0),
                        last_accessed=la,
                        n_reinforced=int(r["n_reinforced"] or 0),
                        pinned=int(r["pinned"] or 0),
                    )
                )

            scored: list[tuple[float, MemoryItem]] = []
            to_delete: list[int] = []
            me = float(max(0.0, min(1.0, min_effective)))

            for m in raw_items:
                eff = self._effective_score(
                    m.importance, now, m.last_updated, m.decay_days, m.pinned
                )
                if (not m.pinned) and eff < me:
                    to_delete.append(m.id)
                    continue
                scored.append((eff, m))

            scored.sort(key=lambda x: (x[0], x[1].last_updated), reverse=True)
            result = [m for _, m in scored[: int(max(0, limit))]]

            if to_delete:
                self._delete_memories_by_ids(to_delete)

            if touch and result:
                self._touch_memories_by_ids([m.id for m in result], now_ts=now)

            logger.debug(
                "MemoryStore query -> %d rows (kept=%d deleted=%d)",
                len(raw_items),
                len(result),
                len(to_delete),
            )
            return result
        finally:
            conn.close()

    def prune_subject(self, subject_type: str, subject_id: str, max_items: int) -> None:
        if max_items <= 0:
            return
        if not subject_type or not subject_id:
            return

        now = time.time()
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, importance, last_updated, decay_days, pinned
                FROM memories
                WHERE subject_type = ? AND subject_id = ?
                """,
                (subject_type, subject_id),
            )
            rows = cur.fetchall()
            if not rows:
                return

            pinned_ids: list[int] = []
            non_pinned: list[tuple[float, float, int]] = []  # (eff, last_updated, id)

            for r in rows:
                mid = int(r["id"])
                imp = float(r["importance"] or 0.0)
                lu = float(r["last_updated"] or 0.0)
                dd = float(r["decay_days"] or 30.0)
                pin = int(r["pinned"] or 0)
                eff = self._effective_score(imp, now, lu, dd, pin)

                if pin:
                    pinned_ids.append(mid)
                else:
                    non_pinned.append((eff, lu, mid))

            non_pinned.sort(key=lambda x: (x[0], x[1]), reverse=True)

            slots = max(0, int(max_items) - len(pinned_ids))
            delete_ids = [mid for _, _, mid in non_pinned[slots:]]

            if delete_ids:
                ph = ",".join("?" for _ in delete_ids)
                cur.execute(f"DELETE FROM memories WHERE id IN ({ph})", delete_ids)
                conn.commit()
                logger.info(
                    "MemoryStore pruned type=%s subject=%s removed=%d (max=%d pinned=%d)",
                    subject_type,
                    subject_id,
                    len(delete_ids),
                    max_items,
                    len(pinned_ids),
                )
        finally:
            conn.close()

    # ----------------------------------------------------------------------------------
    # Structured facts API (slots)
    # ----------------------------------------------------------------------------------

    def upsert_fact(
        self,
        *,
        subject_type: str,
        subject_id: str,
        key: str,
        value: Any,
        tags: list[str] | None = None,
        confidence: float = 0.8,
        source: str = "auto",
        evidence: str = "",
        person_ref: str | None = None,
        decay_days: float | None = None,
        pinned: int | None = None,
    ) -> int:
        """
        Upsert a structured fact by unique (subject_type, subject_id, key).

        Conflict policy:
        - If same value => merge tags, bump confidence, refresh timestamp, pick higher source.
        - If different value => overwrite only if new source priority >= old source priority.
        """
        if not subject_type:
            raise ValueError("subject_type is required")
        if not subject_id:
            raise ValueError("subject_id is required")

        k = self._normalize_fact_key(key)
        if not k:
            raise ValueError("key is required")

        fmt, v = self._encode_fact_value(value)
        v = (v or "").strip()
        src = (source or "auto").strip() or "auto"
        ev = (evidence or "").strip()

        conf = float(max(0.0, min(1.0, confidence)))
        ts = time.time()

        tags_list = list(tags or [])
        tag_str = self._tags_to_str(tags_list)

        pin = (
            int(pinned)
            if pinned is not None
            else (1 if (src in {"manual", "system"} or "pinned" in set(tags_list)) else 0)
        )
        dd = self._clamp_decay_days(
            float(decay_days)
            if isinstance(decay_days, (int, float))
            else self._default_decay_days_for_fact(conf, tags_list, src)
        )

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, value_format, value, tags, confidence, source, evidence
                FROM memory_facts
                WHERE subject_type = ? AND subject_id = ? AND key = ?
                    LIMIT 1
                """,
                (subject_type, subject_id, k),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    """
                    INSERT INTO memory_facts(
                        subject_type, subject_id, key, value_format, value,
                        tags, confidence, decay_days, last_accessed, n_reinforced, pinned,
                        last_updated, source, evidence, person_ref
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        subject_type,
                        subject_id,
                        k,
                        fmt,
                        v,
                        tag_str,
                        conf,
                        dd,
                        ts,
                        0,
                        pin,
                        ts,
                        src,
                        ev,
                        person_ref,
                    ),
                )
                conn.commit()
                rid = cur.lastrowid
                if rid is None:
                    raise RuntimeError("SQLite did not return lastrowid for fact insert")
                return int(rid)

            fact_id = int(row["id"])
            old_fmt = str(row["value_format"] or _FACT_FORMAT_TEXT)
            old_v = str(row["value"] or "")
            old_tags = str(row["tags"] or "")
            old_conf = float(row["confidence"] or 0.0)
            old_src = str(row["source"] or "auto")
            old_ev = str(row["evidence"] or "")

            same_value = (old_fmt == fmt) and (old_v == v)
            merged_tags = self._merge_tags(old_tags, tag_str)
            merged_src = self._pick_source(old_src, src)

            if same_value:
                merged_conf = max(old_conf, conf)
                # Keep best/longest evidence (optional, but helps debugging)
                merged_ev = old_ev if (len(old_ev) >= len(ev)) else ev
                cur.execute(
                    """
                    UPDATE memory_facts
                    SET tags = ?, confidence = ?, last_updated = ?, source = ?, evidence = ?,
                        person_ref = COALESCE(person_ref, ?),
                        decay_days = MAX(decay_days, ?),
                        last_accessed = ?,
                        n_reinforced = n_reinforced + 1,
                        pinned = MAX(pinned, ?)
                    WHERE id = ?
                    """,
                    (
                        merged_tags,
                        merged_conf,
                        ts,
                        merged_src,
                        merged_ev,
                        person_ref,
                        dd,
                        ts,
                        pin,
                        fact_id,
                    ),
                )
                conn.commit()
                return fact_id

            # Different value: overwrite only if new >= old priority.
            if _source_pri(src) < _source_pri(old_src):
                logger.debug(
                    "Fact overwrite blocked (lower priority): key=%s old_src=%s new_src=%s "
                    "old=%r new=%r",
                    k,
                    old_src,
                    src,
                    old_v,
                    v,
                )
                # Still merge tags/evidence lightly (optional) to not lose metadata.
                merged_conf = max(old_conf, conf)
                merged_ev = old_ev if (len(old_ev) >= len(ev)) else ev
                cur.execute(
                    """
                    UPDATE memory_facts
                    SET tags = ?, confidence = ?, last_updated = ?, evidence = ?,
                        person_ref = COALESCE(person_ref, ?),
                        decay_days = MAX(decay_days, ?),
                        last_accessed = ?,
                        n_reinforced = n_reinforced + 1,
                        pinned = MAX(pinned, ?)
                    WHERE id = ?
                    """,
                    (merged_tags, merged_conf, ts, merged_ev, person_ref, dd, ts, pin, fact_id),
                )
                conn.commit()
                return fact_id

            # Allowed overwrite.
            merged_conf = max(old_conf, conf)
            cur.execute(
                """
                UPDATE memory_facts
                SET value_format = ?, value = ?, tags = ?, confidence = ?, last_updated = ?,
                    source = ?, evidence = ?, person_ref = COALESCE(person_ref, ?),
                    decay_days = MAX(decay_days, ?),
                    last_accessed = ?,
                    n_reinforced = n_reinforced + 1,
                    pinned = MAX(pinned, ?)
                WHERE id = ?
                """,
                (
                    fmt,
                    v,
                    merged_tags,
                    merged_conf,
                    ts,
                    merged_src,
                    ev,
                    person_ref,
                    dd,
                    ts,
                    pin,
                    fact_id,
                ),
            )
            conn.commit()
            return fact_id
        finally:
            conn.close()

    def add_fact_value(
        self,
        *,
        subject_type: str,
        subject_id: str,
        key: str,
        value: str,
        tags: list[str] | None = None,
        confidence: float = 0.75,
        source: str = "auto",
        evidence: str = "",
        person_ref: str | None = None,
    ) -> int:
        """
        Treat fact value as a SET (stored as JSON list) and add one element (deduped).
        """
        k = self._normalize_fact_key(key)
        v = (value or "").strip()
        if not k or not v:
            raise ValueError("key/value required")

        existing = self.get_fact(subject_type=subject_type, subject_id=subject_id, key=k)
        items: list[str] = []
        if existing is not None:
            if isinstance(existing.value, list):
                items = [str(x) for x in existing.value if str(x).strip()]
            elif isinstance(existing.value, str) and existing.value.strip():
                items = [existing.value.strip()]

        if v not in items:
            items.append(v)

        return self.upsert_fact(
            subject_type=subject_type,
            subject_id=subject_id,
            key=k,
            value=items,
            tags=tags,
            confidence=confidence,
            source=source,
            evidence=evidence,
            person_ref=person_ref,
        )

    def remove_fact_value(
        self,
        *,
        subject_type: str,
        subject_id: str,
        key: str,
        value: str,
        source: str = "auto",
        evidence: str = "",
    ) -> None:
        """
        Remove one value from a JSON-list fact. If list becomes empty, delete the fact.
        """
        k = self._normalize_fact_key(key)
        v = (value or "").strip()
        if not k or not v:
            return

        existing = self.get_fact(subject_type=subject_type, subject_id=subject_id, key=k)
        if existing is None:
            return

        # Only allow removals if not lower priority than existing
        if _source_pri(source) < _source_pri(existing.source):
            logger.debug(
                "Fact remove blocked (lower priority): key=%s old_src=%s new_src=%s",
                k,
                existing.source,
                source,
            )
            return

        items: list[str] = []
        if isinstance(existing.value, list):
            items = [str(x) for x in existing.value if str(x).strip()]
        elif isinstance(existing.value, str) and existing.value.strip():
            items = [existing.value.strip()]

        items2 = [x for x in items if x != v]
        if not items2:
            self.delete_fact(subject_type=subject_type, subject_id=subject_id, key=k)
            return

        self.upsert_fact(
            subject_type=subject_type,
            subject_id=subject_id,
            key=k,
            value=items2,
            tags=existing.tags,
            confidence=existing.confidence,
            source=existing.source,
            evidence=evidence or existing.evidence,
            person_ref=existing.person_ref,
        )

    def delete_fact(self, *, subject_type: str, subject_id: str, key: str) -> None:
        k = self._normalize_fact_key(key)
        if not subject_type or not subject_id or not k:
            return
        conn = self._get_conn()
        try:
            conn.execute(
                "DELETE FROM memory_facts WHERE subject_type = ? AND subject_id = ? AND key = ?",
                (subject_type, subject_id, k),
            )
            conn.commit()
        finally:
            conn.close()

    def get_fact(self, *, subject_type: str, subject_id: str, key: str) -> FactItem | None:
        k = self._normalize_fact_key(key)
        if not subject_type or not subject_id or not k:
            return None
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT *
                FROM memory_facts
                WHERE subject_type = ? AND subject_id = ? AND key = ?
                    LIMIT 1
                """,
                (subject_type, subject_id, k),
            )
            r = cur.fetchone()
            if r is None:
                return None
            return FactItem(
                id=int(r["id"]),
                subject_type=str(r["subject_type"]),
                subject_id=str(r["subject_id"]),
                key=str(r["key"]),
                value=self._decode_fact_value(str(r["value_format"]), str(r["value"])),
                tags=self._str_to_tags(r["tags"]),
                confidence=float(r["confidence"]),
                last_updated=float(r["last_updated"]),
                source=str(r["source"]),
                evidence=str(r["evidence"] or ""),
                person_ref=r["person_ref"],
                decay_days=float(r["decay_days"] or 365.0),
                last_accessed=float(r["last_accessed"] or 0.0) or float(r["last_updated"] or 0.0),
                n_reinforced=int(r["n_reinforced"] or 0),
                pinned=int(r["pinned"] or 0),
            )
        finally:
            conn.close()

    def query_facts(
        self,
        *,
        subject_type: str | None = None,
        subject_id: str | None = None,
        min_confidence: float = 0.0,
        min_effective: float = 0.05,
        overscan: int = 6,
        now_ts: float | None = None,
        limit: int = 30,
        key_prefix: str | None = None,
        person_ref: str | None = None,
        touch: bool = False,
    ) -> list[FactItem]:
        sql = "SELECT * FROM memory_facts WHERE 1=1"
        params: list[Any] = []

        if subject_type is not None:
            sql += " AND subject_type = ?"
            params.append(subject_type)

        if subject_id is not None:
            sql += " AND subject_id = ?"
            params.append(subject_id)

        sql += " AND confidence >= ?"
        params.append(float(min_confidence))

        if key_prefix:
            kp = self._normalize_fact_key(key_prefix)
            if kp:
                sql += " AND key LIKE ?"
                params.append(kp + "%")

        if person_ref is not None:
            sql += " AND person_ref = ?"
            params.append(person_ref)

        scan = int(max(0, limit) * max(1, int(overscan)))
        sql += " ORDER BY confidence DESC, last_updated DESC LIMIT ?"
        params.append(scan)

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()

            now = float(now_ts) if now_ts is not None else time.time()

            raw_items: list[FactItem] = []
            for r in rows:
                lu = float(r["last_updated"] or 0.0)
                la = float(r["last_accessed"] or 0.0) or lu
                raw_items.append(
                    FactItem(
                        id=int(r["id"]),
                        subject_type=str(r["subject_type"]),
                        subject_id=str(r["subject_id"]),
                        key=str(r["key"]),
                        value=self._decode_fact_value(str(r["value_format"]), str(r["value"])),
                        tags=self._str_to_tags(r["tags"]),
                        confidence=float(r["confidence"]),
                        last_updated=lu,
                        source=str(r["source"]),
                        evidence=str(r["evidence"] or ""),
                        person_ref=r["person_ref"],
                        decay_days=float(r["decay_days"] or 365.0),
                        last_accessed=la,
                        n_reinforced=int(r["n_reinforced"] or 0),
                        pinned=int(r["pinned"] or 0),
                    )
                )

            scored: list[tuple[float, FactItem]] = []
            to_delete: list[int] = []
            me = float(max(0.0, min(1.0, min_effective)))

            for f in raw_items:
                eff = self._effective_score(
                    f.confidence, now, f.last_updated, f.decay_days, f.pinned
                )
                if (not f.pinned) and eff < me:
                    to_delete.append(f.id)
                    continue
                scored.append((eff, f))

            scored.sort(key=lambda x: (x[0], x[1].last_updated), reverse=True)
            result = [f for _, f in scored[: int(max(0, limit))]]

            if to_delete:
                self._delete_facts_by_ids(to_delete)

            if touch and result:
                self._touch_facts_by_ids([f.id for f in result], now_ts=now)

            return result
        finally:
            conn.close()

    @staticmethod
    def global_subject_id() -> str:
        return _GLOBAL_SUBJECT_ID
