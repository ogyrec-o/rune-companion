# src/rune_companion/memory/api.py

from __future__ import annotations

from typing import Any

from ..core.state import AppState
from .store import FactItem, MemoryItem

SUBJECT_USER = "user"
SUBJECT_ROOM = "room"
SUBJECT_RELATIONSHIP = "relationship"
SUBJECT_GLOBAL = "global"


def _global_id(state: AppState) -> str:
    try:
        return state.memory.global_subject_id()
    except Exception:
        return "__GLOBAL__"


def _limits(state: AppState) -> tuple[int, int, int, int]:
    s = getattr(state, "settings", None)
    return (
        int(getattr(s, "memory_max_user", 800)),
        int(getattr(s, "memory_max_room", 400)),
        int(getattr(s, "memory_max_rel", 600)),
        int(getattr(s, "memory_max_global", 800)),
    )


# -------------------------
# Unstructured memories
# -------------------------


def remember_user_fact(
    state: AppState,
    user_id: str,
    text: str,
    *,
    importance: float = 0.9,
    tags: list[str] | None = None,
    person_ref: str | None = None,
    source: str = "auto",
) -> int:
    if not user_id:
        raise ValueError("user_id is required")
    if not person_ref:
        person_ref = f"user:{user_id}"

    max_user, _, _, _ = _limits(state)

    mem_id = state.memory.add_memory(
        subject_type=SUBJECT_USER,
        subject_id=user_id,
        text=text,
        importance=importance,
        tags=tags or [],
        source=source,
        person_ref=person_ref,
    )
    state.memory.prune_subject(SUBJECT_USER, user_id, max_user)
    return mem_id


def remember_room_fact(
    state: AppState,
    room_id: str,
    text: str,
    *,
    importance: float = 0.6,
    tags: list[str] | None = None,
    person_ref: str | None = None,
    source: str = "auto",
) -> int:
    if not room_id:
        raise ValueError("room_id is required")

    _, max_room, _, _ = _limits(state)

    mem_id = state.memory.add_memory(
        subject_type=SUBJECT_ROOM,
        subject_id=room_id,
        text=text,
        importance=importance,
        tags=tags or [],
        source=source,
        person_ref=person_ref,
    )
    state.memory.prune_subject(SUBJECT_ROOM, room_id, max_room)
    return mem_id


def remember_relationship_fact(
    state: AppState,
    user_id: str,
    room_id: str | None,  # kept for future extensibility
    text: str,
    *,
    importance: float = 0.7,
    tags: list[str] | None = None,
    person_ref: str | None = None,
    source: str = "auto",
) -> int:
    if not user_id:
        raise ValueError("user_id is required")
    if not person_ref:
        person_ref = f"user:{user_id}"

    _, _, max_rel, _ = _limits(state)

    mem_id = state.memory.add_memory(
        subject_type=SUBJECT_RELATIONSHIP,
        subject_id=user_id,
        text=text,
        importance=importance,
        tags=tags or [],
        source=source,
        person_ref=person_ref,
    )
    state.memory.prune_subject(SUBJECT_RELATIONSHIP, user_id, max_rel)
    return mem_id


def remember_global_fact(
    state: AppState,
    text: str,
    *,
    importance: float = 0.6,
    tags: list[str] | None = None,
    person_ref: str | None = None,
    source: str = "auto",
) -> int:
    _, _, _, max_global = _limits(state)
    gid = _global_id(state)

    mem_id = state.memory.add_memory(
        subject_type=SUBJECT_GLOBAL,
        subject_id=gid,
        text=text,
        importance=importance,
        tags=tags or [],
        source=source,
        person_ref=person_ref,
    )
    state.memory.prune_subject(SUBJECT_GLOBAL, gid, max_global)
    return mem_id


def get_top_user_memories(
    state: AppState, user_id: str | None, *, limit: int = 10
) -> list[MemoryItem]:
    if not user_id:
        return []
    return state.memory.query_memory(
        subject_type=SUBJECT_USER, subject_id=user_id, limit=limit, touch=True
    )


def get_top_room_memories(
    state: AppState, room_id: str | None, *, limit: int = 10
) -> list[MemoryItem]:
    if not room_id:
        return []
    return state.memory.query_memory(
        subject_type=SUBJECT_ROOM, subject_id=room_id, limit=limit, touch=True
    )


def get_top_relationship_memories(
    state: AppState,
    user_id: str | None,
    room_id: str | None,  # currently unused; kept for future
    *,
    limit: int = 10,
) -> list[MemoryItem]:
    if not user_id:
        return []
    return state.memory.query_memory(
        subject_type=SUBJECT_RELATIONSHIP, subject_id=user_id, limit=limit, touch=True
    )


def get_global_memories(state: AppState, *, limit: int = 10) -> list[MemoryItem]:
    gid = _global_id(state)
    return state.memory.query_memory(
        subject_type=SUBJECT_GLOBAL, subject_id=gid, limit=limit, touch=True
    )


def get_global_userstories(state: AppState, *, limit: int = 10) -> list[MemoryItem]:
    gid = _global_id(state)
    return state.memory.query_memory(
        subject_type=SUBJECT_GLOBAL, subject_id=gid, tag="other_user", limit=limit, touch=True
    )


# -------------------------
# Structured facts (slots)
# -------------------------


def set_fact(
    state: AppState,
    *,
    subject_type: str,
    subject_id: str,
    key: str,
    value: Any,
    confidence: float = 0.85,
    tags: list[str] | None = None,
    source: str = "auto",
    evidence: str = "",
    person_ref: str | None = None,
) -> int:
    return state.memory.upsert_fact(
        subject_type=subject_type,
        subject_id=subject_id,
        key=key,
        value=value,
        tags=tags or [],
        confidence=confidence,
        source=source,
        evidence=evidence,
        person_ref=person_ref,
    )


def add_fact_value(
    state: AppState,
    *,
    subject_type: str,
    subject_id: str,
    key: str,
    value: str,
    confidence: float = 0.75,
    tags: list[str] | None = None,
    source: str = "auto",
    evidence: str = "",
    person_ref: str | None = None,
) -> int:
    return state.memory.add_fact_value(
        subject_type=subject_type,
        subject_id=subject_id,
        key=key,
        value=value,
        tags=tags or [],
        confidence=confidence,
        source=source,
        evidence=evidence,
        person_ref=person_ref,
    )


def remove_fact_value(
    state: AppState,
    *,
    subject_type: str,
    subject_id: str,
    key: str,
    value: str,
    source: str = "auto",
    evidence: str = "",
) -> None:
    state.memory.remove_fact_value(
        subject_type=subject_type,
        subject_id=subject_id,
        key=key,
        value=value,
        source=source,
        evidence=evidence,
    )


def delete_fact(state: AppState, *, subject_type: str, subject_id: str, key: str) -> None:
    state.memory.delete_fact(subject_type=subject_type, subject_id=subject_id, key=key)


def get_top_user_facts(state: AppState, user_id: str | None, *, limit: int = 12) -> list[FactItem]:
    if not user_id:
        return []
    return state.memory.query_facts(
        subject_type=SUBJECT_USER, subject_id=user_id, limit=limit, touch=True
    )


def get_top_room_facts(state: AppState, room_id: str | None, *, limit: int = 12) -> list[FactItem]:
    if not room_id:
        return []
    return state.memory.query_facts(
        subject_type=SUBJECT_ROOM, subject_id=room_id, limit=limit, touch=True
    )


def get_top_relationship_facts(
    state: AppState, user_id: str | None, *, limit: int = 12
) -> list[FactItem]:
    if not user_id:
        return []
    return state.memory.query_facts(
        subject_type=SUBJECT_RELATIONSHIP, subject_id=user_id, limit=limit, touch=True
    )


def get_global_facts(state: AppState, *, limit: int = 12) -> list[FactItem]:
    gid = _global_id(state)
    return state.memory.query_facts(
        subject_type=SUBJECT_GLOBAL, subject_id=gid, limit=limit, touch=True
    )
