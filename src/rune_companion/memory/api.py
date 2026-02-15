# memory_api.py

from __future__ import annotations

from typing import Optional

from ..config import get_settings
from .store import MemoryItem, MemoryStore
from ..core.state import AppState

SUBJECT_USER = "user"
SUBJECT_ROOM = "room"
SUBJECT_RELATIONSHIP = "relationship"
SUBJECT_GLOBAL = "global"


def _global_id(state: AppState) -> str:
    # Central place for the reserved global subject id.
    # If you ever change it, do it here.
    return state.memory.global_subject_id() if isinstance(state.memory, MemoryStore) else "__GLOBAL__"


def _limits() -> tuple[int, int, int, int]:
    s = get_settings()
    return (
        int(getattr(s, "memory_max_user", 800)),
        int(getattr(s, "memory_max_room", 400)),
        int(getattr(s, "memory_max_rel", 600)),
        int(getattr(s, "memory_max_global", 800)),
    )


def remember_user_fact(
        state: AppState,
        user_id: str,
        text: str,
        *,
        importance: float = 0.9,
        tags: Optional[list[str]] = None,
        person_ref: str | None = None,
        source: str = "auto",
) -> int:
    """Store a fact about a specific user."""
    if not user_id:
        raise ValueError("user_id is required")
    if not person_ref:
        person_ref = f"user:{user_id}"

    max_user, _, _, _ = _limits()

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
        tags: Optional[list[str]] = None,
        person_ref: str | None = None,
        source: str = "auto",
) -> int:
    """Store a fact about a specific room/chat."""
    if not room_id:
        raise ValueError("room_id is required")

    _, max_room, _, _ = _limits()

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
        tags: Optional[list[str]] = None,
        person_ref: str | None = None,
        source: str = "auto",
) -> int:
    """Store relationship-context facts tied to a user_id."""
    if not user_id:
        raise ValueError("user_id is required")
    if not person_ref:
        person_ref = f"user:{user_id}"

    _, _, max_rel, _ = _limits()

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
        tags: Optional[list[str]] = None,
        person_ref: str | None = None,
        source: str = "auto",
) -> int:
    """Store a global fact not tied to a specific user or room."""
    _, _, _, max_global = _limits()
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


def get_top_user_memories(state: AppState, user_id: str | None, *, limit: int = 10) -> list[MemoryItem]:
    if not user_id:
        return []
    return state.memory.query_memory(subject_type=SUBJECT_USER, subject_id=user_id, limit=limit)


def get_top_room_memories(state: AppState, room_id: str | None, *, limit: int = 10) -> list[MemoryItem]:
    if not room_id:
        return []
    return state.memory.query_memory(subject_type=SUBJECT_ROOM, subject_id=room_id, limit=limit)


def get_top_relationship_memories(
        state: AppState,
        user_id: str | None,
        room_id: str | None,  # currently unused; kept for future
        *,
        limit: int = 10,
) -> list[MemoryItem]:
    if not user_id:
        return []
    return state.memory.query_memory(subject_type=SUBJECT_RELATIONSHIP, subject_id=user_id, limit=limit)


def get_global_memories(state: AppState, *, limit: int = 10) -> list[MemoryItem]:
    gid = _global_id(state)
    return state.memory.query_memory(subject_type=SUBJECT_GLOBAL, subject_id=gid, limit=limit)


def get_global_userstories(state: AppState, *, limit: int = 10) -> list[MemoryItem]:
    """
    Global memories tagged as "other_user" (stories about other people).
    Kept separate from general global notes.
    """
    gid = _global_id(state)
    return state.memory.query_memory(subject_type=SUBJECT_GLOBAL, subject_id=gid, tag="other_user", limit=limit)
