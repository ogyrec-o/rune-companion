# src/rune_companion/core/chat.py

from __future__ import annotations

"""
Core chat orchestration.

This module is transport-agnostic:
- connectors provide inbound text + optional (user_id, room_id),
- the core builds prompts, injects memory/tasks, and streams LLM output,
- connectors decide how to display/send the stream (console, Matrix, etc.).

Key invariants:
- history is updated only after a successful stream completion (to avoid partial saves),
- memory/tasks are injected inside a single <MEMORY>...</MEMORY> block to keep the prompt structure stable.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

from ..memory.api import (
    get_global_memories,
    get_global_userstories,
    get_top_relationship_memories,
    get_top_room_memories,
    get_top_user_memories,
)
from ..memory.controller import apply_memory_plan, run_memory_controller
from ..memory.summarizer import summarize_dialog_chunk
from .persona import get_system_prompt
from .state import AppState

logger = logging.getLogger(__name__)


def _make_key(user_id: str | None, room_id: str | None) -> str | None:
    """Build a stable key for per-dialog state (counters, histories)."""
    if not user_id and not room_id:
        return None
    return f"{user_id or ''}||{room_id or ''}"


def _format_utc_ts(ts: float) -> str:
    """Format UNIX timestamp into a human-readable UTC string."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _sanitize_for_memory_block(text: str) -> str:
    """
    Prevent accidental prompt structure breaks.

    The memory block is wrapped into <MEMORY>...</MEMORY>. If a stored memory line
    contains these tags, it may break the prompt structure. We neutralize them.
    """
    if not text:
        return ""
    out = text.replace("\x00", "")
    out = out.replace("<MEMORY>", "[MEMORY]").replace("</MEMORY>", "[/MEMORY]")
    return out


def _markers_from_tags(tags: Iterable[str]) -> list[str]:
    """Map selected memory tags to visible markers for the model."""
    t = set(tags or [])
    markers: list[str] = []
    if "todo" in t:
        markers.append("TODO")
    if "request_from_this_user" in t:
        markers.append("REQ_FROM_THIS")
    if "request_from_other_user" in t:
        markers.append("REQ_FROM_OTHER")
    if "request_for_other_user" in t:
        markers.append("REQ_FOR_OTHER")
    if "promise" in t:
        markers.append("PROMISE")
    return markers


def _format_mem_line(m: Any) -> str:
    """
    Format a single memory line for prompt injection.

    Some tags are elevated to markers so the model notices promises/requests.
    """
    ts = _format_utc_ts(float(getattr(m, "last_updated", 0.0)))
    tags = getattr(m, "tags", []) or []
    markers = _markers_from_tags(tags)
    marker_str = f"[{','.join(markers)}] " if markers else ""
    text = _sanitize_for_memory_block(str(getattr(m, "text", "")))
    return f"- ({ts}) {marker_str}{text}"


def _format_task_line(t: Any) -> str:
    """Format a TaskStore item for prompt injection."""
    due_str = "any time"
    due_at = getattr(t, "due_at", None)
    if due_at is not None:
        dt = datetime.fromtimestamp(float(due_at), tz=timezone.utc)
        due_str = dt.strftime("%Y-%m-%d %H:%M UTC")

    who: list[str] = []
    from_user_id = getattr(t, "from_user_id", None)
    to_user_id = getattr(t, "to_user_id", None)
    reply_to = getattr(t, "reply_to_user_id", None)

    if from_user_id:
        who.append(f"from: {from_user_id}")
    if to_user_id:
        who.append(f"to: {to_user_id}")
    if reply_to and reply_to not in (from_user_id, to_user_id):
        who.append(f"reply_to: {reply_to}")

    who_str = (" (" + ", ".join(who) + ")") if who else ""
    desc = _sanitize_for_memory_block(str(getattr(t, "description", "")))
    kind = _sanitize_for_memory_block(str(getattr(t, "kind", "")))
    task_id = getattr(t, "id", "?")
    return f"- [TASK {task_id}] [{kind}] due {due_str}{who_str}: {desc}"


def _build_memory_block(*, sections: list[str]) -> str:
    """
    Build a single <MEMORY>...</MEMORY> block.

    Critical: everything must be inside the tag, otherwise the model may ignore it
    or treat it as normal instructions/text.
    """
    if not sections:
        return ""

    header = (
        "<MEMORY>\n"
        "This is internal memory (facts + tasks). Use it to:\n"
        "- keep entities (people/rooms/events) consistent;\n"
        "- honor promises and user requests;\n"
        "- proactively follow up when appropriate (ask/clarify/remind).\n"
        "Markers [TODO], [REQ_FROM_THIS], [REQ_FROM_OTHER], [REQ_FOR_OTHER], [PROMISE]\n"
        "indicate outstanding tasks/requests/promises.\n\n"
    )
    body = "\n\n".join(sections)
    return f"{header}{body}\n</MEMORY>"


def _maybe_capture_task_reply(state: AppState, user_id: str | None, room_id: str | None, user_text: str) -> None:
    """
    If message looks like an answer to a previously created ask_user* task,
    store it and mark task as ANSWER_RECEIVED so scheduler can do phase-2.
    """
    if not user_id or not room_id:
        return

    try:
        task = state.task_store.find_waiting_ask_task(to_user_id=user_id, room_id=room_id)
    except Exception:
        logger.exception("find_waiting_ask_task failed (user_id=%s room_id=%s)", user_id, room_id)
        return

    if not task:
        return

    try:
        now_ts = datetime.now(timezone.utc).timestamp()
        state.task_store.save_answer_and_mark_received(task.id, user_text, now_ts)  # type: ignore[attr-defined]
        logger.info("Captured answer for task_id=%s user=%s room=%s", getattr(task, "id", "?"), user_id, room_id)
    except Exception:
        logger.exception("save_answer_and_mark_received failed task_id=%s", getattr(task, "id", "?"))


def _maybe_run_episode_summary(
        state: AppState,
        user_id: str | None,
        room_id: str | None,
        messages_for_llm: list[dict[str, str]],
) -> None:
    """
    If enough messages accumulated in this dialog, ask the summarizer model
    to produce an episodic summary and store it as memory.
    """
    if not state.save_history:
        return

    key = _make_key(user_id, room_id)
    if key is None:
        return

    s = state.settings
    threshold = int(getattr(s, "memory_episode_threshold_messages", 20))
    chunk_size = int(getattr(s, "memory_episode_chunk_messages", 24))

    cnt = state.episode_counters.get(key, 0) + 1
    state.episode_counters[key] = cnt
    if cnt < threshold:
        return

    state.episode_counters[key] = 0

    chunk = messages_for_llm[-chunk_size:]
    if not chunk:
        return

    logger.info("Episode summary trigger key=%s chunk_len=%d", key, len(chunk))
    summary = summarize_dialog_chunk(state, chunk)
    if not summary:
        return

    ssum = summary.strip().lower()
    if ssum.startswith(("no significant facts", "нет значимых фактов")):
        logger.info("Episode summary skipped (no significant facts).")
        return

    # local import to avoid cycles
    from ..memory.api import remember_relationship_fact, remember_room_fact

    tags = ["episode", "summary"]

    if user_id:
        mem_id = remember_relationship_fact(
            state,
            user_id,
            room_id,
            summary,
            importance=0.8,
            tags=tags,
        )
        logger.info("Episode summary stored (relationship) mem_id=%s", mem_id)

    if room_id:
        mem_id = remember_room_fact(
            state,
            room_id,
            summary,
            importance=0.7,
            tags=tags,
        )
        logger.info("Episode summary stored (room) mem_id=%s", mem_id)


def _maybe_run_memory_controller(
        state: AppState,
        user_id: str | None,
        room_id: str | None,
        messages_for_llm: list[dict[str, str]],
) -> None:
    """
    Memory controller:
    - looks at recent messages + existing memories
    - decides which facts to add/update/delete
    - can also create tasks
    """
    if not state.save_history:
        return

    key = _make_key(user_id, room_id)
    if key is None:
        return

    s = state.settings
    every_n = int(getattr(s, "memory_ctrl_every_n_messages", 1))
    last_n = int(getattr(s, "memory_ctrl_last_messages", 8))

    cnt = state.memory_ctrl_counters.get(key, 0) + 1
    state.memory_ctrl_counters[key] = cnt
    if cnt < every_n:
        return

    state.memory_ctrl_counters[key] = 0

    last_messages = messages_for_llm[-last_n:]

    # Collect current memories for the controller prompt.
    current_memories: list[Any] = []

    limit_user = int(getattr(s, "memory_prompt_limit_user", 12))
    limit_rel = int(getattr(s, "memory_prompt_limit_rel", 12))
    limit_room = int(getattr(s, "memory_prompt_limit_room", 8))
    limit_global = int(getattr(s, "memory_prompt_limit_global", 8))

    if user_id:
        current_memories.extend(get_top_user_memories(state, user_id, limit=limit_user))
        current_memories.extend(get_top_relationship_memories(state, user_id, room_id, limit=limit_rel))
    if room_id:
        current_memories.extend(get_top_room_memories(state, room_id, limit=limit_room))

    current_memories.extend(get_global_memories(state, limit=limit_global))

    logger.info(
        "Memory controller trigger key=%s last_messages=%d mems=%d",
        key,
        len(last_messages),
        len(current_memories),
    )

    plan = run_memory_controller(
        state,
        user_id=user_id,
        room_id=room_id,
        last_messages=last_messages,
        current_memories=current_memories,
    )
    if not plan:
        return

    apply_memory_plan(state, plan, default_user_id=user_id, default_room_id=room_id)


def _collect_open_tasks_for_user(state: AppState, user_id: str | None) -> list[Any]:
    """Fetch open tasks for a given user_id (best-effort)."""
    if not user_id:
        return []
    try:
        return state.task_store.list_open_tasks_for_user(user_id, limit=16)
    except Exception:
        logger.exception("list_open_tasks_for_user failed (user_id=%s)", user_id)
        return []


def stream_reply(
        state: AppState,
        user_text: str,
        *,
        user_id: str | None = None,
        room_id: str | None = None,
) -> Iterable[str]:
    """
    Transport-neutral streaming API.

    - Yields assistant text chunks as they arrive from the LLM client.
    - Optionally persists per-dialog history (when save_history is enabled).
    - Optionally runs episodic summarization + memory controller (which may create tasks).
    - Captures user replies to ask-user tasks before generating a new response.

    History is appended only if streaming completes successfully, to avoid storing partial assistant outputs.
    """
    # If the user is answering a previously asked question (ask_user* task),
    # capture it first so the scheduler can run phase-2 before we generate a new reply.
    _maybe_capture_task_reply(state, user_id, room_id, user_text)

    # Build history.
    history: list[dict[str, str]] | None = None
    if state.save_history:
        key = _make_key(user_id, room_id)
        if key is not None:
            history = state.dialog_histories.setdefault(key, [])
        else:
            history = state.conversation
        messages_for_llm = [*history, {"role": "user", "content": user_text}]
    else:
        messages_for_llm = [{"role": "user", "content": user_text}]

    # Episodic summaries.
    _maybe_run_episode_summary(state, user_id, room_id, messages_for_llm)

    # Memory controller.
    _maybe_run_memory_controller(state, user_id, room_id, messages_for_llm)

    # Query relevant memories.
    s = state.settings
    limit_user = int(getattr(s, "memory_prompt_limit_user", 12))
    limit_rel = int(getattr(s, "memory_prompt_limit_rel", 12))
    limit_room = int(getattr(s, "memory_prompt_limit_room", 8))
    limit_global = int(getattr(s, "memory_prompt_limit_global", 8))
    limit_userstories = int(getattr(s, "memory_prompt_limit_global_userstories", 8))

    user_mems = get_top_user_memories(state, user_id, limit=limit_user) if user_id else []
    room_mems = get_top_room_memories(state, room_id, limit=limit_room) if room_id else []
    rel_mems = get_top_relationship_memories(state, user_id, room_id, limit=limit_rel) if user_id else []
    global_mems = get_global_memories(state, limit=limit_global)
    global_userstories = get_global_userstories(state, limit=limit_userstories)

    open_tasks = _collect_open_tasks_for_user(state, user_id)

    sections: list[str] = []

    if user_mems:
        header = f"About the current user ({user_id}):"
        lines = [header, *(_format_mem_line(m) for m in user_mems)]
        sections.append("\n".join(lines))

    if rel_mems:
        header = f"About the relationship / prior dialogs with ({user_id}):"
        lines = [header, *(_format_mem_line(m) for m in rel_mems)]
        sections.append("\n".join(lines))

    if room_mems:
        header = f"About this room/chat ({room_id}):"
        lines = [header, *(_format_mem_line(m) for m in room_mems)]
        sections.append("\n".join(lines))

    if global_mems:
        lines = ["General notes (global memory):"]
        for m in global_mems:
            if "other_user" in (getattr(m, "tags", None) or []):
                continue
            lines.append(_format_mem_line(m))
        if len(lines) > 1:
            sections.append("\n".join(lines))

    if global_userstories:
        lines = ["Notes about other people (stories about friends/acquaintances):"]
        lines.extend(_format_mem_line(m) for m in global_userstories)
        sections.append("\n".join(lines))

    if open_tasks:
        lines = ["Open tasks / promises related to this user:"]
        lines.extend(_format_task_line(t) for t in open_tasks)
        sections.append("\n".join(lines))

    memory_block = _build_memory_block(sections=sections)

    # System prompt + injected memory.
    system_prompt = get_system_prompt(state.tts_enabled)
    if memory_block:
        system_prompt = f"{system_prompt.strip()}\n\n{memory_block}\n"

    # Stream from LLM and collect assistant text to update history at the end.
    # We only persist history on a clean completion; interrupted streams should not poison saved dialogs.
    assistant_full = ""
    completed = False
    try:
        for piece in state.llm.stream_chat(messages_for_llm, system_prompt):
            if piece:
                assistant_full += piece
                yield piece
        completed = True
    finally:
        assistant_full = (assistant_full or "").strip()

        if completed and state.save_history and assistant_full and history is not None:
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": assistant_full})

            max_msgs = int(getattr(s, "memory_max_dialog_messages", 80))
            if len(history) > max_msgs:
                overflow = len(history) - max_msgs
                del history[:overflow]


def generate_reply_text(
        state: AppState,
        user_text: str,
        *,
        user_id: str | None = None,
        room_id: str | None = None,
) -> str:
    """
    Non-streaming helper for connectors that want a full string.
    """
    out = ""
    for piece in stream_reply(state, user_text, user_id=user_id, room_id=room_id):
        out += piece
    return out.strip()
