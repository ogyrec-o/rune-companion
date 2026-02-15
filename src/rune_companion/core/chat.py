# src/rune_companion/core/chat.py

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable

import openai

from ..config import get_settings
from ..llm.client import stream_chat_chunks, friendly_llm_error_message
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
from ..tasks.task_api import maybe_handle_reply

logger = logging.getLogger(__name__)

SENTENCE_REGEX = re.compile(r"[\.!\?…]")

# Per dialog (user_id||room_id) counters for episodic summarization and memory controller.
_EPISODE_COUNTERS: dict[str, int] = {}
_MEMORY_CTRL_COUNTERS: dict[str, int] = {}


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
    settings = get_settings()
    key = _make_key(user_id, room_id)
    if key is None or not state.save_history:
        return

    threshold = int(getattr(settings, "memory_episode_threshold_messages", 20))
    chunk_size = int(getattr(settings, "memory_episode_chunk_messages", 24))

    cnt = _EPISODE_COUNTERS.get(key, 0) + 1
    _EPISODE_COUNTERS[key] = cnt
    if cnt < threshold:
        return

    _EPISODE_COUNTERS[key] = 0

    chunk = messages_for_llm[-chunk_size:]
    if not chunk:
        return

    logger.info("Episode summary trigger key=%s chunk_len=%d", key, len(chunk))
    summary = summarize_dialog_chunk(state, chunk)
    if not summary:
        return

    # Avoid polluting memory if summarizer says there are no facts.
    s = summary.strip().lower()
    if s.startswith(("no significant facts", "нет значимых фактов")):
        logger.info("Episode summary skipped (no significant facts).")
        return

    # Local imports to avoid cycles
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
    settings = get_settings()
    key = _make_key(user_id, room_id)
    if key is None or not state.save_history:
        return

    every_n = int(getattr(settings, "memory_ctrl_every_n_messages", 1))
    last_n = int(getattr(settings, "memory_ctrl_last_messages", 8))

    cnt = _MEMORY_CTRL_COUNTERS.get(key, 0) + 1
    _MEMORY_CTRL_COUNTERS[key] = cnt
    if cnt < every_n:
        return

    _MEMORY_CTRL_COUNTERS[key] = 0

    last_messages = messages_for_llm[-last_n:]

    # Collect current memories for the controller prompt.
    current_memories: list[Any] = []

    limit_user = int(getattr(settings, "memory_prompt_limit_user", 12))
    limit_rel = int(getattr(settings, "memory_prompt_limit_rel", 12))
    limit_room = int(getattr(settings, "memory_prompt_limit_room", 8))
    limit_global = int(getattr(settings, "memory_prompt_limit_global", 8))

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


def chat_with_streaming_console(state: AppState, user_text: str) -> None:
    """
    Console mode:
    - streams response to stdout
    - optionally speaks sentences via TTS
    - uses a single global history: state.conversation
    """
    def _ts_local() -> str:
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

    settings = get_settings()
    app_name = str(getattr(settings, "app_name", "rune"))

    if state.save_history:
        messages_for_llm = [*state.conversation, {"role": "user", "content": user_text}]
    else:
        messages_for_llm = [{"role": "user", "content": user_text}]

    buffer = ""
    assistant_full = ""
    system_prompt = get_system_prompt(state.tts_enabled)

    printed_prefix = False

    try:
        for piece in stream_chat_chunks(messages_for_llm, system_prompt):
            if piece:
                if not printed_prefix:
                    print(f"[{_ts_local()}] <<< {app_name}: ", end="", flush=True)
                    printed_prefix = True

                print(piece, end="", flush=True)
                assistant_full += piece
                buffer += piece

                if state.tts_enabled:
                    while True:
                        m = SENTENCE_REGEX.search(buffer)
                        if not m:
                            break
                        sentence = buffer[: m.end()].strip()
                        buffer = buffer[m.end() :]
                        if sentence:
                            state.tts_engine.speak_sentence(sentence)

    except openai.RateLimitError as e:
        logger.warning("LLM rate limit: %s", e)
        print(f"[{_ts_local()}] [LLM] Rate-limited by the provider. Please try again shortly.")
        return
    except RuntimeError as e:
        msg = friendly_llm_error_message(e)
        logger.info("LLM runtime error: %s", msg)
        print(f"[{_ts_local()}] [LLM] {msg}")
        return
    except Exception:
        logger.exception("Unexpected error while streaming LLM output.")
        print(f"[{_ts_local()}] [LLM] Internal error while generating a reply.")
        return

    # If model produced nothing at all, give a clear message.
    if not printed_prefix:
        print(f"[{_ts_local()}] [LLM] No output (model produced no content).")
        return

    if state.tts_enabled:
        tail = buffer.strip()
        if tail:
            state.tts_engine.speak_sentence(tail)

        try:
            state.tts_engine.wait_all()
        except Exception:
            logger.debug("TTS wait_all failed.", exc_info=True)

        print(f"\n[{_ts_local()}] [LLM] Audio done.\n")
    else:
        print("\n")

    assistant_full = assistant_full.strip()
    if state.save_history and assistant_full:
        state.conversation.append({"role": "user", "content": user_text})
        state.conversation.append({"role": "assistant", "content": assistant_full})


def _collect_open_tasks_for_user(state: AppState, user_id: str | None) -> list[Any]:
    """Fetch open tasks for a given user_id (best-effort)."""
    if not user_id:
        return []
    task_store = getattr(state, "task_store", None)
    if task_store is None:
        return []
    try:
        return task_store.list_open_tasks_for_user(user_id, limit=16)
    except Exception:
        logger.exception("list_open_tasks_for_user failed (user_id=%s)", user_id)
        return []


def generate_reply_text(
        state: AppState,
        user_text: str,
        *,
        user_id: str | None = None,
        room_id: str | None = None,
) -> str:
    """
    Chat mode for connectors (Matrix/Discord/etc).

    Key behaviors:
    - per-(user_id, room_id) dialog history stored in state.dialog_histories
    - episodic summaries and memory controller can update MemoryStore / TaskStore
    - inject relevant memories and open tasks into the system prompt
    """
    settings = get_settings()

    # 0) If this might be a reply to a previously created task, handle it.
    if user_id and room_id:
        try:
            maybe_handle_reply(state, user_id, room_id, user_text)
        except Exception:
            logger.exception("maybe_handle_reply failed (user_id=%s room_id=%s)", user_id, room_id)

    # 1) Build history (per user/room) if enabled.
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

    # 1.5) Episodic summaries.
    _maybe_run_episode_summary(state, user_id, room_id, messages_for_llm)

    # 1.7) Memory controller.
    _maybe_run_memory_controller(state, user_id, room_id, messages_for_llm)

    # 2) Query relevant memories (guard against None IDs).
    limit_user = int(getattr(settings, "memory_prompt_limit_user", 12))
    limit_rel = int(getattr(settings, "memory_prompt_limit_rel", 12))
    limit_room = int(getattr(settings, "memory_prompt_limit_room", 8))
    limit_global = int(getattr(settings, "memory_prompt_limit_global", 8))
    limit_userstories = int(getattr(settings, "memory_prompt_limit_global_userstories", 8))

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
            # Keep "stories about other people" separate.
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

    # 3) System prompt + injected memory.
    system_prompt = get_system_prompt(state.tts_enabled)
    if memory_block:
        system_prompt = f"{system_prompt.strip()}\n\n{memory_block}\n"
        logger.debug("Injected memory block (%d chars).", len(memory_block))

    assistant_full = ""
    try:
        for piece in stream_chat_chunks(messages_for_llm, system_prompt):
            assistant_full += piece
    except openai.RateLimitError:
        return "The service is temporarily rate-limited. Please try again shortly."
    except RuntimeError as e:
        # Friendly config/runtime error (e.g. missing API key) — safe to show to user.
        msg = friendly_llm_error_message(e)
        logger.info("LLM runtime error: %s", msg)
        return msg
    except Exception:
        logger.exception("LLM generation failed.")
        return "Internal error while generating a reply. Please try again."

    assistant_full = assistant_full.strip()

    # 4) Update dialog history.
    if state.save_history and assistant_full and history is not None:
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_full})

        max_msgs = int(getattr(settings, "memory_max_dialog_messages", 80))
        if len(history) > max_msgs:
            overflow = len(history) - max_msgs
            del history[:overflow]

    return assistant_full
