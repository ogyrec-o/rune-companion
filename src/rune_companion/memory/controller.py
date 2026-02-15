# src/rune_companion/memory/controller.py

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from .api import (
    remember_global_fact,
    remember_relationship_fact,
    remember_room_fact,
    remember_user_fact,
)
from .store import MemoryItem
from ..core.state import AppState

logger = logging.getLogger(__name__)

MEMORY_CONTROLLER_SYSTEM_PROMPT = """
You are a memory + task planning module.

You do NOT chat with the user.
You analyze:
- recent dialog messages
- current stored memory items

Then you return a JSON plan describing what to change:
- add / update / delete memory items
- optionally add tasks (task_add) for future follow-ups

Hard rule: do NOT invent facts.
Any new fact must be supported by a direct user quote from the provided recent messages.

Evidence field (required):
- For "add": evidence is REQUIRED.
- For "update": evidence is REQUIRED if the "text" field is changed.
Evidence must be a direct substring literally present in the provided history.

Output format:
Return STRICT JSON only. No extra text. No Markdown.

If no actions are needed, return:
{ "ops": [] }
""".strip()


def _format_utc(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _normalize_text(s: str) -> str:
    return " ".join(str(s).lower().split())


def _evidence_matches_history(evidence: str, history_text: str) -> bool:
    if not evidence or not history_text:
        return False
    ev = _normalize_text(evidence)
    hist = _normalize_text(history_text)
    return bool(ev) and ev in hist


def _extract_json_object(raw: str) -> str:
    """
    Best-effort extraction of a JSON object if the model accidentally emits extra text.
    """
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        return raw[first : last + 1]
    return raw


def run_memory_controller(
        state: AppState,
        *,
        user_id: str | None,
        room_id: str | None,
        last_messages: list[dict[str, Any]],
        current_memories: list[MemoryItem],
) -> dict[str, Any] | None:
    """
    Ask the planner LLM for a memory/task update plan.

    Returns:
    - dict with key "ops" (list of operations), or None on failure.
    - Adds internal field "_history_text" used for evidence validation in apply_memory_plan().
    """
    s = getattr(state, "settings", None)
    max_history_msgs = int(getattr(s, "memory_ctrl_planner_max_history_msgs", 12))
    max_msg_chars = int(getattr(s, "memory_ctrl_planner_max_msg_chars", 500))

    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines: list[str] = [
        f"current_time_utc: {now_utc}",
        f"user_id: {user_id or 'None'}",
        f"room_id: {room_id or 'None'}",
        "",
        "Recent messages (oldest -> newest):",
    ]

    msgs = last_messages[-max_history_msgs:]
    history_chunks: list[str] = []

    for i, m in enumerate(msgs, start=1):
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))

        history_chunks.append(content)

        display = content
        if len(display) > max_msg_chars:
            display = display[:max_msg_chars] + "…"

        lines.append(f"{i}. {role}: {display}")

    history_text = "\n".join(history_chunks)

    lines.extend(["", "Current memory items (most important first):"])
    for item in current_memories:
        tag_str = ",".join(item.tags)
        ts_str = _format_utc(item.last_updated)
        text = item.text
        if len(text) > max_msg_chars:
            text = text[:max_msg_chars] + "…"
        lines.append(
            f"- id={item.id}, subject_type={item.subject_type}, subject_id={item.subject_id}, "
            f"importance={item.importance:.2f}, last_updated={ts_str}, tags=[{tag_str}]: {text}"
        )

    user_message = "\n".join(lines)

    raw = ""
    try:
        for piece in state.llm.stream_chat(
                [{"role": "user", "content": user_message}],
                MEMORY_CONTROLLER_SYSTEM_PROMPT,
        ):
            raw += piece
    except Exception:
        logger.exception("Memory controller LLM call failed.")
        return None

    raw = raw.strip()
    if not raw:
        return None

    try:
        plan = json.loads(_extract_json_object(raw))
    except Exception:
        logger.exception("Memory controller JSON parse failed. Raw=%r", raw[:2000])
        return None

    if not isinstance(plan, dict):
        return None

    plan["_history_text"] = history_text
    logger.debug("Memory controller plan ops=%s", len(plan.get("ops", []) or []))
    return plan


def apply_memory_plan(
        state: AppState,
        plan: dict[str, Any],
        *,
        default_user_id: str | None = None,
        default_room_id: str | None = None,
) -> None:
    """
    Execute operations from the plan.

    Supported ops:
    - add: store a new memory item via remember_* helpers
    - update: state.memory.update_memory(...)
    - delete: state.memory.delete_memory(...)
    - task_add: create a task via state.task_store.add_task(...)
    """
    ops = plan.get("ops")
    if not isinstance(ops, list):
        return

    history_text = plan.get("_history_text")
    if not isinstance(history_text, str):
        history_text = ""

    task_store = state.task_store

    for op in ops:
        if not isinstance(op, dict):
            continue

        op_kind = op.get("op")
        if op_kind == "add":
            subject_type = str(op.get("subject_type") or "user")
            subject_id = op.get("subject_id")
            text = op.get("text")

            if not isinstance(text, str) or not text.strip():
                continue

            evidence = op.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                logger.debug("Skipping add: missing evidence")
                continue
            if history_text and not _evidence_matches_history(evidence, history_text):
                logger.debug("Skipping add: evidence not found in history evidence=%r", evidence)
                continue

            importance = op.get("importance")
            if not isinstance(importance, (int, float)):
                importance = 0.7

            tags_raw = op.get("tags")
            tags = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else []

            if not subject_id:
                if subject_type in ("user", "relationship"):
                    subject_id = default_user_id
                elif subject_type == "room":
                    subject_id = default_room_id
                else:
                    subject_id = state.memory.global_subject_id()

            if subject_type == "user" and isinstance(subject_id, str) and subject_id:
                remember_user_fact(state, subject_id, text, importance=float(importance), tags=tags)
            elif subject_type == "room" and isinstance(subject_id, str) and subject_id:
                remember_room_fact(state, subject_id, text, importance=float(importance), tags=tags)
            elif subject_type == "relationship" and isinstance(subject_id, str) and subject_id:
                remember_relationship_fact(
                    state,
                    subject_id,
                    default_room_id,
                    text,
                    importance=float(importance),
                    tags=tags,
                )
            else:
                remember_global_fact(state, text, importance=float(importance), tags=tags)

        elif op_kind == "update":
            mem_id = op.get("id")
            if not isinstance(mem_id, int) or mem_id <= 0:
                continue

            new_text = op.get("text")
            evidence = op.get("evidence")

            if isinstance(new_text, str):
                if not isinstance(evidence, str) or not evidence.strip():
                    logger.debug("Skipping update text: missing evidence id=%s", mem_id)
                    new_text = None
                elif history_text and not _evidence_matches_history(evidence, history_text):
                    logger.debug("Skipping update text: evidence not found id=%s evidence=%r", mem_id, evidence)
                    new_text = None

            new_importance = op.get("importance")
            imp: float | None = float(new_importance) if isinstance(new_importance, (int, float)) else None

            tags_raw = op.get("tags")
            tags: list[str] | None = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else None

            person_ref = op.get("person_ref")
            if person_ref is not None and not isinstance(person_ref, str):
                person_ref = None

            state.memory.update_memory(
                mem_id,
                text=new_text if isinstance(new_text, str) else None,
                tags=tags,
                importance=imp,
                person_ref=person_ref,
            )

        elif op_kind == "delete":
            mem_id = op.get("id")
            if not isinstance(mem_id, int) or mem_id <= 0:
                continue
            state.memory.delete_memory(mem_id)

        elif op_kind == "task_add":
            kind = str(op.get("kind") or "generic")
            description = op.get("description")
            if not isinstance(description, str) or not description.strip():
                continue
            description = description.strip()

            question_text = op.get("question_text")
            if isinstance(question_text, str):
                question_text = question_text.strip() or None
            else:
                question_text = None

            run_after = op.get("run_after_minutes")
            try:
                run_after_minutes = int(run_after) if run_after is not None else 0
            except Exception:
                run_after_minutes = 0

            due_at = time.time() + max(0, run_after_minutes) * 60

            to_user_id = op.get("to_user_id") or default_user_id
            from_user_id = op.get("from_user_id") or default_user_id

            reply_to_user_id = op.get("reply_to_user_id")
            if not reply_to_user_id and from_user_id:
                reply_to_user_id = from_user_id

            task_room_id = op.get("room_id") or default_room_id

            importance_raw = op.get("importance")
            importance = float(importance_raw) if isinstance(importance_raw, (int, float)) else 0.7

            meta = op.get("meta")
            if not isinstance(meta, dict):
                meta = {}

            try:
                task_store.add_task(
                    kind=kind,
                    description=description,
                    from_user_id=from_user_id,
                    to_user_id=to_user_id,
                    reply_to_user_id=reply_to_user_id,
                    room_id=task_room_id,
                    due_at=due_at,
                    importance=importance,
                    meta=meta,
                    question_text=question_text,
                )
            except Exception:
                logger.exception("task_add failed kind=%s to=%s room=%s", kind, to_user_id, task_room_id)

        else:
            continue
