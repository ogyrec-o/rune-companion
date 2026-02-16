# src/rune_companion/memory/controller.py

"""
Memory/task planner.

Giants-grade:
- strict schema but resilient to aliases,
- evidence-gated writes,
- facts ("slots") + unstructured memories,
- safe-by-default (no secrets),
- no crashes on bad plan.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
from datetime import UTC, datetime
from typing import Any

from ..core.ports import ChatMessage
from ..core.state import AppState
from .api import (
    add_fact_value,
    delete_fact,
    remember_global_fact,
    remember_relationship_fact,
    remember_room_fact,
    remember_user_fact,
    remove_fact_value,
    set_fact,
)
from .store import FactItem, MemoryItem

logger = logging.getLogger(__name__)

MEMORY_CONTROLLER_SYSTEM_PROMPT = """
You are a memory + task planning module.

You do NOT chat with the user.

You analyze:
- recent dialog messages
- current stored structured facts (facts/slots)
- current stored unstructured memory items (notes)

Then you return a JSON plan describing what to change.

Hard rule: do NOT invent facts.
Any new/updated fact must be supported by a direct user quote from the provided recent messages.

Evidence field (required for writes):
- For memory "add": evidence REQUIRED.
- For memory "update" changing text: evidence REQUIRED.
- For facts "fact_set"/"fact_add_value"/"fact_remove_value": evidence REQUIRED.
Evidence must be a direct substring from the provided history (case/whitespace-insensitive match).

STRICT schema:
- plan: { "ops": [ ... ] }
- op must be one of:
  - "add", "update", "delete"                    (unstructured memories)
  - "fact_set", "fact_add_value", "fact_remove_value", "fact_delete"
  - "task_add"

Unstructured memory ops:
- "add" requires: subject_type, text, evidence. subject_id optional.
- "update" requires: id. evidence required if changing "text".
- "delete" requires: id.

Facts ops (structured slots):
- "fact_set" requires: subject_type, key, value, evidence. subject_id optional.
- "fact_add_value" requires: subject_type, key, value, evidence. subject_id optional.
- "fact_remove_value" requires: subject_type, key, value, evidence. subject_id optional.
- "fact_delete" requires: subject_type, key. subject_id optional.
  (evidence optional, prefer to include)

Facts notes:
- prefer canonical keys like: preferred_name, age, language, timezone,
  likes, dislikes, goals, projects.
- avoid duplicates: update existing key instead of adding repeated notes.
- avoid storing secrets/credentials.

Tasks:
- "task_add" requires: kind, description (others optional).

Output format:
Return STRICT JSON only. No extra text. No Markdown.

If no actions are needed, return:
{ "ops": [] }
""".strip()


def _format_utc(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=UTC)
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
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        return raw[first : last + 1]
    return raw


def _norm_op_kind(v: Any) -> str:
    s = str(v or "").strip().lower()
    aliases = {
        "add_memory": "add",
        "memory_add": "add",
        "remember": "add",
        "update_memory": "update",
        "memory_update": "update",
        "delete_memory": "delete",
        "memory_delete": "delete",
        "forget": "delete",
        "add_task": "task_add",
        "task_create": "task_add",
        "task": "task_add",
        "slot_set": "fact_set",
        "fact_add": "fact_add_value",
        "slot_add": "fact_add_value",
        "slot_remove": "fact_remove_value",
        "fact_remove": "fact_remove_value",
    }
    return aliases.get(s, s)


def _norm_subject_type(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in {"chat", "channel"}:
        return "room"
    if s in {"rel", "dialog"}:
        return "relationship"
    if s in {"general", "system"}:
        return "global"
    if s not in {"user", "room", "relationship", "global"}:
        return "user"
    return s


def _looks_like_secret(text: str) -> bool:
    t = (text or "").lower()
    return any(
        kw in t
        for kw in (
            "password",
            "пароль",
            "api key",
            "apikey",
            "secret",
            "token",
            "ключ",
            "private key",
            "seed phrase",
            "mnemonic",
        )
    )


def _safe_add_task(task_store: Any, payload: dict[str, Any]) -> None:
    add_fn = getattr(task_store, "add_task", None)
    if not callable(add_fn):
        return

    kind = str(payload.get("kind") or "").strip()
    desc = str(payload.get("description") or "").strip()
    if not kind or not desc:
        return

    kwargs = dict(payload)
    kwargs["kind"] = kind
    kwargs["description"] = desc

    try:
        sig = inspect.signature(add_fn)
        allowed = set(sig.parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        pass

    add_fn(**kwargs)


def run_memory_controller(
    state: AppState,
    *,
    user_id: str | None,
    room_id: str | None,
    last_messages: list[ChatMessage],
    current_memories: list[MemoryItem],
    current_facts: list[FactItem],
) -> dict[str, Any] | None:
    s = getattr(state, "settings", None)
    max_history_msgs = int(getattr(s, "memory_ctrl_planner_max_history_msgs", 12))
    max_msg_chars = int(getattr(s, "memory_ctrl_planner_max_msg_chars", 500))

    now_utc = datetime.now(UTC).replace(microsecond=0).isoformat()
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

        # Include role prefix to make evidence matching robust.
        history_chunks.append(f"{role}: {content}")

        display = content
        if len(display) > max_msg_chars:
            display = display[:max_msg_chars] + "…"

        lines.append(f"{i}. {role}: {display}")

    history_text = "\n".join(history_chunks)

    lines.extend(["", "Current structured facts (most important first):"])
    for f in current_facts or []:
        ts_str = _format_utc(float(getattr(f, "last_updated", 0.0)))
        key = str(getattr(f, "key", ""))
        val = getattr(f, "value", "")
        if isinstance(val, list):
            vstr = ", ".join(str(x) for x in val[:12])
            if len(val) > 12:
                vstr += ", …"
        else:
            vstr = str(val)
        if len(vstr) > max_msg_chars:
            vstr = vstr[:max_msg_chars] + "…"
        lines.append(
            f"- id={f.id}, subject_type={f.subject_type}, subject_id={f.subject_id}, "
            f"key={key}, confidence={f.confidence:.2f}, source={f.source}, "
            f"last_updated={ts_str}: {vstr}"
        )

    lines.extend(["", "Current memory items (most important first):"])
    for item in current_memories or []:
        tag_str = ",".join(item.tags)
        ts_str = _format_utc(item.last_updated)
        text = item.text
        if len(text) > max_msg_chars:
            text = text[:max_msg_chars] + "…"
        lines.append(
            f"- id={item.id}, subject_type={item.subject_type}, subject_id={item.subject_id}, "
            f"importance={item.importance:.2f}, last_updated={ts_str}, tags=[{tag_str}], "
            f"source={item.source}: {text}"
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

    raw = (raw or "").strip()
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

    ops = plan.get("ops", [])
    n_ops = len(ops) if isinstance(ops, list) else 0
    logger.debug("Memory controller plan ops=%s", n_ops)

    if isinstance(ops, list):
        with contextlib.suppress(Exception):
            logger.debug(
                "Memory controller ops preview=%s",
                json.dumps(ops, ensure_ascii=False)[:2000],
            )

    return plan


def apply_memory_plan(
    state: AppState,
    plan: dict[str, Any],
    *,
    default_user_id: str | None = None,
    default_room_id: str | None = None,
) -> None:
    ops = plan.get("ops")
    if not isinstance(ops, list):
        return

    history_text = plan.get("_history_text")
    if not isinstance(history_text, str):
        history_text = ""

    task_store = getattr(state, "task_store", None)
    settings = getattr(state, "settings", None)

    allowlist = set(getattr(settings, "memory_facts_allowlist", []) or [])

    def fact_allowed(key: str) -> bool:
        if not allowlist:
            return True
        k = str(key or "").strip().lower()
        return k in allowlist

    for op in ops:
        if not isinstance(op, dict):
            continue

        try:
            op_kind = _norm_op_kind(op.get("op"))

            # -------------------------
            # Facts (structured slots)
            # -------------------------
            if op_kind in {"fact_set", "fact_add_value", "fact_remove_value", "fact_delete"}:
                subject_type = _norm_subject_type(op.get("subject_type"))
                subject_id = op.get("subject_id")

                key = str(op.get("key") or "").strip().lower()
                if not key or not fact_allowed(key):
                    logger.debug("Skipping fact op: key not allowed key=%r", key)
                    continue

                # Resolve subject_id default
                if not subject_id:
                    if subject_type in ("user", "relationship"):
                        subject_id = default_user_id
                    elif subject_type == "room":
                        subject_id = default_room_id
                    else:
                        subject_id = state.memory.global_subject_id()

                if not isinstance(subject_id, str) or not subject_id.strip():
                    logger.debug(
                        "Skipping fact op: missing subject_id type=%s key=%s", subject_type, key
                    )
                    continue

                if op_kind != "fact_delete":
                    evidence = op.get("evidence")
                    if not isinstance(evidence, str) or not evidence.strip():
                        logger.debug(
                            "Skipping fact op: missing evidence op=%s key=%s", op_kind, key
                        )
                        continue
                    if history_text and not _evidence_matches_history(evidence, history_text):
                        logger.debug(
                            "Skipping fact op: evidence not in history key=%s ev=%r", key, evidence
                        )
                        continue

                src = str(op.get("source") or "auto").strip() or "auto"

                if op_kind == "fact_delete":
                    # evidence optional; still do best-effort
                    delete_fact(state, subject_type=subject_type, subject_id=subject_id, key=key)
                    logger.debug(
                        "Applied fact_delete type=%s subject=%s key=%s",
                        subject_type,
                        subject_id,
                        key,
                    )
                    continue

                # write ops need value
                value = op.get("value")
                if value is None:
                    continue

                if _looks_like_secret(str(value)):
                    logger.debug("Skipping fact op: looks like secret key=%s", key)
                    continue

                confidence = op.get("confidence")
                if not isinstance(confidence, (int, float)):
                    confidence = 0.85

                tags_raw = op.get("tags")
                tags = (
                    [str(t).strip() for t in tags_raw if str(t).strip()]
                    if isinstance(tags_raw, list)
                    else None
                )

                evidence = str(op.get("evidence") or "").strip()

                if op_kind == "fact_set":
                    set_fact(
                        state,
                        subject_type=subject_type,
                        subject_id=subject_id,
                        key=key,
                        value=value,
                        confidence=float(confidence),
                        tags=tags,
                        source=src,
                        evidence=evidence,
                        person_ref=f"user:{subject_id}"
                        if subject_type in ("user", "relationship")
                        else None,
                    )
                    logger.debug("Applied fact_set key=%s subject=%s", key, subject_id)

                elif op_kind == "fact_add_value":
                    add_fact_value(
                        state,
                        subject_type=subject_type,
                        subject_id=subject_id,
                        key=key,
                        value=str(value),
                        confidence=float(confidence),
                        tags=tags,
                        source=src,
                        evidence=evidence,
                        person_ref=f"user:{subject_id}"
                        if subject_type in ("user", "relationship")
                        else None,
                    )
                    logger.debug("Applied fact_add_value key=%s subject=%s", key, subject_id)

                elif op_kind == "fact_remove_value":
                    remove_fact_value(
                        state,
                        subject_type=subject_type,
                        subject_id=subject_id,
                        key=key,
                        value=str(value),
                        source=src,
                        evidence=evidence,
                    )
                    logger.debug("Applied fact_remove_value key=%s subject=%s", key, subject_id)

                continue

            # -------------------------
            # Unstructured memories
            # -------------------------
            if op_kind == "add":
                subject_type = _norm_subject_type(op.get("subject_type"))
                subject_id = op.get("subject_id")
                text = op.get("text")

                if not isinstance(text, str) or not text.strip():
                    continue

                evidence = op.get("evidence")
                if not isinstance(evidence, str) or not evidence.strip():
                    logger.debug("Skipping add: missing evidence")
                    continue
                if history_text and not _evidence_matches_history(evidence, history_text):
                    logger.debug(
                        "Skipping add: evidence not found in history evidence=%r", evidence
                    )
                    continue

                importance = op.get("importance")
                if not isinstance(importance, (int, float)):
                    importance = 0.7

                tags_raw = op.get("tags")
                tags_add = (
                    [str(t).strip() for t in tags_raw if str(t).strip()]
                    if isinstance(tags_raw, list)
                    else None
                )

                if not subject_id:
                    if subject_type in ("user", "relationship"):
                        subject_id = default_user_id
                    elif subject_type == "room":
                        subject_id = default_room_id
                    else:
                        subject_id = state.memory.global_subject_id()

                mem_id: int | None = None
                src = str(op.get("source") or "auto").strip() or "auto"

                if subject_type == "user" and isinstance(subject_id, str) and subject_id:
                    mem_id = remember_user_fact(
                        state,
                        subject_id,
                        text,
                        importance=float(importance),
                        tags=tags_add,
                        source=src,
                    )
                elif subject_type == "room" and isinstance(subject_id, str) and subject_id:
                    mem_id = remember_room_fact(
                        state,
                        subject_id,
                        text,
                        importance=float(importance),
                        tags=tags_add,
                        source=src,
                    )
                elif subject_type == "relationship" and isinstance(subject_id, str) and subject_id:
                    mem_id = remember_relationship_fact(
                        state,
                        subject_id,
                        default_room_id,
                        text,
                        importance=float(importance),
                        tags=tags_add,
                        source=src,
                    )
                elif subject_type == "global":
                    mem_id = remember_global_fact(
                        state, text, importance=float(importance), tags=tags_add, source=src
                    )

                if mem_id is not None:
                    logger.debug(
                        "Applied add(%s) mem_id=%s text=%r", subject_type, mem_id, text[:200]
                    )

            elif op_kind == "update":
                mem_id_raw = op.get("id")
                if mem_id_raw is None:
                    continue
                try:
                    mem_id = int(mem_id_raw)
                except Exception:
                    continue
                if mem_id <= 0:
                    continue

                new_text = op.get("text")
                if new_text is not None:
                    if not isinstance(new_text, str) or not new_text.strip():
                        continue
                    evidence = op.get("evidence")
                    if not isinstance(evidence, str) or not evidence.strip():
                        logger.debug("Skipping update: missing evidence (text change)")
                        continue
                    if history_text and not _evidence_matches_history(evidence, history_text):
                        logger.debug(
                            "Skipping update: evidence not found in history evidence=%r", evidence
                        )
                        continue

                tags_raw = op.get("tags")
                tags = (
                    [str(t).strip() for t in tags_raw if str(t).strip()]
                    if isinstance(tags_raw, list)
                    else None
                )

                importance = op.get("importance")
                if importance is not None and not isinstance(importance, (int, float)):
                    importance = None

                person_ref = op.get("person_ref", None) if "person_ref" in op else None
                kwargs: dict[str, Any] = {}
                if new_text is not None:
                    kwargs["text"] = new_text
                if tags is not None:
                    kwargs["tags"] = tags
                if importance is not None:
                    kwargs["importance"] = float(importance)
                if "person_ref" in op:
                    kwargs["person_ref"] = person_ref

                if kwargs:
                    state.memory.update_memory(mem_id, **kwargs)
                    logger.debug(
                        "Applied update mem_id=%s fields=%s", mem_id, sorted(kwargs.keys())
                    )

            elif op_kind == "delete":
                mem_id_raw = op.get("id")
                if mem_id_raw is None:
                    continue
                try:
                    mem_id = int(mem_id_raw)
                except Exception:
                    continue
                if mem_id <= 0:
                    continue
                state.memory.delete_memory(mem_id)
                logger.debug("Applied delete mem_id=%s", mem_id)

            elif op_kind == "task_add":
                if task_store is None:
                    continue
                _safe_add_task(task_store, op)

            else:
                logger.debug("Ignoring unknown op kind=%r op=%r", op_kind, op)

        except Exception:
            logger.exception("Failed to apply memory op=%r", op)
            continue
