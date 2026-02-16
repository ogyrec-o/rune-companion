# src/rune_companion/cli/commands.py

from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Callable
from datetime import datetime
from typing import cast

from ..core.state import AppState
from ..memory.api import get_top_room_memories, get_top_user_memories
from ..tts.engine import TTSEngine  # concrete engine (not the Protocol)

CommandEmitter = Callable[[str], None]
CommandHandler4 = Callable[[AppState, list[str], str | None, str | None], str]
CommandHandler5 = Callable[
    [AppState, list[str], str | None, str | None, CommandEmitter | None], str
]
CommandHandler = CommandHandler4 | CommandHandler5

logger = logging.getLogger(__name__)


class CommandRegistry:
    """Simple slash-command registry used by connectors (/help, /tts, ...)."""

    def __init__(self) -> None:
        self._handlers: dict[str, CommandHandler] = {}
        self._help: dict[str, str] = {}

    def register(
        self,
        name: str,
        handler: CommandHandler,
        help_text: str,
        aliases: list[str] | None = None,
    ) -> None:
        aliases = aliases or []
        key = name.lower()
        self._handlers[key] = handler
        self._help[key] = help_text
        for alias in aliases:
            self._handlers[alias.lower()] = handler

    def handle(
        self,
        state: AppState,
        line: str,
        user_id: str | None = None,
        room_id: str | None = None,
        emit: CommandEmitter | None = None,
    ) -> str | None:
        """
        Handle a string like "/command args".
        Returns a reply string or None if not a command.
        """
        if not line.startswith("/"):
            return None

        parts = line[1:].split()
        if not parts:
            return "Empty command. Use /help to list available commands."

        name = parts[0].lower()
        args = parts[1:]

        handler = self._handlers.get(name)
        if not handler:
            return f"Unknown command: /{name}. Use /help to list available commands."

        try:
            nparams = len(inspect.signature(handler).parameters)
        except Exception:
            nparams = 5

        if nparams >= 5:
            h5 = cast(CommandHandler5, handler)
            return h5(state, args, user_id, room_id, emit)

        h4 = cast(CommandHandler4, handler)
        return h4(state, args, user_id, room_id)

    def build_help(self) -> str:
        lines = ["Available commands:"]
        for name, help_text in self._help.items():
            lines.append(f"  /{name} - {help_text}")
        return "\n".join(lines)


registry = CommandRegistry()


def _ts_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def cmd_help(
    state: AppState,
    args: list[str],
    user_id: str | None,
    room_id: str | None,
    emit: CommandEmitter | None = None,
) -> str:
    return registry.build_help()


def cmd_status(
    state: AppState,
    args: list[str],
    user_id: str | None,
    room_id: str | None,
    emit: CommandEmitter | None = None,
) -> str:
    mode = "VOICE (TTS)" if state.tts_enabled else "TEXT ONLY"
    hist = "ON" if state.save_history else "OFF"
    models = ", ".join(list(getattr(state.settings, "llm_models", []) or []))
    return (
        "Status:\n"
        f"  Mode: {mode}\n"
        f"  Dialog history: {hist}\n"
        f"  Models (priority -> fallback): {models}"
    )


def cmd_tts(
    state: AppState,
    args: list[str],
    user_id: str | None,
    room_id: str | None,
    emit: CommandEmitter | None = None,
) -> str:
    """
    /tts          -> show status
    /tts on       -> enable TTS
    /tts off      -> disable TTS
    """
    if not args:
        return f"TTS is currently {'ON' if state.tts_enabled else 'OFF'}. Use /tts on or /tts off."

    arg = args[0].lower()

    if arg in ("on", "1", "true", "yes"):
        if state.tts_enabled:
            return "TTS is already ON."

        if emit:
            with contextlib.suppress(Exception):
                emit("[TTS] Enabling... importing deps and loading model (may take a while).")

        # Do NOT duplicate the user-facing message in INFO logs (it prints into console).
        logger.debug("TTS enable requested (user_id=%s room_id=%s)", user_id, room_id)

        state.tts_engine = TTSEngine(enabled=True, settings=state.settings)
        state.tts_enabled = True
        state.save_history = False
        return "TTS enabled. Replies will be spoken."

    if arg in ("off", "0", "false", "no"):
        if not state.tts_enabled:
            return "TTS is already OFF."

        if emit:
            with contextlib.suppress(Exception):
                emit("[TTS] Disabling...")

        logger.debug("TTS disable requested (user_id=%s room_id=%s)", user_id, room_id)

        with contextlib.suppress(Exception):
            state.tts_engine.shutdown()

        state.tts_engine = TTSEngine(enabled=False)
        state.tts_enabled = False
        state.save_history = True
        return "TTS disabled. Replies will be text-only."

    return "Usage: /tts on or /tts off."


def cmd_mem(
    state: AppState,
    args: list[str],
    user_id: str | None,
    room_id: str | None,
    emit: CommandEmitter | None = None,
) -> str:
    """
    /mem user  -> show memory about current user
    /mem room  -> show memory about current room
    /mem stat  -> show totals
    """
    if not args:
        return (
            "Memory diagnostics:\n"
            "  /mem user  - memory about current user\n"
            "  /mem room  - memory about this room\n"
            "  /mem stat  - total memory count\n"
        )

    sub = args[0].lower()

    if sub in ("user", "me"):
        if not user_id:
            return "No user_id in this context."
        mems = get_top_user_memories(state, user_id, limit=20)
        if not mems:
            return f"No memories stored for user_id={user_id}."
        lines = [f"Memories about user {user_id}:"]
        for i, m in enumerate(mems, start=1):
            tags = getattr(m, "tags", None) or []
            tag_str = f" [tags: {', '.join(tags)}]" if tags else ""
            lines.append(f"{i}. ({m.importance:.2f}) {m.text}{tag_str}")
        return "\n".join(lines)

    if sub == "room":
        if not room_id:
            return "This command is only available in a room/chat context (missing room_id)."
        mems = get_top_room_memories(state, room_id, limit=20)
        if not mems:
            return f"No memories stored for room_id={room_id}."
        lines = [f"Memories about room {room_id}:"]
        for i, m in enumerate(mems, start=1):
            tags = getattr(m, "tags", None) or []
            tag_str = f" [tags: {', '.join(tags)}]" if tags else ""
            lines.append(f"{i}. ({m.importance:.2f}) {m.text}{tag_str}")
        return "\n".join(lines)

    if sub == "stat":
        total = state.memory.count_memories()
        return f"Total memory items: {total}"

    return (
        "Unknown /mem subcommand.\n"
        "Usage:\n"
        "  /mem user  - memory about current user\n"
        "  /mem room  - memory about this room\n"
        "  /mem stat  - total memory count\n"
    )


registry.register("help", cmd_help, help_text="Show available commands.", aliases=["h", "?"])
registry.register("status", cmd_status, help_text="Show current settings (mode/history/models).")
registry.register("tts", cmd_tts, help_text="Enable/disable TTS: /tts on | /tts off.")
registry.register(
    "mem", cmd_mem, help_text="Memory diagnostics: /mem user | /mem room | /mem stat."
)
