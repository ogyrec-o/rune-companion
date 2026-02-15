# connectors/console_connector.py

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Optional

from commands import registry as command_registry
from core_chat import chat_with_streaming_console
from state import AppState

logger = logging.getLogger(__name__)


def _ts_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _rewrite_prev_line(line: str) -> None:
    """
    Replace the last terminal line with `line`.
    Best-effort: if not a TTY, just print a new line.
    """
    try:
        if sys.stdout.isatty():
            # up one line, clear it, print replacement
            sys.stdout.write("\033[1A\033[2K\r")
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        else:
            print(line)
    except Exception:
        print(line)


def _print_ts(text: str) -> None:
    print(f"[{_ts_local()}] {text}")


def _print_ts_block(text: str) -> None:
    ts = _ts_local()
    lines = text.splitlines() or [""]
    for i, line in enumerate(lines):
        # keep nice alignment for multi-line command output
        prefix = f"[{ts}] " if i == 0 else " " * (len(ts) + 3)
        print(prefix + line)


def run_console_loop(state: AppState) -> None:
    logger.info("Console connector started (tts=%s).", state.tts_enabled)
    _print_ts("[CONSOLE] Type your messages. Use /help for commands. Use /exit to quit.\n")

    lock = getattr(state, "lock", None)

    while True:
        try:
            user_input = input(">>> You: ").strip()
            sent_ts = _ts_local()
            _rewrite_prev_line(f"[{sent_ts}] >>> You: {user_input}")
        except EOFError:
            logger.info("Console EOF received, exiting.")
            break
        except KeyboardInterrupt:
            logger.info("Console KeyboardInterrupt, exiting.")
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit"):
            logger.info("Console exit command received.")
            break

        try:
            if lock:
                with lock:
                    cmd_response: Optional[str] = command_registry.handle(state, user_input)
            else:
                cmd_response = command_registry.handle(state, user_input)
        except Exception:
            logger.exception("Command handler crashed.")
            cmd_response = "Internal error while handling a command."

        if cmd_response is not None:
            print(f"[{_ts_local()}] {cmd_response}")
            continue

        try:
            if lock:
                with lock:
                    chat_with_streaming_console(state, user_input)
            else:
                chat_with_streaming_console(state, user_input)
        except Exception:
            logger.exception("Console chat handler crashed.")
            _print_ts("Internal error while generating a reply.")

    logger.info("Console connector finished.")
