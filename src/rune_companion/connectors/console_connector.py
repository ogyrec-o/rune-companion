# src/rune_companion/connectors/console_connector.py

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from typing import Optional

from ..cli.commands import registry as command_registry
from ..core.chat import stream_reply
from ..core.state import AppState
from ..llm.client import friendly_llm_error_message

logger = logging.getLogger(__name__)

SENTENCE_REGEX = re.compile(r"[\.!\?…]")


def _ts_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _rewrite_prev_line(line: str) -> None:
    """
    Replace the last terminal line with `line`.
    Best-effort: if not a TTY, just print a new line.
    """
    try:
        if sys.stdout.isatty():
            sys.stdout.write("\033[1A\033[2K\r")
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        else:
            print(line)
    except Exception:
        print(line)


def _print_ts(text: str) -> None:
    print(f"[{_ts_local()}] {text}")


def run_console_loop(state: AppState) -> None:
    logger.info("Console connector started (tts=%s).", state.tts_enabled)
    _print_ts("[CONSOLE] Type your messages. Use /help for commands. Use /exit to quit.\n")

    lock = getattr(state, "lock", None)
    app_name = str(getattr(getattr(state, "settings", None), "app_name", "rune"))

    def emit(text: str) -> None:
        # Immediate user-visible feedback for long operations (e.g., TTS init)
        print(f"[{_ts_local()}] {text}", flush=True)

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

        # Commands (/help, /tts, ...)
        try:
            if lock:
                with lock:
                    cmd_response = command_registry.handle(state, user_input, emit=emit)
            else:
                cmd_response = command_registry.handle(state, user_input, emit=emit)
        except Exception:
            logger.exception("Command handler crashed.")
            cmd_response = "Internal error while handling a command."

        if cmd_response is not None:
            print(f"[{_ts_local()}] {cmd_response}")
            continue

        # Normal chat: stream chunks, print them, optionally speak sentences.
        buffer = ""
        assistant_printed = False

        try:
            if lock:
                with lock:
                    pieces = stream_reply(state, user_input)
                    for piece in pieces:
                        if piece and not assistant_printed:
                            print(f"[{_ts_local()}] <<< {app_name}: ", end="", flush=True)
                            assistant_printed = True

                        if piece:
                            print(piece, end="", flush=True)
                            buffer += piece

                            if state.tts_enabled:
                                while True:
                                    m = SENTENCE_REGEX.search(buffer)
                                    if not m:
                                        break
                                    sentence = buffer[: m.end()].strip()
                                    buffer = buffer[m.end() :]
                                    if sentence:
                                        try:
                                            state.tts_engine.speak_sentence(sentence)
                                        except Exception:
                                            logger.debug("TTS speak_sentence failed.", exc_info=True)
            else:
                pieces = stream_reply(state, user_input)
                for piece in pieces:
                    if piece and not assistant_printed:
                        print(f"[{_ts_local()}] <<< {app_name}: ", end="", flush=True)
                        assistant_printed = True

                    if piece:
                        print(piece, end="", flush=True)
                        buffer += piece

                        if state.tts_enabled:
                            while True:
                                m = SENTENCE_REGEX.search(buffer)
                                if not m:
                                    break
                                sentence = buffer[: m.end()].strip()
                                buffer = buffer[m.end() :]
                                if sentence:
                                    try:
                                        state.tts_engine.speak_sentence(sentence)
                                    except Exception:
                                        logger.debug("TTS speak_sentence failed.", exc_info=True)

        except RuntimeError as e:
            msg = friendly_llm_error_message(e)
            logger.info("LLM runtime error: %s", msg)
            print(f"[{_ts_local()}] [LLM] {msg}")
            continue
        except Exception:
            logger.exception("Console chat handler crashed.")
            _print_ts("Internal error while generating a reply.")
            continue

        if not assistant_printed:
            print(f"[{_ts_local()}] [LLM] No output (model produced no content).")
            continue

        # Finish TTS tail + wait.
        if state.tts_enabled:
            tail = buffer.strip()
            if tail:
                try:
                    state.tts_engine.speak_sentence(tail)
                except Exception:
                    logger.debug("TTS speak_sentence (tail) failed.", exc_info=True)

            try:
                state.tts_engine.wait_all()
            except Exception:
                logger.debug("TTS wait_all failed.", exc_info=True)

            print(f"\n[{_ts_local()}] [LLM] Audio done.\n")
        else:
            print("\n")

    logger.info("Console connector finished.")
