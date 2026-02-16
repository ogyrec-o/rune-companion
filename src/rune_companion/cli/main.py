# src/rune_companion/cli/main.py

"""
CLI entrypoint.

Initializes logging, builds AppState, then starts connectors:
- console REPL in the main thread (optional),
- Matrix connector in a background thread (optional).
"""

from __future__ import annotations

import logging
import signal
import threading
from typing import TYPE_CHECKING

from ..cli.bootstrap import create_initial_state, load_dialog_histories, save_dialog_histories
from ..config import get_settings
from ..connectors.console_connector import run_console_loop
from ..logging_setup import setup_logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..connectors.matrix_connector import MatrixBackgroundRunner


def _shutdown(state) -> None:
    """Best-effort shutdown (no exceptions should escape)."""
    try:
        save_dialog_histories(state)
    except Exception:
        logger.exception("Failed to save dialog histories.")

    # TaskStore uses short-lived sqlite connections per call; no explicit close required.
    try:
        mem = getattr(state, "memory", None)
        if mem is not None and hasattr(mem, "close"):
            mem.close()
    except Exception:
        logger.debug("Memory close failed.", exc_info=True)

    try:
        tts = getattr(state, "tts_engine", None)
        if tts is not None and hasattr(tts, "shutdown"):
            tts.shutdown()
    except Exception:
        logger.debug("TTS shutdown failed.", exc_info=True)


def main() -> None:
    settings = get_settings()

    # choose console log level from settings.log_level
    level_name = str(getattr(settings, "log_level", "INFO")).upper()
    console_level = getattr(logging, level_name, logging.INFO)

    # choose log dir (prefer settings.data_dir if it exists)
    log_dir = getattr(settings, "data_dir", ".local/rune")
    setup_logging(log_dir=log_dir, console_level=console_level)

    # keep noisy libs readable (optional but recommended)
    logging.getLogger("nio").setLevel(max(console_level, logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logger.info("Starting %s...", getattr(settings, "app_name", "rune"))

    # IMPORTANT: reuse same settings object
    state = create_initial_state(settings=settings)

    if getattr(state, "save_history", False):
        try:
            state.dialog_histories = load_dialog_histories(state)
        except Exception:
            logger.exception("Failed to load dialog histories.")

    matrix_runner: MatrixBackgroundRunner | None = None
    if settings.matrix_enabled:
        from ..connectors.matrix_connector import start_matrix_in_background

        matrix_runner = start_matrix_in_background(state)

    # Use an Event so main can wait without a busy while-loop.
    stop_main = threading.Event()

    def _handle_signal(signum, _frame) -> None:
        logger.info("Signal %s received, shutting down...", signum)
        stop_main.set()

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        # Some platforms may not support SIGTERM, etc.
        pass

    try:
        if settings.console_enabled:
            run_console_loop(state)
            stop_main.set()
        else:
            logger.info(
                "Console disabled. Running background connectors only. Press Ctrl+C to stop."
            )
            stop_main.wait()

    finally:
        if matrix_runner is not None:
            matrix_runner.stop()
            matrix_runner.join(timeout=10.0)

        _shutdown(state)
        logger.info("Bye.")


if __name__ == "__main__":
    main()
