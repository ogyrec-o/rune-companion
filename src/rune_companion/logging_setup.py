# src/rune_companion/logging_setup.py

from __future__ import annotations

import logging
import sys
from pathlib import Path


class _ConsoleNoiseFilter(logging.Filter):
    """
    Make interactive console usable:
    - allow most rune_companion logs
    - but suppress chatty background components (matrix) unless WARNING+
    - suppress third-party noise unless ERROR+
    - suppress Python warnings (captured as 'py.warnings') unless ERROR+
    - suppress nio crypto spam unless ERROR+
    """

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name

        # Our app logs: keep, but make Matrix connector quiet (it runs in background thread).
        if name.startswith("rune_companion."):
            if name.startswith("rune_companion.connectors.matrix_"):
                return record.levelno >= logging.WARNING
            return True

        # Python warnings captured into logging.
        if name == "py.warnings":
            return record.levelno >= logging.ERROR

        # NIO crypto warnings are extremely spammy on first sync in encrypted rooms.
        if name.startswith("nio.crypto"):
            return record.levelno >= logging.ERROR

        # Any other 3rd party: only errors to console.
        return record.levelno >= logging.ERROR


def setup_logging(
    *,
    log_dir: str | Path = ".local/rune",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Configure logging with:
    - Console handler: readable + filtered for interactive use
    - File handler: full logs for debugging

    Call this ONCE, very early (before first logger.info).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "rune.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any pre-existing handlers to avoid duplicates.
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console (interactive)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    ch.addFilter(_ConsoleNoiseFilter())
    root.addHandler(ch)

    # File (everything)
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Route warnings.warn(...) into logging as 'py.warnings'
    logging.captureWarnings(True)

    # Optional: keep some libraries from being too verbose in the file as well (tweak if needed)
    # logging.getLogger("nio").setLevel(logging.INFO)
    # logging.getLogger("TTS").setLevel(logging.INFO)
