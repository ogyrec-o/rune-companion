# src/rune_companion/config.py

"""Centralized settings loaded from environment variables (+ optional .env).

Design goals:
- One Settings object for the whole app (normal "settings layer").
- No secrets required at import time.
- Backward compatible: legacy module-level constants are exported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ENV_PREFIX = "RUNE"


def _k(suffix: str) -> str:
    """Build env var name with the project prefix."""
    return f"{ENV_PREFIX}_{suffix}"


def _load_dotenv_if_available() -> None:
    """Load .env locally if python-dotenv is installed. Safe no-op otherwise."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(override=False)


_load_dotenv_if_available()


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v


def _first_env(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v is not None and v.strip() != "":
            return v
    return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    return parts


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return Path(raw).expanduser()


@dataclass(frozen=True, slots=True)
class Settings:
    # ---- App / logging ----
    app_name: str
    log_level: str

    # ---- Global switches ----
    tts_mode: bool
    save_history: bool

    # ---- Connector flags ----
    console_enabled: bool
    matrix_enabled: bool

    # ---- LLM / OpenRouter ----
    openrouter_api_key: Optional[str]
    openrouter_base_url: str
    llm_models: List[str]
    extra_headers: Dict[str, str]

    # ---- TTS ----
    speaker_wav: str
    xtts_speaker_name: str
    xtts_language: str

    # ---- Matrix ----
    matrix_homeserver: str
    matrix_user_id: str
    matrix_password: str
    matrix_rooms: List[str]

    # ---- Local data paths (ignored by git) ----
    data_dir: Path
    matrix_store_path: Path
    memory_db_path: Path
    tasks_db_path: Path
    dialog_history_path: Path

    # ---- Memory tuning ----
    memory_max_user: int
    memory_max_room: int
    memory_max_rel: int
    memory_max_global: int

    memory_prompt_limit_user: int
    memory_prompt_limit_rel: int
    memory_prompt_limit_room: int
    memory_prompt_limit_global: int
    memory_prompt_limit_global_userstories: int

    memory_ctrl_every_n_messages: int
    memory_ctrl_last_messages: int

    memory_episode_threshold_messages: int
    memory_episode_chunk_messages: int

    memory_max_dialog_messages: int

    @staticmethod
    def from_env() -> "Settings":
        # App name: accept both RUNE_APP_NAME and RUNE_APP_TITLE for convenience.
        app_name = _first_env(_k("APP_NAME"), default=_env(_k("APP_TITLE"), "rune")) or "rune"
        log_level = _env(_k("LOG_LEVEL"), "INFO")

        tts_mode = _env_bool(_k("TTS_MODE"), False)
        save_history = _env_bool(_k("SAVE_HISTORY"), not tts_mode)

        console_enabled = _env_bool(_k("CONSOLE_ENABLED"), _env_bool("CONSOLE_ENABLED", True))
        matrix_enabled = _env_bool(_k("MATRIX_ENABLED"), _env_bool("MATRIX_ENABLED", False))

        openrouter_api_key = _first_env(_k("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY", default=None)
        openrouter_base_url = _env(_k("OPENROUTER_BASE_URL"), "https://openrouter.ai/api/v1")

        http_referer = _env(_k("HTTP_REFERER"), "https://example.com")
        # Use explicit title header if provided; else fall back to app_name
        title = _env(_k("APP_TITLE"), app_name)

        extra_headers = {
            "HTTP-Referer": http_referer,
            "X-Title": title,
        }

        llm_models = _env_list(
            _k("LLM_MODELS"),
            [
                "x-ai/grok-4.1-fast:free",
                "qwen/qwen-2.5-72b-instruct:free",
                "deepseek/deepseek-chat-v3-0324:free",
                "deepseek/deepseek-r1-0528:free",
            ],
        )

        speaker_wav = _env(_k("SPEAKER_WAV"), "voice_sample.wav")
        xtts_speaker_name = _env(_k("XTTS_SPEAKER_NAME"), "Ana Florence")
        xtts_language = _env(_k("XTTS_LANGUAGE"), "ru")

        matrix_homeserver = (_first_env(_k("MATRIX_HOMESERVER"), "MATRIX_HOMESERVER", default="") or "").strip()
        matrix_user_id = (_first_env(_k("MATRIX_USER_ID"), "MATRIX_USER_ID", default="") or "").strip()
        matrix_password = (_first_env(_k("MATRIX_PASSWORD"), "MATRIX_PASSWORD", default="") or "").strip()
        matrix_rooms = _env_list(_k("MATRIX_ROOMS"), _env_list("MATRIX_ROOMS", []))

        data_dir = _env_path(_k("DATA_DIR"), Path(".local/rune"))
        matrix_store_path = _env_path(_k("MATRIX_STORE_PATH"), data_dir / "matrix_store")
        memory_db_path = _env_path(_k("MEMORY_DB_PATH"), data_dir / "memory.sqlite3")
        tasks_db_path = _env_path(_k("TASKS_DB_PATH"), data_dir / "tasks.sqlite3")
        dialog_history_path = _env_path(_k("DIALOG_HISTORY_PATH"), data_dir / "dialog_histories.json")

        memory_max_user = _env_int(_k("MEMORY_MAX_USER"), _env_int("MEMORY_MAX_USER", 800))
        memory_max_room = _env_int(_k("MEMORY_MAX_ROOM"), _env_int("MEMORY_MAX_ROOM", 400))
        memory_max_rel = _env_int(_k("MEMORY_MAX_REL"), _env_int("MEMORY_MAX_REL", 600))
        memory_max_global = _env_int(_k("MEMORY_MAX_GLOBAL"), _env_int("MEMORY_MAX_GLOBAL", 800))

        memory_prompt_limit_user = _env_int(_k("MEMORY_PROMPT_LIMIT_USER"), _env_int("MEMORY_PROMPT_LIMIT_USER", 12))
        memory_prompt_limit_rel = _env_int(_k("MEMORY_PROMPT_LIMIT_REL"), _env_int("MEMORY_PROMPT_LIMIT_REL", 12))
        memory_prompt_limit_room = _env_int(_k("MEMORY_PROMPT_LIMIT_ROOM"), _env_int("MEMORY_PROMPT_LIMIT_ROOM", 8))
        memory_prompt_limit_global = _env_int(_k("MEMORY_PROMPT_LIMIT_GLOBAL"), _env_int("MEMORY_PROMPT_LIMIT_GLOBAL", 8))
        memory_prompt_limit_global_userstories = _env_int(
            _k("MEMORY_PROMPT_LIMIT_GLOBAL_USERSTORIES"),
            _env_int("MEMORY_PROMPT_LIMIT_GLOBAL_USERSTORIES", 8),
        )

        memory_ctrl_every_n_messages = _env_int(
            _k("MEMORY_CTRL_EVERY_N_MESSAGES"),
            _env_int("MEMORY_CTRL_EVERY_N_MESSAGES", 1),
        )
        memory_ctrl_last_messages = _env_int(
            _k("MEMORY_CTRL_LAST_MESSAGES"),
            _env_int("MEMORY_CTRL_LAST_MESSAGES", 8),
        )

        memory_episode_threshold_messages = _env_int(
            _k("MEMORY_EPISODE_THRESHOLD_MESSAGES"),
            _env_int("MEMORY_EPISODE_THRESHOLD_MESSAGES", 20),
        )
        memory_episode_chunk_messages = _env_int(
            _k("MEMORY_EPISODE_CHUNK_MESSAGES"),
            _env_int("MEMORY_EPISODE_CHUNK_MESSAGES", 24),
        )

        memory_max_dialog_messages = _env_int(
            _k("MEMORY_MAX_DIALOG_MESSAGES"),
            _env_int("MEMORY_MAX_DIALOG_MESSAGES", 80),
        )

        return Settings(
            app_name=app_name,
            log_level=log_level,
            tts_mode=tts_mode,
            save_history=save_history,
            console_enabled=console_enabled,
            matrix_enabled=matrix_enabled,
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=openrouter_base_url,
            llm_models=llm_models,
            extra_headers=extra_headers,
            speaker_wav=speaker_wav,
            xtts_speaker_name=xtts_speaker_name,
            xtts_language=xtts_language,
            matrix_homeserver=matrix_homeserver,
            matrix_user_id=matrix_user_id,
            matrix_password=matrix_password,
            matrix_rooms=matrix_rooms,
            data_dir=data_dir,
            matrix_store_path=matrix_store_path,
            memory_db_path=memory_db_path,
            tasks_db_path=tasks_db_path,
            dialog_history_path=dialog_history_path,
            memory_max_user=memory_max_user,
            memory_max_room=memory_max_room,
            memory_max_rel=memory_max_rel,
            memory_max_global=memory_max_global,
            memory_prompt_limit_user=memory_prompt_limit_user,
            memory_prompt_limit_rel=memory_prompt_limit_rel,
            memory_prompt_limit_room=memory_prompt_limit_room,
            memory_prompt_limit_global=memory_prompt_limit_global,
            memory_prompt_limit_global_userstories=memory_prompt_limit_global_userstories,
            memory_ctrl_every_n_messages=memory_ctrl_every_n_messages,
            memory_ctrl_last_messages=memory_ctrl_last_messages,
            memory_episode_threshold_messages=memory_episode_threshold_messages,
            memory_episode_chunk_messages=memory_episode_chunk_messages,
            memory_max_dialog_messages=memory_max_dialog_messages,
        )


SETTINGS = Settings.from_env()

# ---- Optional local overrides (never committed) ----
# Prefer .env for secrets; use config_local.py only for safe overrides.
try:
    import config_local as _config_local  # type: ignore

    # Simple overrides for selected legacy names. Keep it explicit.
    if hasattr(_config_local, "CONSOLE_ENABLED"):
        object.__setattr__(SETTINGS, "console_enabled", bool(_config_local.CONSOLE_ENABLED))  # type: ignore[misc]
    if hasattr(_config_local, "MATRIX_ENABLED"):
        object.__setattr__(SETTINGS, "matrix_enabled", bool(_config_local.MATRIX_ENABLED))  # type: ignore[misc]
except Exception:
    pass


def get_settings() -> Settings:
    return SETTINGS


# --------------------------------------------------------------------------------------
# Backward-compatible exports (legacy module-level constants).
# --------------------------------------------------------------------------------------

# App / logging
APP_NAME = SETTINGS.app_name
LOG_LEVEL = SETTINGS.log_level

# switches
TTS_MODE = SETTINGS.tts_mode
SAVE_HISTORY = SETTINGS.save_history

# connectors
CONSOLE_ENABLED = SETTINGS.console_enabled
MATRIX_ENABLED = SETTINGS.matrix_enabled

# LLM / OpenRouter
OPENROUTER_API_KEY = SETTINGS.openrouter_api_key
OPENROUTER_BASE_URL = SETTINGS.openrouter_base_url
LLM_MODELS = SETTINGS.llm_models
EXTRA_HEADERS = SETTINGS.extra_headers

# TTS
SPEAKER_WAV = SETTINGS.speaker_wav
XTTS_SPEAKER_NAME = SETTINGS.xtts_speaker_name
XTTS_LANGUAGE = SETTINGS.xtts_language

# Matrix
MATRIX_HOMESERVER = SETTINGS.matrix_homeserver
MATRIX_USER_ID = SETTINGS.matrix_user_id
MATRIX_PASSWORD = SETTINGS.matrix_password
MATRIX_ROOMS = SETTINGS.matrix_rooms

# Paths
DATA_DIR = SETTINGS.data_dir
MATRIX_STORE_PATH = SETTINGS.matrix_store_path
MEMORY_DB_PATH = SETTINGS.memory_db_path
TASKS_DB_PATH = SETTINGS.tasks_db_path
DIALOG_HISTORY_PATH = SETTINGS.dialog_history_path

# Memory tuning
MEMORY_MAX_USER = SETTINGS.memory_max_user
MEMORY_MAX_ROOM = SETTINGS.memory_max_room
MEMORY_MAX_REL = SETTINGS.memory_max_rel
MEMORY_MAX_GLOBAL = SETTINGS.memory_max_global

MEMORY_PROMPT_LIMIT_USER = SETTINGS.memory_prompt_limit_user
MEMORY_PROMPT_LIMIT_REL = SETTINGS.memory_prompt_limit_rel
MEMORY_PROMPT_LIMIT_ROOM = SETTINGS.memory_prompt_limit_room
MEMORY_PROMPT_LIMIT_GLOBAL = SETTINGS.memory_prompt_limit_global
MEMORY_PROMPT_LIMIT_GLOBAL_USERSTORIES = SETTINGS.memory_prompt_limit_global_userstories

MEMORY_CTRL_EVERY_N_MESSAGES = SETTINGS.memory_ctrl_every_n_messages
MEMORY_CTRL_LAST_MESSAGES = SETTINGS.memory_ctrl_last_messages

MEMORY_EPISODE_THRESHOLD_MESSAGES = SETTINGS.memory_episode_threshold_messages
MEMORY_EPISODE_CHUNK_MESSAGES = SETTINGS.memory_episode_chunk_messages

MEMORY_MAX_DIALOG_MESSAGES = SETTINGS.memory_max_dialog_messages
