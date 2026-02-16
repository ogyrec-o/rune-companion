# config.example.py

"""
Documentation-only module (safe to commit).

The real configuration is loaded from environment variables (optionally via a local .env file).
Do NOT commit real secrets. Use:
- .env (local, gitignored)
- config_local.py (local safe overrides, gitignored)

This file exists to make the repo self-documenting even without opening .env.example.
"""

ENV_VARS = {
    # App / logging
    "RUNE_APP_NAME": "App display name (default: rune).",
    "RUNE_LOG_LEVEL": "Logging level (default: INFO).",
    # Switches
    "RUNE_TTS_MODE": "Enable TTS mode (true/false).",
    "RUNE_SAVE_HISTORY": "Persist dialog history (true/false).",
    # Connectors
    "RUNE_CONSOLE_ENABLED": "Enable console connector (true/false).",
    "RUNE_MATRIX_ENABLED": "Enable Matrix connector (true/false).",
    # LLM / OpenRouter
    "RUNE_OPENROUTER_API_KEY": "OpenRouter API key (required only when using LLM).",
    "RUNE_OPENROUTER_BASE_URL": "OpenRouter base URL (default: https://openrouter.ai/api/v1).",
    "RUNE_LLM_MODELS": "Comma/space separated list of models to try in order.",
    "RUNE_HTTP_REFERER": "Optional OpenRouter metadata header.",
    "RUNE_APP_TITLE": "Optional OpenRouter metadata header title.",
    # Matrix
    "RUNE_MATRIX_HOMESERVER": "Matrix homeserver URL.",
    "RUNE_MATRIX_USER_ID": "Matrix user ID (bot).",
    "RUNE_MATRIX_PASSWORD": "Password for first login (session stored locally).",
    "RUNE_MATRIX_ROOMS": "Optional allowlist of room IDs (empty => all rooms).",
    # Paths (gitignored)
    "RUNE_DATA_DIR": "Local data directory (default: .local/rune).",
    "RUNE_MATRIX_STORE_PATH": "Matrix E2EE store path (default: <data_dir>/matrix_store).",
    "RUNE_MEMORY_DB_PATH": "MemoryStore SQLite path (default: <data_dir>/memory.sqlite3).",
    "RUNE_TASKS_DB_PATH": "TaskStore SQLite path (default: <data_dir>/tasks.sqlite3).",
    "RUNE_DIALOG_HISTORY_PATH": (
        "Dialog history JSON path (default: <data_dir>/dialog_histories.json)."
    ),
    # TTS
    "RUNE_SPEAKER_WAV": "Optional speaker WAV path (default: voice_sample.wav).",
    "RUNE_XTTS_SPEAKER_NAME": "XTTS built-in speaker name fallback (default: Ana Florence).",
    "RUNE_XTTS_LANGUAGE": "XTTS language (default: ru).",
    # Tuning
    "RUNE_MEMORY_MAX_USER": "Max user memories per user subject.",
    "RUNE_MEMORY_MAX_ROOM": "Max room memories per room subject.",
    "RUNE_MEMORY_MAX_REL": "Max relationship memories per user.",
    "RUNE_MEMORY_MAX_GLOBAL": "Max global memories.",
}
