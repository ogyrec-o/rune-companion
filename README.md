# Rune Companion

![CI](https://github.com/ogyrec-o/rune-companion/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)

Rune Companion is a small, publishable chat companion framework with:

- streaming LLM replies (OpenAI/OpenRouter-compatible),
- per-dialog history (optional JSON persistence),
- long-term memory (SQLite),
- a task system (SQLite) for reminders / ask-and-reply workflows,
- connectors: console REPL + Matrix (optional; E2EE supported when available).

The project is intentionally structured as a tiny "core + ports + connectors" app:
connectors handle I/O, the core builds prompts and streams replies, and storage/LLM/TTS are injected.

## Architecture overview (short)

High-level flow:

1) A connector receives a message (console or Matrix).
2) It calls `core.chat.stream_reply(...)`.
3) `stream_reply`:
    - optionally updates per-dialog history,
    - optionally runs summarizer + memory controller,
    - injects relevant memories + open tasks into the system prompt,
    - streams tokens from the LLM client back to the connector.
4) The connector prints streamed chunks (and can speak sentences when TTS is enabled).

Contracts live in `src/rune_companion/core/ports.py`:
LLM client, memory repo, task repo, outbound messenger, and TTS engine are all "ports".

More details:
- `docs/architecture.md`
- `docs/connectors.md`
- `docs/memory.md`
- `docs/running-locally.md`

## Quickstart

### 1) Create a venv and install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install the package in editable mode
pip install -e .

# Dev tools (ruff/mypy/pytest)
pip install -e ".[dev]"

# Matrix connector is optional
pip install -e ".[matrix]"
```

### 2) Configure environment
Configuration is loaded from environment variables (and optionally from a local `.env` file).
See `config.example.py` for the full list and descriptions.

Minimum for LLM usage (example `.env`):
```text
RUNE_OPENROUTER_API_KEY=...
RUNE_LLM_MODELS=x-ai/grok-4.1-fast
```
Useful defaults:
```text
RUNE_DATA_DIR=.local/rune
RUNE_LOG_LEVEL=INFO
RUNE_CONSOLE_ENABLED=true
RUNE_MATRIX_ENABLED=false
RUNE_SAVE_HISTORY=true
RUNE_TTS_MODE=false
```

### Offline demo (no external services)
If you don't set `RUNE_OPENROUTER_API_KEY` (or disable LLM explicitly), the app runs in a deterministic
offline demo mode. This keeps the console connector usable without external services.

Enable offline mode explicitly:
```text
RUNE_LLM_ENABLED=false
```

### 3) Run
The package installs a CLI entrypoint:
```bash
rune-companion
```
You can also run it as a module:
```bash
python -m rune_companion
```

## Connectors

#### Console
- Interactive REPL in your terminal.
- Streams assistant output as it arrives.
- If TTS is enabled, the connector speaks sentence-by-sentence.

#### Matrix (optional)
- Runs in a background thread with its own async loop.
- Can be restricted to an allowlist of rooms.
- Stores Matrix session/token data under the local data directory.

See `docs/connectors.md` for details.

## Memory subsystem (short)
- Memories are stored in SQLite as small "facts" with tags and importance.
- The memory controller periodically decides what to add/update/delete.
- Episodic summaries can store "what happened recently" as compact memory.
- Open tasks can be injected into the prompt as internal context.

See `docs/memory.md`.

### Local data
By default everything goes under `RUNE_DATA_DIR` (default: `.local/rune`), for example:
- `memory.sqlite3`
- `tasks.sqlite3`
- `dialog_histories.json`
- `matrix_store/`
- `rune.log`

These files are expected to be gitignored.

### Development notes
- Python: 3.12+
- Formatting/linting: ruff (see pyproject.toml)

## Quality checks (same as CI)

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src/rune_companion
python -m pytest
```

License: MIT.
