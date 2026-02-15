# How to run locally

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Matrix connector (optional):
```bash
pip install -e ".[matrix]"
```

## Configure
Settings are read from environment variables and optionally from a local .env file.
- For a full list of variables, see `config.example.py`.
- `config_local.py` is an optional local-only override file (should remain gitignored). 
See `config_local.example.py`.

Minimal `.env` for LLM:
```text
RUNE_OPENROUTER_API_KEY=...
RUNE_LLM_MODELS=x-ai/grok-4.1-fast
```
Recommended local defaults:
```text
RUNE_DATA_DIR=.local/rune
RUNE_LOG_LEVEL=INFO
RUNE_CONSOLE_ENABLED=true
RUNE_MATRIX_ENABLED=false
RUNE_SAVE_HISTORY=true
RUNE_TTS_MODE=false
```

## Run
```bash
rune-companion
```
Or:
```bash
python -m rune_companion
```

## Logs and data
All local data goes under `RUNE_DATA_DIR` (default `.local/rune`), including:
- `rune.log`
- `memory.sqlite3`
- `tasks.sqlite3`
- `dialog_histories.json`
- Matrix store directory

These should be gitignored.
