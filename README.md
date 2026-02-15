# README.md

# Rune Companion

A small, publishable chat companion framework with:
- streaming LLM replies (OpenRouter-compatible),
- per-dialog history,
- long-term memory (SQLite),
- a task system (SQLite) with reminders / ask-and-reply workflows,
- connectors: console REPL + Matrix (optional, E2EE supported when available).

This repo is designed to be safe to publish: no secrets and no user data are committed.

## Quickstart

### 1) Create a venv and install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
