# Architecture overview

Rune Companion is built around a small core with injected "ports" and thin connectors.

## Why ports?
Connectors (console/Matrix/etc.) should not know:
- how prompts are built,
- how memory/tasks are stored,
- how the LLM provider is called.

Instead, they depend on a stable interface. This makes it easy to:
- add another connector later,
- swap storage or LLM client implementations,
- test the core in isolation.

The contracts live in `src/rune_companion/core/ports.py`.

## Data flow

Connector → Core → LLM → Connector

1) A connector receives a message.
2) It calls `core.chat.stream_reply(state, user_text, user_id=..., room_id=...)`.
3) Core builds the prompt:
    - optional per-dialog history,
    - memory injection (<MEMORY> block),
    - open tasks injection,
    - system persona prompt.
4) LLM client streams text chunks back.
5) Connector prints chunks (and may feed them into TTS sentence-by-sentence).

## State ownership
`AppState` is a composition root container:
- `settings` (loaded once)
- `llm`, `memory`, `task_store`, `tts_engine` (injected implementations)
- dialog histories and per-dialog counters

Bootstrap code lives in `cli/bootstrap.py` to keep `core/state.py` focused and import-safe.

## Storage
- Memory: SQLite (fact-like items with tags, importance, timestamps)
- Tasks: SQLite (reminders and ask/reply workflows)
- Dialog histories: optional JSON file (when `RUNE_SAVE_HISTORY=true`)
- Matrix session store: under data dir (gitignored)
