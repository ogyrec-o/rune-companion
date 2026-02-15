# Connectors

Connectors are thin adapters that:
- receive inbound text (from a specific transport),
- call the core (`stream_reply` or `generate_reply_text`),
- send output back (print to console / send Matrix message),
- optionally pass context identifiers: `user_id` and `room_id`.

## Console connector
File: `src/rune_companion/connectors/console_connector.py`

- interactive REPL loop
- supports slash commands (`/help`, `/tts`, `/mem`, ...)
- streams assistant output to stdout
- if TTS is enabled, speaks sentence-by-sentence while streaming

## Matrix connector
Files:
- `src/rune_companion/connectors/matrix_connector.py`
- `src/rune_companion/connectors/matrix_client.py`
- `src/rune_companion/connectors/matrix_e2ee.py` (optional)

Key points:
- runs in a background thread with its own asyncio loop
- can restrict which rooms it responds to (`RUNE_MATRIX_ROOMS`)
- keeps Matrix session info under `RUNE_DATA_DIR` (gitignored)
- task scheduler can send outbound messages using an injected messenger port

## Slash commands
Commands are defined in `cli/commands.py`.
Connectors call `command_registry.handle(...)` and can pass an `emit(...)` callback
to show immediate feedback for long operations (e.g. enabling TTS).
