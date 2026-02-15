# Memory subsystem

The memory subsystem provides:
- long-term memory items (facts) stored in SQLite,
- optional episodic summaries,
- a memory controller that decides what to store/update/delete,
- task injection so the assistant can honor open requests/promises.

## Memory items
Memories are small fact-like records with:
- subject scope (user / room / relationship / global),
- text,
- importance score,
- tags (e.g. `promise`, `todo`, `episode`, ...),
- timestamps.

## Prompt injection
Before calling the LLM, the core builds a `<MEMORY>...</MEMORY>` block that contains:
- top memories relevant to the current dialog,
- open tasks for the current user (if any).

This block is treated as internal context.

## Episodic summaries
If dialog history is enabled, the core can periodically summarize a recent chunk
and store it as memory ("what happened recently") to keep the memory store compact.

## Memory controller
The controller periodically:
- looks at recent messages + existing memories,
- proposes add/update/delete operations,
- may create tasks (e.g. reminders or ask-user flows).

The controller output is applied defensively (best-effort) to avoid polluting memory.

## Tasks in memory context
Open tasks are injected into the prompt to help the assistant:
- remember pending promises,
- ask for missing info when needed,
- follow up proactively.
