# Contributing

Thanks for your interest in contributing!

## Development setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running checks
```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src/rune_companion
python -m pytest
```

## Code style
- Production code, docs, and comments must be in English.
- Keep modules small and responsibilities clear.
- Prefer typed public APIs and deterministic behavior.

## Pull requests
- Keep PRs focused and reviewable.
- Include tests for behavioral changes.
- Update CHANGELOG.md for user-facing changes.

## Versioning / Releases
- This project follows SemVer.
- Tag releases as vMAJOR.MINOR.PATCH (e.g. v0.2.0).
