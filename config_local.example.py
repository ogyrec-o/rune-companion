# config_local.example.py

"""
Example local overrides.

Usage:
  1) Copy this file to `config_local.py`
  2) Adjust values for your machine
  3) Never commit `config_local.py` (it is gitignored)

Prefer `.env` for secrets. This file should contain only safe overrides.
"""

# Example: enable Matrix connector locally
# MATRIX_ENABLED = True

# Example: change model order (if your config.py supports overriding LLM_MODELS from python)
# LLM_MODELS = [
#     "qwen/qwen-2.5-72b-instruct:free",
# ]

# Example: override local paths (prefer env vars; only do this if you really need it)
# from pathlib import Path
# DATA_DIR = Path(".local/rune")
# MEMORY_DB_PATH = DATA_DIR / "memory.sqlite3"
