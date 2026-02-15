"""
Task subsystem.

Components:
- task_models.py: data structures (Task, TaskStatus)
- task_store.py: SQLite-backed storage + query/update helpers
- task_scheduler.py: polling scheduler that dispatches runnable tasks
- task_api.py: small high-level helpers used by the rest of the app
"""
