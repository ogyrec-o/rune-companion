# src/rune_companion/connectors/matrix_client.py

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from nio import AsyncClient, AsyncClientConfig, LoginResponse

logger = logging.getLogger(__name__)

try:
    import olm  # type: ignore  # noqa: F401

    OLM_AVAILABLE = True
except Exception:
    OLM_AVAILABLE = False


def _session_path(store_dir: Path) -> Path:
    # Keep session tokens in a single predictable place under a gitignored local dir.
    return store_dir / "session.json"


def _safe_mkdir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("Failed to create directory %s: %r", path, e)


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_text("utf-8")
    val = json.loads(raw)
    if isinstance(val, dict):
        return val
    raise ValueError("Expected JSON object")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except Exception:
        # Best-effort: not critical on Windows or restricted FS.
        pass


async def create_matrix_client(settings) -> AsyncClient | None:
    """
    Create a Matrix AsyncClient with optional E2EE support.

    Why we persist session.json:
    - It allows reusing the access token/device id across restarts without logging in again.
    - The file contains sensitive data and must never be committed (store under a gitignored dir).
    """
    homeserver = (getattr(settings, "matrix_homeserver", "") or "").strip()
    user_id = (getattr(settings, "matrix_user_id", "") or "").strip()
    password = (getattr(settings, "matrix_password", "") or "").strip()
    store_dir = Path(getattr(settings, "matrix_store_path", Path(".local/rune/matrix_store")))

    if not homeserver or not user_id:
        logger.error("Matrix is not configured: set RUNE_MATRIX_HOMESERVER and RUNE_MATRIX_USER_ID")
        return None

    # We keep session.json even without olm; encryption store is only enabled if olm is available.
    _safe_mkdir(store_dir)
    session_file = _session_path(store_dir)

    encryption_enabled = bool(OLM_AVAILABLE)
    if encryption_enabled:
        logger.info("python-olm detected: E2EE enabled")
    else:
        logger.warning("python-olm not installed: E2EE disabled")

    config = AsyncClientConfig(
        encryption_enabled=encryption_enabled,
        store_sync_tokens=True,
    )

    client = AsyncClient(
        homeserver,
        user_id,
        store_path=str(store_dir) if encryption_enabled else None,
        config=config,
    )

    # ---- Session restore ----
    if session_file.exists():
        try:
            data = _load_json(session_file)

            access_token = data.get("access_token")
            sess_user_id = data.get("user_id")
            device_id = data.get("device_id")

            if not access_token or not sess_user_id or not device_id:
                raise ValueError("session.json is missing required fields")

            client.access_token = str(access_token)
            client.user_id = str(sess_user_id)
            client.device_id = str(device_id)

            if encryption_enabled:
                try:
                    client.load_store()
                    logger.info("Matrix session restored for %s (E2EE store loaded)", client.user_id)
                except Exception as e:
                    # We can still operate without the local crypto store, but E2EE will be limited.
                    logger.warning("Failed to load E2EE store: %r", e)
            else:
                logger.info("Matrix session restored for %s (E2EE disabled)", client.user_id)

            return client
        except Exception as e:
            logger.warning("Failed to restore Matrix session.json, will try password login: %r", e)

    # ---- Password login bootstrap ----
    if not password:
        logger.error(
            "Matrix session.json not found and password is not set. "
            "Set RUNE_MATRIX_PASSWORD once to bootstrap a session."
        )
        return None

    device_name = f"{getattr(settings, 'app_name', 'rune')} (Python)"
    logger.info("Logging in to Matrix to bootstrap a new session (device_name=%r)...", device_name)

    resp = await client.login(password=password, device_name=device_name)

    if not isinstance(resp, LoginResponse):
        logger.error("Matrix login failed: %r", resp)
        return None

    session_data = {
        "access_token": resp.access_token,
        "user_id": resp.user_id,
        "device_id": resp.device_id,
    }

    try:
        _atomic_write_json(session_file, session_data)
        logger.info("Matrix session saved to %s (user=%s)", session_file, resp.user_id)
    except Exception as e:
        logger.error("Failed to write Matrix session.json (%s): %r", session_file, e)
        return None

    return client
