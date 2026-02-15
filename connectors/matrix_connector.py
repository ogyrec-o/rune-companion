# connectors/matrix_connector.py

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Set

from nio import MatrixRoom, RoomMessageText, exceptions

from commands import registry as command_registry
from config import get_settings
from connectors.matrix_client import create_matrix_client
from connectors.matrix_e2ee import setup_self_verification
from core_chat import generate_reply_text
from state import AppState
from tasks.task_models import Task, TaskStatus
from tasks.task_scheduler import DispatchPhase, TaskDispatch, run_task_scheduler

logger = logging.getLogger(__name__)


def _ms_now() -> int:
    return int(time.time() * 1000)


def _room_allowlist(settings_rooms: list[str]) -> Optional[Set[str]]:
    rooms = [r.strip() for r in (settings_rooms or []) if str(r).strip()]
    return set(rooms) if rooms else None


def _render_task_text(dispatch: TaskDispatch) -> str:
    """
    Render a task message for Matrix.

    Why this lives here:
    - The scheduler is infra-only; it decides what is runnable and which phase it is in.
    - The connector decides how to format and deliver messages in a specific transport (Matrix).
    """
    task: Task = dispatch.task

    if task.kind.startswith("ask_user"):
        if dispatch.phase == DispatchPhase.ASK:
            return (task.question_text or task.description or "").strip()

        if dispatch.phase == DispatchPhase.REPLY_BACK:
            ans = (task.answer_text or "").strip()
            if ans:
                return f"Answer received: {ans}"
            return (task.description or "").strip()

        return (task.description or "").strip()

    return (task.description or "").strip()


async def _send_text(client, *, room_id: str, text: str) -> None:
    await client.room_send(
        room_id=room_id,
        message_type="m.room.message",
        content={"msgtype": "m.text", "body": text},
        ignore_unverified_devices=True,
    )


async def _run_matrix_bot(state: AppState, stop_event: asyncio.Event) -> None:
    """
    Matrix connector (async):

    init -> scheduler -> callbacks -> sync loop

    Shutdown model:
    - main thread sets stop_event via loop.call_soon_threadsafe(stop_event.set)
    - we run a manual sync loop so we can exit promptly.
    """
    settings = get_settings()
    if not settings.matrix_enabled:
        logger.info("Matrix connector disabled via settings.")
        return

    if not settings.matrix_homeserver or not settings.matrix_user_id:
        logger.error("Matrix is enabled but not configured (homeserver/user_id).")
        return

    startup_ts = _ms_now()
    logger.info("Matrix startup timestamp (ms): %d", startup_ts)

    allowed_rooms = _room_allowlist(getattr(settings, "matrix_rooms", []) or [])
    logger.info("Matrix allowed_rooms=%s", allowed_rooms if allowed_rooms is not None else "ALL")

    client = await create_matrix_client()
    if client is None:
        logger.error("Matrix client creation failed; connector will stop.")
        return

    logger.info(
        "Matrix client started (user=%s, homeserver=%s).",
        settings.matrix_user_id,
        settings.matrix_homeserver,
    )

    # Self-verification handlers (SAS). Safe no-op if E2EE is off.
    try:
        setup_self_verification(client)
    except Exception:
        logger.exception("Failed to attach E2EE verification handlers.")

    # ---- Task scheduler ----

    async def send_task_dispatch(dispatch: TaskDispatch) -> None:
        text = _render_task_text(dispatch)
        if not text:
            logger.debug("Task %s: empty text -> skipped", dispatch.task.id)
            return

        room_id = (dispatch.room_id or "").strip()

        # If task has no room_id, pick:
        # - first allowed room
        # - otherwise any joined room
        if not room_id:
            if allowed_rooms:
                room_id = next(iter(allowed_rooms))
            elif client.rooms:
                room_id = next(iter(client.rooms.keys()))

        if not room_id:
            logger.warning("Task %s: no room_id -> skipped", dispatch.task.id)
            return

        try:
            await _send_text(client, room_id=room_id, text=text)
            logger.info("Task %s sent to room %s (phase=%s).", dispatch.task.id, room_id, dispatch.phase.value)
        except Exception:
            logger.exception("Failed to send task %s to room %s.", dispatch.task.id, room_id)

    scheduler_task = asyncio.create_task(
        run_task_scheduler(
            state.task_store,
            send_task_dispatch,
            interval_seconds=15.0,
        )
    )

    # ---- Message callback ----

    lock = getattr(state, "lock", None)

    async def message_callback(room: MatrixRoom, event: RoomMessageText) -> None:
        # 1) Ignore messages sent before bot startup.
        ts = getattr(event, "server_timestamp", None)
        if ts is not None and ts <= startup_ts:
            return

        # 2) Ignore own messages.
        if event.sender == client.user_id:
            return

        # 3) Room allowlist filter.
        if allowed_rooms is not None and room.room_id not in allowed_rooms:
            return

        body = (event.body or "").strip()
        if not body:
            return

        encrypted_flag = " (encrypted)" if getattr(event, "decrypted", False) else ""
        logger.info("Matrix <%s> %s%s: %r", room.display_name, event.sender, encrypted_flag, body)

        # Commands (/help, /tts, ...)
        if body.startswith("/"):
            try:
                if lock:
                    with lock:
                        resp = command_registry.handle(state, body, user_id=event.sender, room_id=room.room_id)
                else:
                    resp = command_registry.handle(state, body, user_id=event.sender, room_id=room.room_id)
            except Exception:
                logger.exception("Command handler crashed.")
                resp = "Internal error while handling a command."

            if resp:
                try:
                    await _send_text(client, room_id=room.room_id, text=resp)
                except exceptions.OlmUnverifiedDeviceError:
                    logger.warning("Cannot send command reply: unverified device.")
                except Exception:
                    logger.exception("Failed to send command reply.")
            return

        # Normal message: typing + generate reply
        try:
            try:
                await client.room_typing(room.room_id, typing_state=True, timeout=30000)
            except Exception:
                logger.debug("Failed to send typing notification.", exc_info=True)

            # NOTE: generate_reply_text is synchronous and may take time.
            # We keep it in the same thread and guard shared state with a lock.
            if lock:
                with lock:
                    reply = generate_reply_text(state, body, user_id=event.sender, room_id=room.room_id)
            else:
                reply = generate_reply_text(state, body, user_id=event.sender, room_id=room.room_id)

            reply = (reply or "").strip()
            if not reply:
                with contextlib.suppress(Exception):
                    await client.room_typing(room.room_id, typing_state=False, timeout=30000)
                return

            # Simulate typing speed: ~25 chars/sec, clamp 1..8 seconds.
            typing_delay = max(1.0, min(len(reply) / 25.0, 8.0))
            await asyncio.sleep(typing_delay)

            try:
                await client.room_typing(room.room_id, typing_state=False, timeout=30000)
            except Exception:
                logger.debug("Failed to stop typing notification.", exc_info=True)

            await _send_text(client, room_id=room.room_id, text=reply)
            logger.info("Replied in %s (%s).", room.display_name, room.room_id)

        except exceptions.OlmUnverifiedDeviceError:
            logger.warning("Cannot send reply: unverified device.")
        except Exception:
            logger.exception("Failed to handle Matrix message.")
            with contextlib.suppress(Exception):
                await client.room_typing(room.room_id, typing_state=False, timeout=30000)

    client.add_event_callback(message_callback, RoomMessageText)

    # ---- Sync loop ----

    try:
        logger.info("Matrix initial sync...")
        await client.sync(timeout=30000, full_state=True)
        logger.info("Matrix initial sync done. Joined rooms: %d", len(client.rooms))

        logger.info("Matrix sync loop started.")
        while not stop_event.is_set():
            await client.sync(timeout=30000, full_state=False)

    except asyncio.CancelledError:
        logger.info("Matrix connector cancelled.")
    except Exception:
        logger.exception("Matrix connector crashed.")
    finally:
        scheduler_task.cancel()
        with contextlib.suppress(Exception):
            await scheduler_task

        with contextlib.suppress(Exception):
            await client.close()

        logger.info("Matrix connector stopped.")


@dataclass
class MatrixBackgroundRunner:
    thread: threading.Thread
    loop: asyncio.AbstractEventLoop
    stop_event: asyncio.Event

    def stop(self) -> None:
        try:
            self.loop.call_soon_threadsafe(self.stop_event.set)
        except Exception:
            logger.debug("Failed to signal Matrix stop.", exc_info=True)

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)


def start_matrix_in_background(state: AppState) -> MatrixBackgroundRunner | None:
    """
    Start Matrix connector in a background thread (so console REPL can run in parallel).

    Why a thread:
    - console REPL is blocking (input()).
    - Matrix connector is async and wants its own event loop.
    """
    settings = get_settings()
    if not settings.matrix_enabled:
        logger.info("Matrix connector disabled, not starting.")
        return None

    ready = threading.Event()
    holder: dict[str, object] = {}

    def runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stop_event = asyncio.Event()

        holder["loop"] = loop
        holder["stop_event"] = stop_event
        ready.set()

        try:
            loop.run_until_complete(_run_matrix_bot(state, stop_event))
        finally:
            with contextlib.suppress(Exception):
                loop.stop()
            with contextlib.suppress(Exception):
                loop.close()

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    ready.wait(timeout=5.0)
    loop = holder.get("loop")
    stop_event = holder.get("stop_event")

    if not isinstance(loop, asyncio.AbstractEventLoop) or not isinstance(stop_event, asyncio.Event):
        logger.error("Matrix thread did not initialize properly.")
        return None

    logger.info("Matrix background thread started.")
    return MatrixBackgroundRunner(thread=t, loop=loop, stop_event=stop_event)
