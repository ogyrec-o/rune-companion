# connectors/matrix_e2ee.py

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from nio import AsyncClient, ToDeviceMessage
from nio.events.to_device import (
    KeyVerificationCancel,
    KeyVerificationKey,
    KeyVerificationMac,
    KeyVerificationStart,
    UnknownToDeviceEvent,
)

logger = logging.getLogger(__name__)


def setup_self_verification(client: AsyncClient) -> None:
    """
    Attach handlers for SAS self-verification between this bot device and another
    device of the same account (e.g. Element).

    Why "self only":
    - This code is for bootstrapping trust for the bot's own devices.
    - We intentionally do not implement verification flows with external users here.
    """
    verify_peer_by_txid: dict[str, str] = {}

    def _extract_sender(ev: Any) -> str | None:
        sender = getattr(ev, "sender", None)
        if sender:
            return sender
        src = getattr(ev, "source", None)
        if isinstance(src, dict):
            return src.get("sender")
        return None

    def _iter_events(event: Any) -> Iterable[Any]:
        # nio may pass nested lists; flatten defensively.
        if isinstance(event, list):
            for item in event:
                yield from _iter_events(item)
        else:
            yield event

    async def _handle_one_to_device(ev: Any) -> None:
        sender = _extract_sender(ev)

        # Only accept events coming from ourselves (self-verification).
        if sender is None or client.user_id is None or sender != client.user_id:
            return

        # Element may send verification request as UnknownToDeviceEvent.
        if (
            isinstance(ev, UnknownToDeviceEvent)
            and getattr(ev, "type", None) == "m.key.verification.request"
        ):
            src = ev.source or {}
            content = (src.get("content", {}) or {}) if isinstance(src, dict) else {}
            txid = content.get("transaction_id")
            peer_device = content.get("from_device")
            if not (txid and peer_device):
                return

            verify_peer_by_txid[str(txid)] = str(peer_device)

            ready = ToDeviceMessage(
                "m.key.verification.ready",
                sender,
                str(peer_device),
                {
                    "from_device": client.device_id,
                    "methods": ["m.sas.v1"],
                    "transaction_id": str(txid),
                },
            )
            await client.to_device(ready)
            logger.info("E2EE: sent READY (txid=%s, peer_device=%s)", txid, peer_device)
            return

        if isinstance(ev, KeyVerificationStart):
            txid = getattr(ev, "transaction_id", None)
            if not txid:
                return

            from_device = getattr(ev, "from_device", None)
            if from_device:
                verify_peer_by_txid[str(txid)] = str(from_device)

            await client.accept_key_verification(str(txid))

            sas = client.key_verifications.get(str(txid))
            if sas:
                await client.to_device(sas.share_key())

            logger.info("E2EE: accepted START (txid=%s)", txid)
            return

        if isinstance(ev, KeyVerificationKey):
            txid = getattr(ev, "transaction_id", None)
            if not txid:
                return

            sas = client.key_verifications.get(str(txid))
            if sas:
                # Show emoji SAS to the operator so they can confirm on the other device.
                try:
                    emojis = sas.get_emoji() or []
                    parts: list[str] = []
                    for e in emojis:
                        # nio typically returns tuples (emoji, description).
                        if isinstance(e, tuple) and len(e) >= 2:
                            parts.append(f"{e[0]} {e[1]}")
                        else:
                            sym = getattr(e, "symbol", None) or getattr(e, "emoji", None) or str(e)
                            desc = getattr(e, "description", None) or getattr(e, "name", None) or ""
                            parts.append(f"{sym} {desc}".strip())
                    if parts:
                        logger.info("E2EE: SAS %s", " | ".join(parts))
                except Exception as ex:
                    logger.warning("E2EE: failed to render SAS: %s: %r", type(ex).__name__, ex)

            await client.confirm_short_auth_string(str(txid))
            logger.info("E2EE: confirmed SAS (txid=%s)", txid)
            return

        if isinstance(ev, KeyVerificationMac):
            txid = getattr(ev, "transaction_id", None)
            if not txid:
                return

            sas = client.key_verifications.get(str(txid))
            if not sas:
                return

            await client.to_device(sas.get_mac())
            logger.info("E2EE: sent MAC (txid=%s)", txid)

            peer_device = verify_peer_by_txid.get(str(txid))
            if peer_device:
                done = ToDeviceMessage(
                    "m.key.verification.done",
                    sender,
                    peer_device,
                    {"transaction_id": str(txid)},
                )
                await client.to_device(done)
                logger.info("E2EE: sent DONE (txid=%s, peer_device=%s)", txid, peer_device)
            else:
                logger.warning("E2EE: peer_device unknown, DONE not sent (txid=%s)", txid)
            return

        if isinstance(ev, KeyVerificationCancel):
            txid = getattr(ev, "transaction_id", None)
            reason = getattr(ev, "reason", None)
            logger.info("E2EE: cancelled (txid=%s, reason=%r)", txid, reason)
            return

    async def to_device_callback(event: Any) -> None:
        try:
            for ev in _iter_events(event):
                await _handle_one_to_device(ev)
        except Exception as e:
            logger.exception("E2EE: verification handler crashed: %r", e)

    client.add_to_device_callback(
        to_device_callback,
        (
            UnknownToDeviceEvent,
            KeyVerificationStart,
            KeyVerificationKey,
            KeyVerificationMac,
            KeyVerificationCancel,
        ),
    )
