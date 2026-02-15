# src/rune_companion/llm/client.py

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import openai
from openai import OpenAI

from ..config import get_settings

logger = logging.getLogger(__name__)

_BAD_MODELS: dict[str, float] = {}  # model -> retry_at (monotonic)

_client: OpenAI | None = None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    s = raw.strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _timeouts_from_env() -> dict[str, float]:
    """
    Timeouts are intentionally configurable via env to avoid "hanging forever"
    when a model has a long time-to-first-token.

    Defaults are conservative for local console UX:
    - connect timeout: 5s
    - read timeout: 25s (no data from server)
    - first token timeout: 20s (no content tokens)
    """
    first_token = _env_float("RUNE_LLM_FIRST_TOKEN_TIMEOUT_SECONDS", 20.0)
    read_timeout = _env_float("RUNE_LLM_READ_TIMEOUT_SECONDS", 25.0)
    connect_timeout = _env_float("RUNE_LLM_CONNECT_TIMEOUT_SECONDS", 5.0)

    # keep read >= first_token as a sane baseline
    read_timeout = max(read_timeout, first_token)

    return {
        "first_token": first_token,
        "read": read_timeout,
        "connect": connect_timeout,
    }


def _is_auth_error(exc: Exception) -> bool:
    return exc.__class__.__name__ in {
        "AuthenticationError",
        "PermissionDeniedError",
        "UnauthorizedError",
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    return exc.__class__.__name__ in {"RateLimitError", "TooManyRequestsError"}


def _is_connection_error(exc: Exception) -> bool:
    return exc.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "Timeout",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
    }


def _is_not_found_error(exc: Exception) -> bool:
    # OpenAI-compatible SDK often uses NotFoundError for HTTP 404
    return exc.__class__.__name__ in {"NotFoundError"}


def _make_timeout_obj(connect_s: float, read_s: float) -> Any:
    """
    Best-effort: build an httpx.Timeout object if httpx is available.
    If not, fall back to a plain float (some SDK versions accept it).
    """
    try:
        import httpx  # type: ignore

        return httpx.Timeout(
            connect=connect_s,
            read=read_s,
            write=10.0,
            pool=connect_s,
        )
    except Exception:
        # Fallback: a float may still be respected by some client versions
        return float(read_s)


def _get_client() -> OpenAI:
    """
    Lazily create and cache the OpenAI-compatible client.

    IMPORTANT:
    - No secrets required at import time.
    - We disable automatic retries to allow quick fallback across models.
    """
    global _client
    if _client is not None:
        return _client

    settings = get_settings()

    api_key = getattr(settings, "openrouter_api_key", None)
    base_url = getattr(settings, "openrouter_base_url", "") or ""

    if not api_key or not str(api_key).strip():
        raise RuntimeError("LLM API key is not set. Set RUNE_OPENROUTER_API_KEY in your .env.")

    if not base_url.strip():
        raise RuntimeError("LLM base URL is not set. Set RUNE_OPENROUTER_BASE_URL in your .env.")

    t = _timeouts_from_env()
    timeout_obj = _make_timeout_obj(connect_s=t["connect"], read_s=t["read"])

    # Some SDK versions accept timeout/max_retries in the constructor, some might not.
    try:
        _client = OpenAI(
            base_url=str(base_url),
            api_key=str(api_key),
            timeout=timeout_obj,
            max_retries=0,
        )
    except TypeError:
        _client = OpenAI(
            base_url=str(base_url),
            api_key=str(api_key),
        )
    return _client


def friendly_llm_error_message(err: Exception) -> str:
    msg = str(err).strip() or "LLM error."
    if "LLM API key is not set" in msg or "Missing OpenRouter API key" in msg:
        return "LLM is not configured (missing API key). Set RUNE_OPENROUTER_API_KEY in .env (see .env.example)."
    if "LLM model list is empty" in msg:
        return "LLM is not configured (no models). Set RUNE_LLM_MODELS in .env (see .env.example)."
    if "LLM base URL is not set" in msg:
        return "LLM is not configured (missing base URL). Set RUNE_OPENROUTER_BASE_URL in .env (see .env.example)."
    return msg


def _close_stream(stream: Any) -> None:
    """
    Best-effort close for streaming responses.
    Not all SDK versions expose close(), so keep it defensive.
    """
    try:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def _create_stream(client: OpenAI, *, model: str, headers: Dict[str, str], messages: list[dict[str, str]], timeout: Any) -> Any:
    """
    Create a streaming chat completion.

    Some SDK versions accept per-request timeout=..., others might not.
    """
    try:
        return client.chat.completions.create(
            model=model,
            stream=True,
            extra_headers=headers or None,
            messages=messages,
            timeout=timeout,
        )
    except TypeError:
        return client.chat.completions.create(
            model=model,
            stream=True,
            extra_headers=headers or None,
            messages=messages,
        )


def stream_chat_chunks(
        messages: List[Dict[str, str]],
        system_prompt: str,
) -> Iterable[str]:
    """
    Stream the LLM response in text chunks.

    Behavior:
    - Tries models in the order from env (RUNE_LLM_MODELS).
    - If a model doesn't produce a first content token within FIRST_TOKEN timeout,
      we abort and try the next model.
    - 404 (model not available) -> try next.
    - Rate limit / network issues -> try next.
    - Auth issues -> fail fast (no retries across models).
    """
    settings = get_settings()
    models: List[str] = list(getattr(settings, "llm_models", []) or [])
    headers: Dict[str, str] = dict(getattr(settings, "extra_headers", {}) or {})

    if not models:
        raise RuntimeError("LLM model list is empty. Set RUNE_LLM_MODELS in your .env.")

    client = _get_client()

    t = _timeouts_from_env()
    first_token_timeout = float(t["first_token"])
    timeout_obj = _make_timeout_obj(connect_s=t["connect"], read_s=t["read"])

    last_error: Optional[Exception] = None

    now = time.monotonic()

    for model in models:
        model = (model or "").strip()
        if not model:
            continue

        retry_at = _BAD_MODELS.get(model)
        if retry_at is not None and retry_at > now:
            continue

        logger.info(
            "LLM: trying model=%s (first_token_timeout=%.1fs, read_timeout=%.1fs)",
            model,
            first_token_timeout,
            float(t["read"]),
        )
        t0 = time.monotonic()
        deadline = t0 + first_token_timeout

        stream = None
        used_any = False

        try:
            stream = _create_stream(
                client,
                model=model,
                headers=headers,
                messages=[{"role": "system", "content": system_prompt}, *messages],
                timeout=timeout_obj,
            )

            for chunk in stream:
                # If the SDK yields chunks but no content (rare), still enforce first-token deadline.
                if not used_any and time.monotonic() > deadline:
                    last_error = TimeoutError(f"First token timeout on model: {model}")
                    logger.info("LLM: first token timeout on model=%s -> trying next", model)
                    break

                try:
                    choice0 = chunk.choices[0]
                    delta = getattr(choice0, "delta", None)
                    content = getattr(delta, "content", None) if delta is not None else None
                except Exception:
                    content = None

                if content:
                    if not used_any:
                        logger.info("LLM: first token from model=%s (%.2fs)", model, time.monotonic() - t0)
                    used_any = True
                    yield content

            if used_any:
                logger.debug("LLM: completed with model=%s", model)
                return

            # No content produced.
            if last_error is None:
                last_error = RuntimeError(f"Model returned no content: {model}")

        except Exception as e:
            last_error = e

            if _is_auth_error(e):
                raise RuntimeError(
                    "LLM authentication failed. Check your API key (RUNE_OPENROUTER_API_KEY)."
                ) from e

            if _is_not_found_error(e):
                _BAD_MODELS[model] = time.monotonic() + 3600.0  # 1 hour
                logger.info("LLM: model not available (404): %s", model)
                continue

            if _is_rate_limit_error(e):
                logger.info("LLM: rate-limited on model=%s, trying next", model)
                continue

            if _is_connection_error(e):
                logger.info("LLM: network/timeout error on model=%s, trying next", model)
                continue

            logger.info("LLM: error on model=%s (%s), trying next", model, e.__class__.__name__)
            continue

        finally:
            if stream is not None:
                _close_stream(stream)

    if last_error is not None:
        if _is_rate_limit_error(last_error):
            raise RuntimeError("LLM is rate-limited. Try again later.") from last_error
        if _is_connection_error(last_error):
            raise RuntimeError("LLM network/timeout error. Try again later or change models.") from last_error
        raise RuntimeError("All LLM models failed.") from last_error

    raise RuntimeError("All LLM models failed.")
