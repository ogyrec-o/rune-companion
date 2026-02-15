# tts_engine.py

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSConfig:
    """Runtime TTS config resolved from settings/env."""
    speaker_wav: Optional[str]
    xtts_speaker_name: str
    xtts_language: str


def _resolve_tts_config() -> TTSConfig:
    """
    Resolve TTS config from settings if available, otherwise fall back to legacy config constants.
    This keeps the module import-safe and compatible during refactors.
    """
    # Preferred: settings object
    try:
        from config import get_settings  # type: ignore

        s = get_settings()
        return TTSConfig(
            speaker_wav=getattr(s, "speaker_wav", None) or getattr(s, "tts_speaker_wav", None),
            xtts_speaker_name=getattr(s, "xtts_speaker_name", "Ana Florence"),
            xtts_language=getattr(s, "xtts_language", "en"),
        )
    except Exception:
        pass

    # Legacy fallback: config.py constants
    try:
        from config import SPEAKER_WAV, XTTS_LANGUAGE, XTTS_SPEAKER_NAME  # type: ignore

        return TTSConfig(
            speaker_wav=SPEAKER_WAV or None,
            xtts_speaker_name=XTTS_SPEAKER_NAME or "Ana Florence",
            xtts_language=XTTS_LANGUAGE or "en",
        )
    except Exception:
        return TTSConfig(
            speaker_wav=None,
            xtts_speaker_name="Ana Florence",
            xtts_language="en",
        )


class TTSEngine:
    """
    Best-effort text-to-speech engine.

    Design goals:
    - Optional dependencies (does not crash if TTS libs are not installed).
    - Does not block the main thread: synthesis and playback happen in a worker thread.
    - Friendly logging and safe shutdown.

    Notes:
    - If TTS is enabled but dependencies are missing, this engine disables itself.
    - Speaker WAV is optional; if missing/unreadable, we fall back to a named speaker.
    """

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)

        self._queue: Optional["queue.Queue[str | None]"] = None
        self._worker: Optional[threading.Thread] = None

        self._tts_model: Any = None
        self._sample_rate: Optional[int] = None

        self._sd: Any = None  # sounddevice module (runtime import)
        self._stop_requested = False

        if not self.enabled:
            logger.info("TTS disabled.")
            return

        # NOTE: imports can be slow, log before doing them so the user isn't stuck in silence.
        logger.info("TTS enabling: importing dependencies (torch/TTS/sounddevice)... this may take a while.")

        # Lazy / optional imports
        try:
            import torch  # type: ignore
            from TTS.api import TTS  # type: ignore
            import sounddevice as sd  # type: ignore
        except Exception as e:
            self.enabled = False
            logger.warning(
                "TTS is enabled, but dependencies are missing or failed to import. "
                "Install torch + TTS + sounddevice to enable TTS. Error: %s",
                repr(e),
            )
            return

        logger.info("TTS dependencies imported. Initializing XTTS model...")

        self._sd = sd

        cfg = _resolve_tts_config()

        # Initialize XTTS model
        try:
            device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
            logger.info("Initializing XTTS (device=%s). First run may download large model files.", device)

            self._tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

            # Best-effort sample rate extraction
            sr = None
            try:
                sr = int(self._tts_model.synthesizer.output_sample_rate)
            except Exception:
                sr = None
            self._sample_rate = sr or 24000

        except Exception as e:
            self.enabled = False
            logger.error("Failed to initialize XTTS model: %s", repr(e))
            return

        self._queue = queue.Queue()

        def audio_worker() -> None:
            logger.info("TTS worker thread started.")
            assert self._queue is not None

            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        logger.info("TTS worker received stop signal.")
                        return

                    text = " ".join(str(item).split()).strip()
                    if not text:
                        continue

                    wav_path = (cfg.speaker_wav or "").strip()
                    use_wav = False
                    wav_file = None

                    if wav_path:
                        p = Path(wav_path)
                        if p.exists() and p.is_file():
                            use_wav = True
                            wav_file = str(p)
                        else:
                            logger.warning(
                                "speaker_wav is set but file does not exist: %s. Falling back to speaker name.",
                                wav_path,
                            )

                    try:
                        if use_wav and wav_file:
                            audio = self._tts_model.tts(
                                text=text,
                                language=cfg.xtts_language,
                                speaker_wav=wav_file,
                            )
                        else:
                            audio = self._tts_model.tts(
                                text=text,
                                language=cfg.xtts_language,
                                speaker=cfg.xtts_speaker_name,
                            )
                    except Exception as e:
                        logger.error("TTS synthesis failed: %s", repr(e))
                        continue

                    try:
                        self._sd.play(audio, self._sample_rate)
                        self._sd.wait()
                    except Exception as e:
                        logger.error("TTS playback failed: %s", repr(e))

                finally:
                    self._queue.task_done()

        self._worker = threading.Thread(target=audio_worker, daemon=True)
        self._worker.start()

        logger.info("TTS ready (sample_rate=%s).", self._sample_rate)

    def speak_sentence(self, text: str) -> None:
        """Queue a sentence for playback (no-op if disabled)."""
        if not self.enabled or self._queue is None:
            return
        self._queue.put(text)

    def wait_all(self) -> None:
        """Block until all queued items are processed (no-op if disabled)."""
        if not self.enabled or self._queue is None:
            return
        self._queue.join()

    def shutdown(self) -> None:
        """Request a clean shutdown of the worker (no-op if disabled)."""
        if not self.enabled or self._queue is None:
            return
        if self._stop_requested:
            return
        self._stop_requested = True

        logger.info("Stopping TTS worker...")
        self._queue.put(None)
        self._queue.join()

        if self._worker is not None:
            self._worker.join(timeout=2.0)

        logger.info("TTS stopped.")
