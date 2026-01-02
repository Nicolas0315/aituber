import asyncio
import logging
from typing import List, Optional

import numpy as np

from . import SpeechRecognizer

logger = logging.getLogger(__name__)


class FasterWhisperSpeechRecognizer(SpeechRecognizer):
    def __init__(
        self,
        *,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "int8",
        sample_rate: int = 16000,
        language: str = "ja",
        alternative_languages: Optional[List[str]] = None,
        vad_filter: bool = False,
        beam_size: int = 5,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 30.0,
        debug: bool = False,
    ):
        super().__init__(
            language=language,
            alternative_languages=alternative_languages,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            debug=debug,
        )
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError("faster-whisper is required for FasterWhisperSpeechRecognizer") from exc

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.sample_rate = sample_rate
        self.vad_filter = vad_filter
        self.beam_size = beam_size

    def _transcribe_sync(self, data: bytes) -> str:
        if not data:
            return ""

        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        language = None
        if self.language:
            language = self.language.split("-")[0] if "-" in self.language else self.language

        segments, _ = self.model.transcribe(
            audio,
            language=language,
            vad_filter=self.vad_filter,
            beam_size=self.beam_size,
        )

        text = "".join(segment.text for segment in segments).strip()
        if self.debug:
            logger.info(f"FasterWhisper recognized: {text}")
        return text

    async def transcribe(self, data: bytes) -> str:
        return await asyncio.to_thread(self._transcribe_sync, data)
