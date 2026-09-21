from __future__ import annotations

import asyncio
import io
import soundfile as sf
from transformers import pipeline

from src.voice.contracts import STTService


class WhisperSTTService(STTService):
    """Concrete STT adapter using HuggingFace Whisper."""

    def __init__(self, model_name: str = "openai/whisper-tiny") -> None:
        self.model_name = model_name
        self._pipeline = None

    async def initialize(self) -> None:
        if self._pipeline is None:
            loop = asyncio.get_running_loop()
            self._pipeline = await loop.run_in_executor(
                None,
                lambda: pipeline("automatic-speech-recognition", model=self.model_name),
            )

    async def transcribe(self, audio_bytes: bytes, media_type: str = "audio/wav") -> str:
        if not audio_bytes or len(audio_bytes) < 100:
            raise ValueError("Audio input is empty or too short")

        if self._pipeline is None:
            await self.initialize()

        loop = asyncio.get_running_loop()

        def _run_transcription() -> str:
            import os
            import tempfile

            generate_kwargs = {"language": "en", "task": "transcribe"}

            try:
                with io.BytesIO(audio_bytes) as bio:
                    data, samplerate = sf.read(bio)
                    if len(data.shape) > 1:
                        data = data.mean(axis=1)
                    result = self._pipeline(
                        {"raw": data.astype("float32"), "sampling_rate": samplerate},
                        generate_kwargs=generate_kwargs,
                    )
                    text = (result.get("text") or "").strip()
                    if text:
                        return text
            except Exception:
                pass

            try:
                suffix = ".wav" if "wav" in media_type else ".tmp"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                    tf.write(audio_bytes)
                    tmp_name = tf.name
                try:
                    result = self._pipeline(tmp_name, generate_kwargs=generate_kwargs)
                    text = (result.get("text") or "").strip()
                    if text:
                        return text
                finally:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
            except Exception:
                pass

            try:
                result = self._pipeline(audio_bytes, generate_kwargs=generate_kwargs)
                text = (result.get("text") or "").strip()
                if text:
                    return text
            except Exception as exc:
                raise ValueError(f"Could not decode audio or detect speech: {exc}")

        transcript = await loop.run_in_executor(None, _run_transcription)
        if not transcript:
            raise ValueError("No understandable speech was detected")
        return transcript

    async def cleanup(self) -> None:
        self._pipeline = None
