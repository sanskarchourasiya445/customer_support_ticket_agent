from __future__ import annotations

import edge_tts

from src.voice.contracts import TTSService


class EdgeTTSService(TTSService):
    """Concrete TTS adapter using Microsoft Edge Neural voices."""

    def __init__(self, voice: str = "en-US-JennyNeural") -> None:
        self.voice = voice

    async def initialize(self) -> None:
        pass

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Text input is empty")

        communicate = edge_tts.Communicate(cleaned_text, self.voice)
        audio_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        audio_bytes = b"".join(audio_chunks)
        if not audio_bytes:
            raise RuntimeError("Edge TTS produced empty audio")

        return audio_bytes, "audio/mpeg"

    async def cleanup(self) -> None:
        pass
