from __future__ import annotations

from src.voice.contracts import STTService, TTSService


class FakeSTT(STTService):
    """Fake STT service for unit testing."""

    def __init__(
        self,
        transcript: str = "My payment was charged twice",
        fail: bool = False,
        empty: bool = False,
    ) -> None:
        self.transcript = transcript
        self.fail = fail
        self.empty = empty

    async def initialize(self) -> None:
        pass

    async def transcribe(self, audio_bytes: bytes, media_type: str = "audio/wav") -> str:
        if self.fail:
            raise RuntimeError("STT transcription engine failed")
        if self.empty:
            return ""
        return self.transcript

    async def cleanup(self) -> None:
        pass


class FakeTTS(TTSService):
    """Fake TTS service for unit testing."""

    def __init__(
        self,
        audio_bytes: bytes = b"fake-audio-bytes-mp3",
        media_type: str = "audio/mpeg",
        fail: bool = False,
        empty: bool = False,
    ) -> None:
        self.audio_bytes = audio_bytes
        self.media_type = media_type
        self.fail = fail
        self.empty = empty

    async def initialize(self) -> None:
        pass

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        if self.fail:
            raise RuntimeError("TTS synthesis engine failed")
        if self.empty:
            return b"", self.media_type
        return self.audio_bytes, self.media_type

    async def cleanup(self) -> None:
        pass
