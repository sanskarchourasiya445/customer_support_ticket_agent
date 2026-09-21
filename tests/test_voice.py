import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.server import app, voice_pipeline
from src.voice.contracts import STTService, TTSService
from src.voice.fakes import FakeSTT, FakeTTS
from src.voice.models import SynthesisRequest, TranscriptionResponse
from src.voice.pipeline import VoicePipeline
from src.voice.stt import WhisperSTTService
from src.voice.tts import EdgeTTSService


@pytest.fixture
def fake_voice_pipeline():
    orig_stt = voice_pipeline.stt
    orig_tts = voice_pipeline.tts

    fake_stt = FakeSTT(transcript="My payment was charged twice")
    fake_tts = FakeTTS(audio_bytes=b"fake-mp3-bytes", media_type="audio/mpeg")

    voice_pipeline.stt = fake_stt
    voice_pipeline.tts = fake_tts

    yield voice_pipeline

    voice_pipeline.stt = orig_stt
    voice_pipeline.tts = orig_tts


@pytest.fixture
def voice_client(fake_voice_pipeline):
    with TestClient(app) as client:
        yield client


@pytest.mark.asyncio
async def test_voice_pipeline_fake_transcribe_and_synthesize() -> None:
    pipeline = VoicePipeline(
        FakeSTT(transcript="How long does standard shipping take?"),
        FakeTTS(audio_bytes=b"audio-bytes", media_type="audio/mpeg"),
    )
    await pipeline.initialize()

    transcript, t_time = await pipeline.transcribe(b"fake-audio-bytes", "audio/wav")
    assert transcript == "How long does standard shipping take?"
    assert t_time >= 0

    audio, media_type, s_time = await pipeline.synthesize("Standard shipping takes 3-5 days.")
    assert audio == b"audio-bytes"
    assert media_type == "audio/mpeg"
    assert s_time >= 0

    await pipeline.cleanup()


@pytest.mark.asyncio
async def test_voice_pipeline_empty_audio_raises_error() -> None:
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    with pytest.raises(ValueError, match="Audio input is empty"):
        await pipeline.transcribe(b"")


@pytest.mark.asyncio
async def test_voice_pipeline_empty_text_raises_error() -> None:
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    with pytest.raises(ValueError, match="Text input is empty"):
        await pipeline.synthesize("   ")


@pytest.mark.asyncio
async def test_whisper_stt_empty_audio_raises() -> None:
    stt = WhisperSTTService()
    with pytest.raises(ValueError, match="Audio input is empty or too short"):
        await stt.transcribe(b"")


@pytest.mark.asyncio
async def test_edge_tts_empty_text_raises() -> None:
    tts = EdgeTTSService()
    with pytest.raises(ValueError, match="Text input is empty"):
        await tts.synthesize("   ")


def test_api_transcribe_multipart_success(voice_client: TestClient) -> None:
    audio_file = io.BytesIO(b"fake-valid-audio-data-at-least-some-bytes")
    files = {"file": ("test.wav", audio_file, "audio/wav")}

    response = voice_client.post("/voice/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["transcript"] == "My payment was charged twice"
    assert data["processing_time_ms"] >= 0


def test_api_transcribe_raw_binary_success(voice_client: TestClient) -> None:
    headers = {"Content-Type": "audio/wav"}
    response = voice_client.post(
        "/voice/transcribe",
        content=b"fake-valid-audio-data-at-least-some-bytes",
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["transcript"] == "My payment was charged twice"


def test_api_transcribe_empty_audio_returns_400(voice_client: TestClient) -> None:
    response = voice_client.post("/voice/transcribe", content=b"")
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "Audio input is empty" in data["error"]


def test_api_transcribe_no_speech_detected_returns_400(voice_client: TestClient) -> None:
    class FailingSTT(STTService):
        async def initialize(self) -> None:
            pass

        async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
            return ""

        async def cleanup(self) -> None:
            pass

    voice_pipeline.stt = FailingSTT()
    response = voice_client.post(
        "/voice/transcribe",
        content=b"some-noise-audio-bytes-long-enough",
        headers={"Content-Type": "audio/wav"},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "No understandable speech was detected" in data["error"]


def test_api_synthesize_success(voice_client: TestClient) -> None:
    payload = {
        "message_id": "msg-12345",
        "text": "Your ticket CST-2026-0001 has been created.",
    }
    response = voice_client.post("/voice/synthesize", json=payload)
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("audio/mpeg")
    assert response.headers["X-Message-ID"] == "msg-12345"
    assert "X-Processing-Time-MS" in response.headers
    assert response.content == b"fake-mp3-bytes"


def test_api_synthesize_empty_text_returns_400(voice_client: TestClient) -> None:
    payload = {
        "message_id": "msg-12345",
        "text": "   ",
    }
    response = voice_client.post("/voice/synthesize", json=payload)
    assert response.status_code == 400


def test_api_synthesize_failure_returns_500(voice_client: TestClient) -> None:
    class ErrorTTS(TTSService):
        async def initialize(self) -> None:
            pass

        async def synthesize(self, text: str) -> tuple[bytes, str]:
            raise RuntimeError("TTS upstream service error")

        async def cleanup(self) -> None:
            pass

    voice_pipeline.tts = ErrorTTS()
    payload = {
        "message_id": "msg-12345",
        "text": "Hello world",
    }
    response = voice_client.post("/voice/synthesize", json=payload)
    assert response.status_code == 500
    assert "Synthesis failed" in response.json()["detail"]
