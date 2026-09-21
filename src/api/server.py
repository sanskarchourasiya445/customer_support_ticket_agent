from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from src.config import load_settings
from src.models import ChatRequest, ChatResponse, Ticket
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError
from src.voice import EdgeTTSService, SynthesisRequest, TranscriptionResponse, VoicePipeline, WhisperSTTService


settings = load_settings()
pipeline = SupportPipeline(settings, Path(__file__).resolve().parents[2] / "knowledge_base")
voice_pipeline = VoicePipeline(stt=WhisperSTTService(), tts=EdgeTTSService())


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup owns expensive shared initialization. Request handlers reuse the
    # resulting components, while shutdown makes readiness false immediately.
    await pipeline.initialize()
    yield
    pipeline.ready = False
    await voice_pipeline.cleanup()


app = FastAPI(title="Customer Support Ticket Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    if not pipeline.ready:
        raise HTTPException(status_code=503, detail="Support pipeline is not ready")
    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # Keep this transport boundary thin: Pydantic validates the public request,
    # the pipeline owns orchestration, and known service errors are translated
    # to stable HTTP responses here.
    try:
        return await pipeline.process(request.session_id, request.message)
    except ComponentNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentProcessingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/tickets/{ticket_id}", response_model=Ticket)
async def get_ticket(ticket_id: str) -> Ticket:
    # Keep this endpoint read-only. It should return the exact repository record,
    # not ask the model to reconstruct ticket details. Test both the successful
    # lookup and unknown-ID response through FastAPI's test client.
    ticket = pipeline.tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@app.post("/voice/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    request: Request,
    file: UploadFile | None = None,
) -> TranscriptionResponse | JSONResponse:
    audio_bytes: bytes = b""
    media_type = "audio/wav"

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        if file is not None:
            audio_bytes = await file.read()
            if file.content_type:
                media_type = file.content_type
        else:
            form = await request.form()
            for val in form.values():
                if hasattr(val, "read"):
                    audio_bytes = await val.read()
                    if hasattr(val, "content_type") and val.content_type:
                        media_type = val.content_type
                    break
    else:
        audio_bytes = await request.body()
        if content_type:
            media_type = content_type

    if not audio_bytes or len(audio_bytes) < 10:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Audio input is empty", "detail": "Audio input is empty"},
        )

    try:
        transcript, processing_time_ms = await voice_pipeline.transcribe(audio_bytes, media_type)
        return TranscriptionResponse(
            success=True,
            transcript=transcript,
            processing_time_ms=processing_time_ms,
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(exc), "detail": str(exc)},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Transcription failed: {exc}", "detail": str(exc)},
        )


@app.post("/voice/synthesize")
async def synthesize(request: SynthesisRequest) -> Response:
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input is empty")

    try:
        audio_bytes, media_type, processing_time_ms = await voice_pipeline.synthesize(request.text)
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "X-Message-ID": request.message_id,
                "X-Processing-Time-MS": str(processing_time_ms),
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc

