import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import audio
from .config import settings
from .engine import TTSEngine
from .jobs import JobManager
from .text import LANGUAGES, Segment, prepare
from .voices import VoiceError, VoiceStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

Language = Literal["auto", "en", "yue", "zh"]
Format = Literal["mp3", "wav", "flac", "opus", "pcm"]

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    voices = VoiceStore(settings.voices_dir)
    engine = await run_in_threadpool(TTSEngine, voices)
    state.update(voices=voices, engine=engine, jobs=JobManager(engine, settings.jobs_dir))
    yield


app = FastAPI(
    title="text2voice",
    version="0.1.0",
    description="Text-to-speech for articles in English and Cantonese, powered by CosyVoice.",
    lifespan=lifespan,
)


def require_key(authorization: Annotated[str | None, Header()] = None,
                x_api_key: Annotated[str | None, Header()] = None) -> None:
    if not settings.api_key:
        return
    token = x_api_key or (authorization or "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token.encode(), settings.api_key.encode()):
        raise HTTPException(401, "invalid or missing API key")


auth = [Depends(require_key)]


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Plain text or Markdown. Markup, links and code blocks are stripped.")
    language: Language = Field("auto", description="'auto' detects per paragraph: CJK text -> Cantonese (yue), otherwise English.")
    voice: str = Field(default_factory=lambda: settings.default_voice)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: Format = "mp3"


class JobRequest(TTSRequest):
    callback_url: str | None = Field(None, description="POSTed the job status JSON when the job finishes.")
    metadata: dict = Field(default_factory=dict, description="Opaque data echoed back in job status.")


class OpenAISpeechRequest(BaseModel):
    model: str = "cosyvoice"
    input: str = Field(..., min_length=1)
    voice: str = Field(default_factory=lambda: settings.default_voice)
    response_format: Format = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    language: Language = "auto"  # extension to the OpenAI schema


def _segments(text: str, language: str, limit: int) -> list[Segment]:
    if len(text) > limit:
        raise HTTPException(413, f"text is {len(text)} characters; the limit here is {limit}"
                                 + (" (use POST /v1/jobs for long articles)" if limit == settings.max_sync_chars else ""))
    segments = prepare(text, language, settings.auto_chinese_language)
    if not segments:
        raise HTTPException(422, "no speakable text after removing markup")
    return segments


def _voice(name: str):
    try:
        return state["voices"].get(name)
    except VoiceError as e:
        raise HTTPException(404, str(e)) from e


def _synthesize(text: str, language: str, voice_name: str, speed: float, fmt: str) -> Response:
    segments = _segments(text, language, settings.max_sync_chars)
    voice = _voice(voice_name)
    engine: TTSEngine = state["engine"]
    pcm = engine.synthesize(segments, voice, speed)
    return Response(audio.encode(pcm, engine.sample_rate, fmt), media_type=audio.media_type(fmt, engine.sample_rate),
                    headers={"X-Audio-Duration": f"{len(pcm) / engine.sample_rate:.2f}",
                             "X-Languages": ",".join(sorted({s.language for s in segments}))})


@app.get("/health")
def health():
    engine: TTSEngine | None = state.get("engine")
    if engine is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "model": engine.kind,
        "model_dir": str(settings.model_dir.name),
        "sample_rate": engine.sample_rate,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "queued_jobs": state["jobs"].queue.qsize(),
    }


@app.post("/v1/tts", dependencies=auth, response_class=Response,
          responses={200: {"content": {"audio/mpeg": {}, "audio/wav": {}}}})
def tts(req: TTSRequest):
    """Synthesize short text synchronously and return the audio file."""
    return _synthesize(req.text, req.language, req.voice, req.speed, req.format)


@app.post("/v1/audio/speech", dependencies=auth, response_class=Response)
def openai_speech(req: OpenAISpeechRequest):
    """OpenAI-compatible endpoint. Unknown voice names (e.g. 'alloy') fall back to the default voice."""
    voice = req.voice if (settings.voices_dir / req.voice / "voice.json").exists() else settings.default_voice
    return _synthesize(req.input, req.language, voice, min(max(req.speed, 0.5), 2.0), req.response_format)


# -- jobs --------------------------------------------------------------------------------
@app.post("/v1/jobs", dependencies=auth, status_code=202)
def create_job(req: JobRequest):
    """Queue a long article for synthesis. Poll GET /v1/jobs/{id} or pass callback_url."""
    segments = _segments(req.text, req.language, settings.max_job_chars)
    _voice(req.voice)
    job = state["jobs"].submit(segments, req.voice, req.language, req.speed, req.format,
                               req.callback_url, req.metadata)
    return job.public()


@app.get("/v1/jobs", dependencies=auth)
def list_jobs(limit: int = 50):
    jobs = sorted(state["jobs"].jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]
    return {"jobs": [j.public() for j in jobs]}


@app.get("/v1/jobs/{job_id}", dependencies=auth)
def get_job(job_id: str):
    jobs: JobManager = state["jobs"]
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {**job.public(), "queue_position": jobs.queue_position(job_id)}


@app.get("/v1/jobs/{job_id}/audio", dependencies=auth, response_class=FileResponse)
def get_job_audio(job_id: str):
    jobs: JobManager = state["jobs"]
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != "done":
        raise HTTPException(409, f"job is {job.status}")
    return FileResponse(jobs.audio_path(job), media_type=audio.media_type(job.format, state["engine"].sample_rate),
                        filename=f"{job_id}.{job.format}")


@app.delete("/v1/jobs/{job_id}", dependencies=auth, status_code=204)
def delete_job(job_id: str):
    if not state["jobs"].delete(job_id):
        raise HTTPException(409, "job not found or still running")


# -- voices ------------------------------------------------------------------------------
@app.get("/v1/voices", dependencies=auth)
def list_voices():
    return {"default": settings.default_voice, "languages": LANGUAGES,
            "voices": [v.public() for v in state["voices"].list()]}


@app.post("/v1/voices", dependencies=auth, status_code=201)
async def add_voice(
    name: Annotated[str, Form()],
    language: Annotated[str, Form(description="Language spoken in the clip: en, yue or zh")],
    transcript: Annotated[str, Form(description="Exact words spoken in the clip")],
    audio_file: Annotated[UploadFile, File(alias="audio", description="3-30 s of clean speech")],
    description: Annotated[str, Form()] = "",
    overwrite: Annotated[bool, Form()] = False,
):
    data = await audio_file.read()
    try:
        voice = await run_in_threadpool(state["voices"].add, name, data, transcript, language, description, overwrite)
    except VoiceError as e:
        raise HTTPException(422, str(e)) from e
    return voice.public()


@app.delete("/v1/voices/{name}", dependencies=auth, status_code=204)
def delete_voice(name: str):
    if name == settings.default_voice:
        raise HTTPException(409, "cannot delete the default voice")
    try:
        state["voices"].delete(name)
    except VoiceError as e:
        raise HTTPException(404, str(e)) from e
