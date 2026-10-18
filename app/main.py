import logging
import re
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import Annotated, Literal

import torch
from fastapi import Cookie, Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from . import audio
from .config import settings
from .engine import TTSEngine
from .fetch import FetchError, fetch_article
from .jobs import JobManager
from .qwen import QwenClient, QwenEngine, QwenUnavailable
from .text import LANGUAGES, Segment, prepare
from .voices import NAME_RE, VoiceError, VoiceStore

STATIC = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

Language = Literal["auto", "en", "yue", "zh"]
Format = Literal["mp3", "wav", "flac", "opus", "pcm"]
Engine = Literal["cosyvoice", "qwen"]

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    voices = VoiceStore(settings.voices_dir)
    engine = await run_in_threadpool(TTSEngine, voices)
    engines = {"cosyvoice": engine}
    qwen = QwenClient(settings.qwen_url) if settings.qwen_url else None
    if qwen:
        engines["qwen"] = QwenEngine(qwen, voices, engine.lock)
    state.update(voices=voices, engine=engine, engines=engines, qwen=qwen, drafts={}, drafts_lock=threading.Lock(),
                 jobs=JobManager(engines, settings.jobs_dir))
    yield


app = FastAPI(
    title="text2voice",
    version="0.1.0",
    description="Text-to-speech for articles in English and Cantonese, powered by CosyVoice.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


SESSION_COOKIE = "t2v_session"


def _valid_key(token: str | None) -> bool:
    return bool(token) and secrets.compare_digest(token.encode(), settings.api_key.encode())


def require_key(authorization: Annotated[str | None, Header()] = None,
                x_api_key: Annotated[str | None, Header()] = None,
                t2v_session: Annotated[str | None, Cookie()] = None) -> None:
    """API clients send the key as a header; the web portal holds it in an HttpOnly cookie."""
    if not settings.api_key:
        return
    token = x_api_key or (authorization or "").removeprefix("Bearer ").strip() or t2v_session
    if not _valid_key(token):
        raise HTTPException(401, "invalid or missing API key")


auth = [Depends(require_key)]


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Plain text or Markdown. Markup, links and code blocks are stripped.")
    language: Language = Field("auto", description="'auto' detects per paragraph: CJK text -> Cantonese (yue), otherwise English.")
    voice: str = Field(default_factory=lambda: settings.default_voice)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: Format = "mp3"
    engine: Engine = Field("cosyvoice", description="'qwen' speaks with Qwen3-TTS via the local worker (speed must be 1.0).")


class JobRequest(TTSRequest):
    text: str = Field("", description="Plain text or Markdown. Give either this or url.")
    url: str | None = Field(None, description="Web article to fetch and read aloud instead of text.")
    title: str = Field("", description="Library title; defaults to the page title or the first line.")
    source_url: str | None = Field(None, description="Where the text came from, shown in the library.")
    callback_url: str | None = Field(None, description="POSTed the job status JSON when the job finishes.")
    metadata: dict = Field(default_factory=dict, description="Opaque data echoed back in job status.")

    @model_validator(mode="after")
    def _text_or_url(self):
        if bool(self.text.strip()) == bool(self.url):
            raise ValueError("provide exactly one of text or url")
        if self.source_url and not re.match(r"https?://", self.source_url, re.I):
            raise ValueError("source_url must be an http(s) URL")
        return self


class FetchRequest(BaseModel):
    url: str


class JobUpdate(BaseModel):
    title: str | None = None
    shared: bool | None = Field(None, description="true creates a public share link, false revokes it")


class OpenAISpeechRequest(BaseModel):
    model: str = Field("cosyvoice", description="'qwen' (or any name starting with it) selects the Qwen engine.")
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


def _engine(name: str, speed: float):
    engine = state["engines"].get(name)
    if engine is None:
        raise HTTPException(422, f"the {name} engine is not configured (set T2V_QWEN_URL and run qwen_worker/)")
    if name == "qwen" and speed != 1.0:
        raise HTTPException(422, "the qwen engine does not support speed changes; use speed 1.0")
    return engine


def _synthesize(text: str, language: str, voice_name: str, speed: float, fmt: str, engine_name: str) -> Response:
    engine = _engine(engine_name, speed)
    segments = _segments(text, language, settings.max_sync_chars)
    voice = _voice(voice_name)
    try:
        pcm = engine.synthesize(segments, voice, speed)
    except QwenUnavailable as e:
        raise HTTPException(503, str(e)) from e
    return Response(audio.encode(pcm, engine.sample_rate, fmt), media_type=audio.media_type(fmt, engine.sample_rate),
                    headers={"X-Audio-Duration": f"{len(pcm) / engine.sample_rate:.2f}",
                             "X-Languages": ",".join(sorted({s.language for s in segments})),
                             "X-Engine": engine_name})


@app.get("/health")
def health():
    engine: TTSEngine | None = state.get("engine")
    if engine is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "model": engine.kind,
        "model_dir": engine.main.name,
        "cantonese_model": engine.yue.name if engine.yue else None,
        "sample_rate": engine.sample_rate,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "queued_jobs": state["jobs"].queue.qsize(),
        "qwen": state["qwen"].health() if state.get("qwen") else {"status": "not configured"},
    }


@app.post("/v1/tts", dependencies=auth, response_class=Response,
          responses={200: {"content": {"audio/mpeg": {}, "audio/wav": {}}}})
def tts(req: TTSRequest):
    """Synthesize short text synchronously and return the audio file."""
    return _synthesize(req.text, req.language, req.voice, req.speed, req.format, req.engine)


@app.post("/v1/audio/speech", dependencies=auth, response_class=Response)
def openai_speech(req: OpenAISpeechRequest):
    """OpenAI-compatible endpoint. Unknown voice names (e.g. 'alloy') fall back to the default voice."""
    voice = req.voice if (settings.voices_dir / req.voice / "voice.json").exists() else settings.default_voice
    engine = "qwen" if req.model.lower().startswith("qwen") else "cosyvoice"
    speed = 1.0 if engine == "qwen" else min(max(req.speed, 0.5), 2.0)
    return _synthesize(req.input, req.language, voice, speed, req.response_format, engine)


# -- jobs --------------------------------------------------------------------------------
async def _fetch(url: str):
    try:
        return await fetch_article(url)
    except FetchError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/v1/fetch", dependencies=auth)
async def fetch_url(req: FetchRequest):
    """Fetch a web page and return the article text that would be read aloud."""
    article = await _fetch(req.url)
    return {"title": article.title, "url": article.url, "text": article.text, "characters": len(article.text)}


@app.post("/v1/jobs", dependencies=auth, status_code=202)
async def create_job(req: JobRequest):
    """Queue an article (text or URL) for synthesis. Poll GET /v1/jobs/{id} or pass callback_url."""
    _engine(req.engine, req.speed)
    text, title, source_url = req.text, req.title, req.source_url
    if req.url:
        article = await _fetch(req.url)
        text, title, source_url = article.text, title or article.title, article.url
    segments = _segments(text, req.language, settings.max_job_chars)
    _voice(req.voice)
    job = state["jobs"].submit(segments, req.voice, req.language, req.speed, req.format,
                               req.callback_url, req.metadata, title=title, source_url=source_url, text=text,
                               engine=req.engine)
    return job.public()


@app.get("/v1/jobs", dependencies=auth)
def list_jobs(limit: int = 200, q: str = "", engine: Engine | None = None):
    """List jobs (the audio library), newest first. q filters by title or source URL; engine by engine."""
    jobs = sorted(state["jobs"].jobs.values(), key=lambda j: j.created_at, reverse=True)
    if engine:
        jobs = [j for j in jobs if j.engine == engine]
    if q:
        needle = q.casefold()
        jobs = [j for j in jobs if needle in j.title.casefold() or needle in (j.source_url or "").casefold()]
    return {"jobs": [j.public() for j in jobs[:limit]]}


@app.get("/v1/jobs/{job_id}", dependencies=auth)
def get_job(job_id: str):
    jobs: JobManager = state["jobs"]
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {**job.public(), "queue_position": jobs.queue_position(job_id)}


def _job(job_id: str):
    job = state["jobs"].get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


def _audio_file(job, download: bool) -> FileResponse:
    if job.status != "done":
        raise HTTPException(409, f"job is {job.status}")
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", job.title).strip("_")[:80] or job.id
    return FileResponse(state["jobs"].audio_path(job),
                        media_type=audio.media_type(job.format, state["engine"].sample_rate),
                        filename=f"{name}.{job.format}",
                        content_disposition_type="attachment" if download else "inline")


@app.get("/v1/jobs/{job_id}/audio", dependencies=auth, response_class=FileResponse)
def get_job_audio(job_id: str, download: bool = False):
    return _audio_file(_job(job_id), download)


@app.get("/v1/jobs/{job_id}/text", dependencies=auth)
def get_job_text(job_id: str):
    """The text that was read aloud (after URL extraction, before Markdown cleanup)."""
    return {"text": state["jobs"].text(_job(job_id))}


@app.patch("/v1/jobs/{job_id}", dependencies=auth)
def update_job(job_id: str, req: JobUpdate):
    """Rename a library item, or create/revoke its public share link."""
    return state["jobs"].update(_job(job_id), title=req.title, shared=req.shared).public()


@app.delete("/v1/jobs/{job_id}", dependencies=auth, status_code=204)
def delete_job(job_id: str):
    if not state["jobs"].delete(job_id):
        raise HTTPException(409, "job not found or still running")


# -- portal and public share links --------------------------------------------------------
class LoginRequest(BaseModel):
    api_key: str


@app.get("/v1/session", include_in_schema=False)
def session(t2v_session: Annotated[str | None, Cookie()] = None):
    return {"auth_required": bool(settings.api_key),
            "authenticated": not settings.api_key or _valid_key(t2v_session)}


@app.post("/v1/login", include_in_schema=False)
def login(req: LoginRequest, request: Request, response: Response):
    if settings.api_key and not _valid_key(req.api_key):
        raise HTTPException(401, "wrong API key")
    response.set_cookie(SESSION_COOKIE, req.api_key, max_age=90 * 86400, httponly=True, samesite="strict",
                        secure=request.url.scheme == "https")
    return {"ok": True}


@app.post("/v1/logout", include_in_schema=False)
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def portal():
    return HTMLResponse((STATIC / "index.html").read_text("utf-8"))


def _shared(token: str):
    job = state["jobs"].by_share_token(token)
    if job is None or job.status != "done":
        raise HTTPException(404, "link not found or revoked")
    return job


@app.get("/s/{token}", response_class=HTMLResponse, include_in_schema=False)
def share_page(token: str):
    job = _shared(token)
    html = (STATIC / "share.html").read_text("utf-8")
    return HTMLResponse(html.replace("{{title}}", escape(job.title)).replace("{{token}}", escape(token))
                        .replace("{{duration}}", f"{job.duration_seconds or 0:.0f}")
                        .replace("{{source}}", escape(job.source_url or "")))


@app.get("/s/{token}/audio", include_in_schema=False)
def share_audio(token: str, download: bool = False):
    return _audio_file(_shared(token), download)


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


@app.get("/v1/voices/{name}/audio", dependencies=auth, response_class=FileResponse)
def voice_audio(name: str):
    """The voice's reference recording."""
    _voice(name)
    return FileResponse(state["voices"].wav_path(name), media_type="audio/wav")


@app.delete("/v1/voices/{name}", dependencies=auth, status_code=204)
def delete_voice(name: str):
    if name == settings.default_voice:
        raise HTTPException(409, "cannot delete the default voice")
    try:
        state["voices"].delete(name)
    except VoiceError as e:
        raise HTTPException(404, str(e)) from e


# -- voice design (Qwen3-TTS VoiceDesign via the worker) ----------------------------------
DESIGN_SAMPLES = {
    "yue": "大家好，歡迎收聽今日嘅新聞，我哋而家為你整理最新嘅消息。",
    "en": "Hello and welcome to today's news. Here are the stories you need to know.",
    "zh": "大家好，欢迎收听今天的新闻，我们现在为你整理最新的消息。",
}
DRAFT_TTL, MAX_DRAFTS = 3600, 20


class DesignRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=500,
                             description="Who is speaking and how, e.g. 'Hong Kong woman in her 50s, warm storyteller'.")
    language: Literal["yue", "en", "zh"] = "yue"
    sample_text: str = Field("", max_length=200, description="What the preview says; becomes the voice's transcript.")


class SaveDesignRequest(BaseModel):
    name: str
    overwrite: bool = False


def _qwen() -> QwenClient:
    if not state.get("qwen"):
        raise HTTPException(422, "voice design needs the Qwen worker (set T2V_QWEN_URL and run qwen_worker/)")
    return state["qwen"]


def _draft(draft_id: str) -> dict:
    with state["drafts_lock"]:
        now = time.time()
        for k in [k for k, d in state["drafts"].items() if now - d["created"] > DRAFT_TTL]:
            del state["drafts"][k]
        draft = state["drafts"].get(draft_id)
    if draft is None:
        raise HTTPException(404, "design draft not found or expired; generate a new preview")
    return draft


@app.post("/v1/voices/design", dependencies=auth, status_code=201)
def design_voice(req: DesignRequest):
    """Generate a preview of a new voice from a description. Save it with POST .../{draft_id}/save.

    Each call samples a new voice; save the draft you liked rather than regenerating it.
    """
    client = _qwen()
    text = req.sample_text.strip() or DESIGN_SAMPLES[req.language]
    try:
        with state["engine"].lock:  # same GPU as the CosyVoice engine
            pcm = client.design(text, req.language, req.description)
    except QwenUnavailable as e:
        raise HTTPException(503, str(e)) from e
    draft_id = uuid.uuid4().hex
    with state["drafts_lock"]:
        while len(state["drafts"]) >= MAX_DRAFTS:
            del state["drafts"][min(state["drafts"], key=lambda k: state["drafts"][k]["created"])]
        state["drafts"][draft_id] = {"wav": audio.encode(pcm, 24000, "wav"), "text": text, "language": req.language,
                                     "description": req.description, "created": time.time()}
    return {"draft_id": draft_id, "audio_url": f"/v1/voices/design/{draft_id}/audio", "sample_text": text,
            "language": req.language, "description": req.description, "duration_seconds": round(len(pcm) / 24000, 2)}


@app.get("/v1/voices/design/{draft_id}/audio", dependencies=auth, response_class=Response)
def design_voice_audio(draft_id: str):
    return Response(_draft(draft_id)["wav"], media_type="audio/wav")


@app.post("/v1/voices/design/{draft_id}/save", dependencies=auth, status_code=201)
def save_designed_voice(draft_id: str, req: SaveDesignRequest):
    """Register a previewed design as a voice, exactly as it sounded in the preview."""
    draft = _draft(draft_id)
    if not NAME_RE.match(req.name):
        raise HTTPException(422, "voice name must be 1-64 chars of letters, digits, '_', '-', '.'")
    description = ("Synthetic: designed with Qwen3-TTS-12Hz-1.7B-VoiceDesign (Apache-2.0), not a recording "
                   "of a real person. Design prompt: " + draft["description"])
    try:
        voice = state["voices"].add(req.name, draft["wav"], draft["text"], draft["language"], description,
                                    overwrite=req.overwrite)
    except VoiceError as e:
        raise HTTPException(422, str(e)) from e
    with state["drafts_lock"]:
        state["drafts"].pop(draft_id, None)
    return voice.public()
