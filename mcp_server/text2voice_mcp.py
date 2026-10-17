# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=2.3,<3", "httpx>=0.28,<1"]
# ///
"""MCP server exposing the text2voice HTTP service as tools.

It is a thin client: the model runs in the text2voice web service, so this script needs only
`mcp` and `httpx` and can run on any machine that can reach the service.

    uv run --script mcp_server/text2voice_mcp.py                    # stdio (Claude Code, Claude Desktop, ...)
    uv run --script mcp_server/text2voice_mcp.py --http --port 8021 # streamable HTTP

Environment:
    T2V_URL               service base URL (default http://127.0.0.1:8020)
    T2V_API_KEY           API key, if the service requires one
    T2V_OUTPUT_DIR        where audio files are saved (default ~/text2voice-audio)
    T2V_WAIT_TIMEOUT      seconds to wait for a long article before returning its job id (default 3600)
    T2V_SYNC_LIMIT        texts longer than this many characters go through a background job (default 5000)
"""
import argparse
import asyncio
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

BASE_URL = os.environ.get("T2V_URL", "http://127.0.0.1:8020").rstrip("/")
API_KEY = os.environ.get("T2V_API_KEY", "")
OUTPUT_DIR = Path(os.environ.get("T2V_OUTPUT_DIR", "~/text2voice-audio")).expanduser()
WAIT_TIMEOUT = float(os.environ.get("T2V_WAIT_TIMEOUT", "3600"))
SYNC_LIMIT = int(os.environ.get("T2V_SYNC_LIMIT", "5000"))  # matches the service's T2V_MAX_SYNC_CHARS
POLL_SECONDS = 3

logging.getLogger("httpx").setLevel(logging.WARNING)

Language = Literal["auto", "en", "yue", "zh"]
Format = Literal["mp3", "wav", "flac", "opus"]

class SpeechResult(TypedDict):
    status: str  # done | queued | running
    path: NotRequired[str]  # absolute path of the saved audio file, when done
    duration_seconds: NotRequired[float]
    languages: NotRequired[list[str]]
    size_bytes: NotRequired[int]
    job_id: NotRequired[str]
    progress: NotRequired[float]
    queue_position: NotRequired[int | None]
    note: NotRequired[str]


class VoiceInfo(TypedDict):
    name: str
    language: str
    transcript: str
    description: str


class VoiceList(TypedDict):
    default: str
    languages: list[str]
    voices: list[VoiceInfo]


mcp = MCPServer(
    "text2voice",
    instructions=(
        "Converts text and articles to speech in English and Cantonese (Mandarin also works). "
        "Use text_to_speech to get an audio file; Markdown is cleaned up automatically and "
        "language='auto' picks English or Cantonese per sentence. Long articles take roughly as "
        "long to synthesize as the audio lasts."
    ),
)


@asynccontextmanager
async def _client(timeout: float = 600) -> AsyncIterator[httpx.AsyncClient]:
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=timeout) as client:
            yield client
    except httpx.TransportError as e:
        raise ToolError(f"cannot reach the text2voice service at {BASE_URL} ({type(e).__name__}: {e}); "
                        "is it running?") from e


def _read(path: str, binary: bool = False) -> str | bytes:
    p = Path(path).expanduser()
    try:
        return p.read_bytes() if binary else p.read_text("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ToolError(f"cannot read {p}: {e}") from e


def _raise_for_status(r: httpx.Response) -> None:
    if r.is_success:
        return
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    raise ToolError(f"text2voice service returned {r.status_code}: {detail}")


def _output_path(output_path: str, text: str, fmt: str) -> Path:
    if output_path:
        path = Path(output_path).expanduser()
        return path if path.suffix else path.with_suffix(f".{fmt}")
    words = re.findall(r"\w+", text)[:6]
    slug = "-".join(words)[:48] or "speech"
    return OUTPUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}.{fmt}"


def _save(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path.resolve())


async def _download_job(client: httpx.AsyncClient, job: dict, path: Path) -> SpeechResult:
    r = await client.get(f"/v1/jobs/{job['id']}/audio")
    _raise_for_status(r)
    return {
        "status": "done",
        "path": _save(path, r.content),
        "duration_seconds": job["duration_seconds"],
        "size_bytes": len(r.content),
        "job_id": job["id"],
    }


@mcp.tool(annotations=ToolAnnotations(title="Text to speech", readOnlyHint=False, openWorldHint=False))
async def text_to_speech(
    ctx: Context,
    text: str = "",
    text_file: str = "",
    language: Language = "auto",
    voice: str = "",
    speed: float = 1.0,
    format: Format = "mp3",
    output_path: str = "",
    wait: bool = True,
) -> SpeechResult:
    """Convert text or an article to speech and save it as an audio file.

    Args:
        text: Text or Markdown to read aloud. Give either this or text_file.
        text_file: Path to a local text/Markdown file to read aloud instead of passing text.
        language: "auto" (per sentence: CJK -> Cantonese, else English), "en", "yue" (Cantonese) or "zh" (Mandarin).
        voice: Voice name from list_tts_voices; empty uses the service default.
        speed: Speaking rate, 0.5 to 2.0.
        format: Audio format of the saved file.
        output_path: Where to save the audio; defaults to a timestamped file in T2V_OUTPUT_DIR.
        wait: For long text, wait for synthesis to finish. If false, returns a job_id to pass to get_tts_job.

    Returns the saved file's path and duration, or a job_id if the job is still running.
    """
    if bool(text) == bool(text_file):
        raise ToolError("provide exactly one of text or text_file")
    if text_file:
        text = _read(text_file)
    if not text.strip():
        raise ToolError("text is empty")
    if not 0.5 <= speed <= 2.0:
        raise ToolError("speed must be between 0.5 and 2.0")
    path = _output_path(output_path, text, format)
    body = {"text": text, "language": language, "speed": speed, "format": format}
    if voice:
        body["voice"] = voice

    async with _client() as client:
        if len(text) <= SYNC_LIMIT and wait:
            r = await client.post("/v1/tts", json=body)
            if r.status_code != 413:  # 413: service has a lower sync limit; fall through to a job
                _raise_for_status(r)
                return {
                    "status": "done",
                    "path": _save(path, r.content),
                    "duration_seconds": float(r.headers.get("x-audio-duration", 0)),
                    "languages": r.headers.get("x-languages", "").split(","),
                    "size_bytes": len(r.content),
                }

        r = await client.post("/v1/jobs", json={**body, "metadata": {"output_path": str(path)}})
        _raise_for_status(r)
        job = r.json()
        if not wait:
            return {"status": job["status"], "job_id": job["id"],
                    "note": "call get_tts_job with this job_id to check progress and fetch the audio"}

        deadline = time.monotonic() + WAIT_TIMEOUT
        while job["status"] in ("queued", "running"):
            if time.monotonic() > deadline:
                return {"status": job["status"], "job_id": job["id"], "progress": job["progress"],
                        "note": f"still running after {WAIT_TIMEOUT:.0f}s; call get_tts_job later"}
            await ctx.report_progress(job["segments_done"], job["segments_total"],
                                      f"{job['status']}: {job['segments_done']}/{job['segments_total']} segments")
            await asyncio.sleep(POLL_SECONDS)
            r = await client.get(f"/v1/jobs/{job['id']}")
            _raise_for_status(r)
            job = r.json()
        if job["status"] == "failed":
            raise ToolError(f"synthesis failed: {job['error']}")
        return await _download_job(client, job, path)


@mcp.tool(annotations=ToolAnnotations(title="Get TTS job", readOnlyHint=False, idempotentHint=True))
async def get_tts_job(job_id: str, output_path: str = "") -> SpeechResult:
    """Check a text_to_speech job started with wait=false; once done, save its audio and return the path.

    Args:
        job_id: The job_id returned by text_to_speech.
        output_path: Where to save the audio; defaults to the path chosen when the job was started.
    """
    async with _client() as client:
        r = await client.get(f"/v1/jobs/{job_id}")
        _raise_for_status(r)
        job = r.json()
        if job["status"] == "failed":
            raise ToolError(f"synthesis failed: {job['error']}")
        if job["status"] != "done":
            return {"status": job["status"], "job_id": job_id, "progress": job["progress"],
                    "queue_position": job.get("queue_position")}
        planned = output_path or job["metadata"].get("output_path") or ""
        return await _download_job(client, job, _output_path(planned, job_id, job["format"]))


@mcp.tool(annotations=ToolAnnotations(title="List TTS voices", readOnlyHint=True))
async def list_tts_voices() -> VoiceList:
    """List the voices available to text_to_speech, with the language each was recorded in."""
    async with _client(timeout=30) as client:
        r = await client.get("/v1/voices")
        _raise_for_status(r)
        return r.json()


@mcp.tool(annotations=ToolAnnotations(title="Add TTS voice", readOnlyHint=False, destructiveHint=False))
async def add_tts_voice(name: str, audio_path: str, transcript: str, language: Literal["en", "yue", "zh"],
                        description: str = "") -> VoiceInfo:
    """Register a new voice by cloning it from a short reference recording.

    Only use recordings of people who have agreed to have their voice cloned.

    Args:
        name: Voice name (letters, digits, '_', '-', '.').
        audio_path: Local WAV/FLAC/OGG/MP3 file with 3-30 seconds of clean speech from one speaker.
        transcript: The exact words spoken in the recording.
        language: Language spoken in the recording.
        description: Optional note, e.g. "calm female newsreader".
    """
    audio = Path(audio_path).expanduser()
    async with _client(timeout=120) as client:
        r = await client.post(
            "/v1/voices",
            data={"name": name, "language": language, "transcript": transcript, "description": description},
            files={"audio": (audio.name, _read(audio_path, binary=True))},
        )
        _raise_for_status(r)
        return r.json()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8021)
    args = ap.parse_args()
    if args.http:
        mcp.run("streamable-http", host=args.host, port=args.port)
    else:
        mcp.run()
