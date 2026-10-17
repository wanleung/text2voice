"""Background synthesis jobs (one GPU worker, FIFO). Finished jobs form the audio library."""
import json
import logging
import queue
import secrets
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from . import audio
from .config import settings
from .engine import TTSEngine
from .text import Segment

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    status: str  # queued | running | done | failed
    voice: str
    language: str
    speed: float
    format: str
    characters: int
    segments_total: int
    segments_done: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    duration_seconds: float | None = None
    error: str | None = None
    callback_url: str | None = None
    metadata: dict = field(default_factory=dict)
    title: str = ""
    source_url: str | None = None
    languages: list[str] = field(default_factory=list)
    share_token: str | None = None

    def public(self) -> dict:
        d = asdict(self)
        d.pop("callback_url")
        d["progress"] = round(self.segments_done / self.segments_total, 3) if self.segments_total else 0
        d["audio_url"] = f"/v1/jobs/{self.id}/audio" if self.status == "done" else None
        d["share_url"] = f"/s/{self.share_token}" if self.share_token else None
        return d


def default_title(segments: list[Segment], limit: int = 60) -> str:
    first = segments[0].text.rstrip("。.!?！？") if segments else ""
    return first if len(first) <= limit else first[:limit].rstrip() + "…"


class JobManager:
    def __init__(self, engine: TTSEngine, root: Path):
        self.engine = engine
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.segments: dict[str, list[Segment]] = {}
        self.queue: queue.Queue[str] = queue.Queue()
        self._load()
        threading.Thread(target=self._worker, name="tts-jobs", daemon=True).start()

    # -- persistence ---------------------------------------------------------------------
    def _dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: Job) -> None:
        path = self._dir(job.id) / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(job), ensure_ascii=False), "utf-8")
        tmp.replace(path)

    def _load(self) -> None:
        for meta in self.root.glob("*/job.json"):
            try:
                job = Job(**json.loads(meta.read_text("utf-8")))
            except Exception:
                log.exception("skipping unreadable job %s", meta)
                continue
            if job.status in ("queued", "running"):
                job.status, job.error = "failed", "service restarted before the job finished"
                job.finished_at = time.time()
                self._save(job)
            self.jobs[job.id] = job

    # -- API -----------------------------------------------------------------------------
    def submit(self, segments: list[Segment], voice: str, language: str, speed: float, fmt: str,
               callback_url: str | None, metadata: dict, title: str = "", source_url: str | None = None,
               text: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex, status="queued", voice=voice, language=language, speed=speed,
                  format=fmt, characters=sum(len(s.text) for s in segments),
                  segments_total=len(segments), callback_url=callback_url, metadata=metadata,
                  title=title.strip() or default_title(segments), source_url=source_url,
                  languages=sorted({s.language for s in segments}))
        self._dir(job.id).mkdir(parents=True)
        (self._dir(job.id) / "text.md").write_text(text, "utf-8")
        self.jobs[job.id] = job
        self.segments[job.id] = segments
        self._save(job)
        self.queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def audio_path(self, job: Job) -> Path:
        return self._dir(job.id) / f"audio.{job.format}"

    def text(self, job: Job) -> str:
        path = self._dir(job.id) / "text.md"
        return path.read_text("utf-8") if path.exists() else ""

    def by_share_token(self, token: str) -> Job | None:
        for job in self.jobs.values():
            if job.share_token and secrets.compare_digest(job.share_token, token):
                return job
        return None

    def update(self, job: Job, title: str | None = None, shared: bool | None = None) -> Job:
        if title is not None:
            job.title = title.strip() or job.title
        if shared is True and not job.share_token:
            job.share_token = secrets.token_urlsafe(16)
        elif shared is False:
            job.share_token = None
        self._save(job)
        return job

    def queue_position(self, job_id: str) -> int | None:
        queued = list(self.queue.queue)
        return queued.index(job_id) + 1 if job_id in queued else None

    def delete(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status == "running":
            return False
        self.jobs.pop(job_id, None)
        self.segments.pop(job_id, None)
        shutil.rmtree(self._dir(job_id), ignore_errors=True)
        return True

    # -- worker --------------------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
            try:
                self._run(job)
            except Exception as e:
                log.exception("job %s failed", job.id)
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            finally:
                job.finished_at = time.time()
                self.segments.pop(job.id, None)
                if job.id in self.jobs:
                    self._save(job)
                    self._notify(job)
            self._cleanup()

    def _run(self, job: Job) -> None:
        job.status, job.started_at = "running", time.time()
        self._save(job)
        voice = self.engine.voices.get(job.voice)

        def progress(done: int, total: int) -> None:
            job.segments_done = done
            self._save(job)

        pcm = self.engine.synthesize(self.segments[job.id], voice, job.speed, progress=progress)
        self.audio_path(job).write_bytes(audio.encode(pcm, self.engine.sample_rate, job.format))
        job.duration_seconds = round(len(pcm) / self.engine.sample_rate, 2)
        job.status = "done"
        log.info("job %s done: %.0fs audio in %.0fs", job.id, job.duration_seconds, time.time() - job.started_at)

    def _notify(self, job: Job) -> None:
        if not job.callback_url:
            return
        try:
            httpx.post(job.callback_url, json=job.public(), timeout=10)
        except Exception as e:
            log.warning("callback for job %s to %s failed: %s", job.id, job.callback_url, e)

    def _cleanup(self) -> None:
        if settings.job_ttl_hours <= 0:  # keep the library forever
            return
        cutoff = time.time() - settings.job_ttl_hours * 3600
        for job in list(self.jobs.values()):
            if job.finished_at and job.finished_at < cutoff:
                self.delete(job.id)
