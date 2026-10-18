"""Reference-voice registry.

Each voice lives in ``voices/<name>/`` as ``prompt.wav`` (3-30 s of clean speech) plus
``voice.json`` with the clip's transcript and the language spoken in it.
"""
import io
import json
import re
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np
import soundfile as sf
import torch
import torchaudio

from .text import LANGUAGES

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
PROMPT_SR = 24000
MIN_SECONDS, MAX_SECONDS = 2.0, 30.0
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # a 30 s clip is far smaller even as uncompressed WAV
FORMATS_HINT = "upload WAV, FLAC, OGG, MP3, M4A/AAC (e.g. iPhone Voice Memos) or WebM"


class VoiceError(ValueError):
    pass


def decode_audio(audio: bytes) -> tuple[np.ndarray, int]:
    """Decode an uploaded clip to mono float32 samples and its sample rate.

    soundfile (libsndfile) reads WAV, FLAC, OGG and MP3. Everything else, notably M4A/AAC from
    phone voice recorders and WebM/Opus from browsers, goes through PyAV's bundled FFmpeg.
    """
    try:
        data, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        return data.mean(axis=1), sr
    except Exception:
        pass
    try:
        with av.open(io.BytesIO(audio)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise VoiceError(f"the file has no audio track; {FORMATS_HINT}")
            resampler = av.AudioResampler(format="flt", layout="mono")  # keeps the source rate
            chunks, sr = [], 0
            for frame in container.decode(stream):
                sr = frame.sample_rate
                chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    except (av.FFmpegError, ValueError) as e:
        raise VoiceError(f"cannot decode audio ({e}); {FORMATS_HINT}") from e
    if not chunks or not sr:
        raise VoiceError(f"the file contains no audio samples; {FORMATS_HINT}")
    return np.concatenate(chunks).astype(np.float32), sr


@dataclass
class Voice:
    name: str
    language: str
    transcript: str
    description: str = ""

    def public(self) -> dict:
        return asdict(self)


class VoiceStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def wav_path(self, name: str) -> Path:
        return self.root / name / "prompt.wav"

    def get(self, name: str) -> Voice:
        meta = self.root / name / "voice.json"
        if not NAME_RE.match(name) or not meta.exists():
            raise VoiceError(f"unknown voice '{name}'")
        return Voice(**json.loads(meta.read_text("utf-8")))

    def list(self) -> list[Voice]:
        voices = []
        for meta in sorted(self.root.glob("*/voice.json")):
            voices.append(Voice(**json.loads(meta.read_text("utf-8"))))
        return voices

    def add(self, name: str, audio: bytes, transcript: str, language: str, description: str = "",
            overwrite: bool = False) -> Voice:
        if not NAME_RE.match(name):
            raise VoiceError("voice name must be 1-64 chars of letters, digits, '_', '-', '.'")
        if language not in LANGUAGES:
            raise VoiceError(f"language must be one of {LANGUAGES}")
        transcript = transcript.strip()
        if not transcript:
            raise VoiceError("transcript of the reference audio is required")
        if len(audio) > MAX_UPLOAD_BYTES:
            raise VoiceError(f"audio file is larger than {MAX_UPLOAD_BYTES // 2**20} MB; a 3-30 s clip is enough")
        samples, sr = decode_audio(audio)
        wav = torch.from_numpy(samples).unsqueeze(0)
        if sr < 16000:
            raise VoiceError("reference audio sample rate must be at least 16 kHz")
        if sr != PROMPT_SR:
            wav = torchaudio.functional.resample(wav, sr, PROMPT_SR)
        seconds = wav.shape[1] / PROMPT_SR
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise VoiceError(f"reference audio must be {MIN_SECONDS:.0f}-{MAX_SECONDS:.0f} s long, got {seconds:.1f} s")
        peak = wav.abs().max().item()
        if peak > 0.9:
            wav = wav * (0.9 / peak)

        voice = Voice(name=name, language=language, transcript=transcript, description=description)
        with self._lock:
            target = self.root / name
            if target.exists() and not overwrite:
                raise VoiceError(f"voice '{name}' already exists")
            target.mkdir(parents=True, exist_ok=True)
            sf.write(target / "prompt.wav", wav.squeeze(0).numpy().astype(np.float32), PROMPT_SR)
            (target / "voice.json").write_text(json.dumps(voice.public(), ensure_ascii=False, indent=2), "utf-8")
        return voice

    def delete(self, name: str) -> None:
        self.get(name)
        with self._lock:
            shutil.rmtree(self.root / name)
