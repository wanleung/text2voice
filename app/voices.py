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

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .text import LANGUAGES

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
PROMPT_SR = 24000
MIN_SECONDS, MAX_SECONDS = 2.0, 30.0


class VoiceError(ValueError):
    pass


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
        try:
            data, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        except Exception as e:
            raise VoiceError(f"cannot decode audio ({e}); upload WAV, FLAC, OGG or MP3") from e
        wav = torch.from_numpy(data.mean(axis=1, keepdims=True).T.copy())
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
