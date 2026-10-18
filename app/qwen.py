"""Client for the local Qwen3-TTS worker (qwen_worker/): voice design and the optional "qwen" engine.

The worker runs in its own environment because qwen-tts needs a newer transformers than CosyVoice.
"""
import base64
import io
import re
import threading

import httpx
import numpy as np
import soundfile as sf
import torch
import torchaudio

from .engine import ChunkCallback, SegmentSynthesizer, report_chunks
from .text import Segment, split_sentences
from .voices import Voice, VoiceStore

SAMPLE_RATE = 24000
# Qwen3-TTS has no Cantonese language tag. Cloning a Cantonese voice with "Auto" lets the reference
# clip carry the language, which listeners preferred in testing; voice design takes "Chinese" plus a
# description that asks for Cantonese.
CLONE_LANGUAGE = {"yue": "Auto", "zh": "Chinese", "en": "English"}
DESIGN_LANGUAGE = {"yue": "Chinese", "zh": "Chinese", "en": "English"}
MAX_CHUNK_CHARS = 150  # keeps each generation short and stable
_CANTONESE_WORDS = re.compile(r"粤|粵|广东话|廣東話|cantonese", re.I)


class QwenUnavailable(RuntimeError):
    pass


class QwenClient:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self._http = httpx.Client(base_url=self.url, timeout=httpx.Timeout(600, connect=5))

    def health(self) -> dict:
        try:
            r = self._http.get("/health", timeout=3)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            return {"status": "unreachable", "error": f"{type(e).__name__}: {e}"}

    def _post(self, path: str, body: dict) -> tuple[np.ndarray, int]:
        try:
            r = self._http.post(path, json=body)
        except httpx.HTTPError as e:
            raise QwenUnavailable(f"cannot reach the Qwen worker at {self.url} ({type(e).__name__})") from e
        if r.status_code == 503:
            raise QwenUnavailable(r.json().get("detail", "Qwen worker is not ready"))
        if r.status_code != 200:
            raise RuntimeError(f"Qwen worker returned {r.status_code}: {r.text[:300]}")
        audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()
        return audio, SAMPLE_RATE

    def design(self, text: str, language: str, description: str) -> np.ndarray:
        if language == "yue" and not _CANTONESE_WORDS.search(description):
            description = description.rstrip("。. ") + "，用地道的香港粤语（广东话）说话。"
        return self._post("/design", {"text": text, "language": DESIGN_LANGUAGE[language],
                                      "instruct": description})[0]

    def clone(self, text: str, language: str, ref_wav: bytes, ref_text: str) -> np.ndarray:
        return self._post("/clone", {"text": text, "language": CLONE_LANGUAGE[language],
                                     "ref_audio": base64.b64encode(ref_wav).decode(), "ref_text": ref_text})[0]


def chunk_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Group sentences into chunks of at most `limit` characters (a longer sentence stays whole)."""
    chunks: list[str] = []
    for sent in split_sentences(text):
        if chunks and len(chunks[-1]) + len(sent) + 1 <= limit:
            chunks[-1] += ("" if re.search(r"[　-鿿]$", chunks[-1]) else " ") + sent
        else:
            chunks.append(sent)
    return chunks


class QwenEngine(SegmentSynthesizer):
    """Speaks articles with Qwen3-TTS by cloning the request's voice."""

    sample_rate = SAMPLE_RATE

    def __init__(self, client: QwenClient, voices: VoiceStore, lock: threading.Lock):
        self.client = client
        self.voices = voices
        self.lock = lock  # shared with the CosyVoice engine: both run on the same GPU

    def synthesize_segment(self, segment: Segment, voice: Voice, speed: float = 1.0,
                           on_chunk: ChunkCallback | None = None) -> np.ndarray:
        if speed != 1.0:
            raise ValueError("the Qwen engine does not support speed changes")
        ref = self.voices.wav_path(voice.name).read_bytes()
        pieces = []
        with self.lock:
            for chunk in report_chunks(chunk_text(segment.text), on_chunk):
                pieces.append(self.client.clone(chunk, segment.language, ref, voice.transcript))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
