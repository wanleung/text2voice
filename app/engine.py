"""CosyVoice 2 / Fun-CosyVoice 3 wrapper, with optional routing to a Cantonese fine-tune.

Main model (Fun-CosyVoice3 by default), per segment:
  * target is Cantonese                           -> instruct mode ("speak Cantonese") + voice timbre
  * the voice was recorded in the target language -> zero-shot cloning with its transcript
  * otherwise                                     -> cross-lingual mode (voice timbre only)

Cantonese model (T2V_YUE_MODEL_DIR, e.g. CosyVoice2-Yue-ZoengJyutGaai), when configured, takes every
Cantonese sentence that has no Latin letters. In a blind listening test it sounded the most natural
on pure Cantonese, but it garbles English words ("Microsoft" -> 北福), so those stay on the main
model. It clones a Cantonese voice zero-shot, or uses instruct mode for voices in other languages.
"""
import logging
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .config import settings
from .text import Segment
from .voices import Voice, VoiceStore

log = logging.getLogger(__name__)

sys.path.insert(0, str(settings.cosyvoice_dir))
sys.path.insert(0, str(settings.cosyvoice_dir / "third_party" / "Matcha-TTS"))

CV3_SYSTEM = "You are a helpful assistant.<|endofprompt|>"
INSTRUCT = {
    "CosyVoice3": {"yue": "You are a helpful assistant. 请用广东话表达。<|endofprompt|>"},
    "CosyVoice2": {"yue": "用粤语说这句话<|endofprompt|>"},
}
# The Cantonese fine-tunes were trained with the instruction exactly as their model card shows it.
YUE_MODEL_INSTRUCT = "用粤语说这句话"
_LATIN = re.compile(r"[A-Za-z]")


def _load_wav(wav, target_sr, min_sr=16000):
    """Replacement for cosyvoice.utils.file_utils.load_wav; torchaudio>=2.9 dropped its I/O backends."""
    data, sr = sf.read(wav, dtype="float32", always_2d=True)
    speech = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    if sr != target_sr:
        assert sr >= min_sr, f"wav sample rate {sr} must be at least {min_sr}"
        speech = torchaudio.functional.resample(speech, sr, target_sr)
    return speech


def _patch_cosyvoice() -> None:
    import cosyvoice.cli.frontend as cv_frontend
    import cosyvoice.utils.file_utils as cv_file_utils
    cv_file_utils.load_wav = cv_frontend.load_wav = _load_wav
    if (settings.wetext_dir / "zh" / "tn" / "tagger.fst").exists():
        # Avoid a ModelScope round-trip (and its rate limits) on every start.
        import wetext.wetext
        wetext.wetext.snapshot_download = lambda *a, **k: str(settings.wetext_dir)
    if torch.version.hip and not settings.use_miopen:
        # MIOpen's conv kernels make the vocoder ~4x slower on gfx1151 than PyTorch's fallbacks.
        torch.backends.cudnn.enabled = False


def use_yue_model(chunk: str, language: str) -> bool:
    """Route a sentence to the Cantonese model: Cantonese text without English words."""
    return language == "yue" and not _LATIN.search(chunk)


class _Model:
    """One loaded CosyVoice model plus its cache of per-voice speaker features."""

    def __init__(self, model_dir: Path, voices: VoiceStore, cantonese_only: bool = False):
        from cosyvoice.cli.cosyvoice import AutoModel
        t0 = time.time()
        self.cv = AutoModel(model_dir=str(model_dir), fp16=settings.fp16)
        self.kind = type(self.cv).__name__  # "CosyVoice2" or "CosyVoice3"
        if self.kind not in INSTRUCT:
            raise RuntimeError(f"unsupported model type {self.kind}; use CosyVoice2 or Fun-CosyVoice3")
        self.name = model_dir.name
        self.cantonese_only = cantonese_only
        self.voices = voices
        self._prompt_cache: dict[str, tuple[float, dict]] = {}
        log.info("loaded %s (%s) in %.1fs", self.name, self.kind, time.time() - t0)

    def _voice_prompt(self, voice: Voice) -> dict:
        """Speaker features of a voice, cached until its prompt.wav changes."""
        wav = self.voices.wav_path(voice.name)
        mtime = wav.stat().st_mtime
        cached = self._prompt_cache.get(voice.name)
        if cached and cached[0] == mtime:
            return cached[1]
        fe = self.cv.frontend
        prompt_text = fe.text_normalize(voice.transcript, split=False)
        if self.kind == "CosyVoice3":
            prompt_text = CV3_SYSTEM + prompt_text
        feats = fe.frontend_zero_shot("", prompt_text, str(wav), self.cv.sample_rate, "")
        del feats["text"], feats["text_len"]
        self._prompt_cache[voice.name] = (mtime, feats)
        return feats

    def _tokens(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cv.frontend._extract_text_token(text)

    def _instruct(self, inp: dict, instruction: str) -> None:
        inp["prompt_text"], inp["prompt_text_len"] = self._tokens(instruction)
        del inp["llm_prompt_speech_token"], inp["llm_prompt_speech_token_len"]

    def model_input(self, chunk: str, language: str, voice: Voice) -> tuple[str, dict]:
        inp = dict(self._voice_prompt(voice))
        if self.cantonese_only:
            # Zero-shot cloning works well here (unlike on the main model) when the voice is Cantonese.
            mode = "zero_shot" if voice.language == "yue" else "instruct"
            if mode == "instruct":
                self._instruct(inp, YUE_MODEL_INSTRUCT)
        # Cantonese always goes through instruct mode on the main model: zero-shot cloning from a
        # Cantonese clip drifted towards Mandarin and garbled words in testing (41-49% CER).
        elif language in INSTRUCT[self.kind]:
            mode = "instruct"
            self._instruct(inp, INSTRUCT[self.kind][language])
        elif voice.language == language:
            mode = "zero_shot"
        else:
            mode = "cross_lingual"
            if self.kind == "CosyVoice3":
                chunk = CV3_SYSTEM + chunk
            for k in ("prompt_text", "prompt_text_len", "llm_prompt_speech_token", "llm_prompt_speech_token_len"):
                del inp[k]
        inp["text"], inp["text_len"] = self._tokens(chunk)
        return mode, inp

    def tts(self, chunk: str, language: str, voice: Voice, speed: float) -> list[np.ndarray]:
        mode, inp = self.model_input(chunk, language, voice)
        log.debug("%s %s [%s/%s] %s", self.name, mode, language, voice.name, chunk)
        return [out["tts_speech"].squeeze(0).float().cpu().numpy()
                for out in self.cv.model.tts(**inp, stream=False, speed=speed)]


class TTSEngine:
    def __init__(self, voices: VoiceStore):
        _patch_cosyvoice()
        self.voices = voices
        self.main = _Model(settings.model_dir, voices)
        self.yue = _Model(settings.yue_model_dir, voices, cantonese_only=True) if settings.yue_model_dir else None
        if self.yue and self.yue.cv.sample_rate != self.main.cv.sample_rate:
            raise RuntimeError("the Cantonese model and the main model must share a sample rate")
        self.kind = self.main.kind
        self.sample_rate = self.main.cv.sample_rate
        # Models are not safe for concurrent use; requests interleave at segment granularity.
        self.lock = threading.Lock()
        log.info("TTS ready on %s (main: %s, Cantonese: %s)",
                 torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                 self.main.name, self.yue.name if self.yue else "main model")

    def chunks(self, segment: Segment) -> list[str]:
        """Normalise a segment and split it into model-sized sentences."""
        return self.main.cv.frontend.text_normalize(segment.text, split=True)

    def model_for(self, chunk: str, language: str) -> _Model:
        return self.yue if self.yue and use_yue_model(chunk, language) else self.main

    def synthesize_segment(self, segment: Segment, voice: Voice, speed: float = 1.0) -> np.ndarray:
        pieces = []
        with self.lock, torch.inference_mode():
            for chunk in self.chunks(segment):
                pieces += self.model_for(chunk, segment.language).tts(chunk, segment.language, voice, speed)
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def synthesize(self, segments: list[Segment], voice: Voice, speed: float = 1.0,
                   pause_ms: int | None = None,
                   progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        audio = list(self.iter_synthesize(segments, voice, speed, pause_ms, progress))
        return np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)

    def iter_synthesize(self, segments: list[Segment], voice: Voice, speed: float = 1.0,
                        pause_ms: int | None = None,
                        progress: Callable[[int, int], None] | None = None) -> Iterator[np.ndarray]:
        pause_ms = settings.paragraph_pause_ms if pause_ms is None else pause_ms
        pause = np.zeros(int(self.sample_rate * pause_ms / 1000), dtype=np.float32)
        gap = np.zeros(int(self.sample_rate * 0.15), dtype=np.float32)  # language switch mid-paragraph
        for i, seg in enumerate(segments):
            if i:
                yield pause if segments[i - 1].paragraph_end else gap
            yield self.synthesize_segment(seg, voice, speed)
            if progress:
                progress(i + 1, len(segments))
