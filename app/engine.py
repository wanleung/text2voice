"""CosyVoice 2 / Fun-CosyVoice 3 wrapper.

Language handling (per segment):
  * target is Cantonese                           -> instruct mode ("speak Cantonese") + voice timbre
  * the voice was recorded in the target language -> zero-shot cloning with its transcript
  * otherwise                                     -> cross-lingual mode (voice timbre only)
"""
import logging
import sys
import threading
import time
from collections.abc import Callable, Iterator

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


def _load_wav(wav, target_sr, min_sr=16000):
    """Replacement for cosyvoice.utils.file_utils.load_wav; torchaudio>=2.9 dropped its I/O backends."""
    data, sr = sf.read(wav, dtype="float32", always_2d=True)
    speech = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    if sr != target_sr:
        assert sr >= min_sr, f"wav sample rate {sr} must be at least {min_sr}"
        speech = torchaudio.functional.resample(speech, sr, target_sr)
    return speech


class TTSEngine:
    def __init__(self, voices: VoiceStore):
        import cosyvoice.cli.frontend as cv_frontend
        import cosyvoice.utils.file_utils as cv_file_utils
        cv_file_utils.load_wav = cv_frontend.load_wav = _load_wav
        if (settings.wetext_dir / "zh" / "tn" / "tagger.fst").exists():
            # Avoid a ModelScope round-trip (and its rate limits) on every start.
            import wetext.wetext
            wetext.wetext.snapshot_download = lambda *a, **k: str(settings.wetext_dir)
        from cosyvoice.cli.cosyvoice import AutoModel

        if torch.version.hip and not settings.use_miopen:
            # MIOpen's conv kernels make the vocoder ~4x slower on gfx1151 than PyTorch's fallbacks.
            torch.backends.cudnn.enabled = False

        t0 = time.time()
        self.cv =AutoModel(model_dir=str(settings.model_dir), fp16=settings.fp16)
        self.kind = type(self.cv).__name__  # "CosyVoice2" or "CosyVoice3"
        if self.kind not in INSTRUCT:
            raise RuntimeError(f"unsupported model type {self.kind}; use CosyVoice2 or Fun-CosyVoice3")
        self.sample_rate = self.cv.sample_rate
        self.voices = voices
        self._prompt_cache: dict[str, tuple[float, dict]] = {}
        # The model is not safe for concurrent use; requests interleave at paragraph granularity.
        self.lock = threading.Lock()
        log.info("loaded %s from %s in %.1fs on %s", self.kind, settings.model_dir, time.time() - t0,
                 torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    # -- prompt features -------------------------------------------------------------
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
        feats = fe.frontend_zero_shot("", prompt_text, str(wav), self.sample_rate, "")
        del feats["text"], feats["text_len"]
        self._prompt_cache[voice.name] = (mtime, feats)
        return feats

    def _tokens(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cv.frontend._extract_text_token(text)

    def _model_input(self, chunk: str, language: str, voice: Voice) -> dict:
        inp = dict(self._voice_prompt(voice))
        # Cantonese always goes through instruct mode: zero-shot cloning from a Cantonese clip
        # drifted towards Mandarin and garbled words in testing, while instruct mode stayed accurate.
        if language in INSTRUCT[self.kind]:
            mode = "instruct"
            inp["prompt_text"], inp["prompt_text_len"] = self._tokens(INSTRUCT[self.kind][language])
            del inp["llm_prompt_speech_token"], inp["llm_prompt_speech_token_len"]
        elif voice.language == language:
            mode = "zero_shot"
        else:
            mode = "cross_lingual"
            if self.kind == "CosyVoice3":
                chunk = CV3_SYSTEM + chunk
            for k in ("prompt_text", "prompt_text_len", "llm_prompt_speech_token", "llm_prompt_speech_token_len"):
                del inp[k]
        inp["text"], inp["text_len"] = self._tokens(chunk)
        log.debug("%s [%s/%s] %s", mode, language, voice.name, chunk)
        return inp

    # -- synthesis -----------------------------------------------------------------------
    def chunks(self, segment: Segment) -> list[str]:
        """Normalise a paragraph and split it into model-sized sentences."""
        return self.cv.frontend.text_normalize(segment.text, split=True)

    def synthesize_segment(self, segment: Segment, voice: Voice, speed: float = 1.0) -> np.ndarray:
        pieces = []
        with self.lock, torch.inference_mode():
            for chunk in self.chunks(segment):
                inp = self._model_input(chunk, segment.language, voice)
                for out in self.cv.model.tts(**inp, stream=False, speed=speed):
                    pieces.append(out["tts_speech"].squeeze(0).float().cpu().numpy())
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
