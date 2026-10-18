"""Local Qwen3-TTS worker for text2voice.

Runs in its own environment (qwen-tts needs transformers 4.57; CosyVoice needs 4.51) and listens on
localhost only. The main service calls it for voice design and for the optional "qwen" engine.

    cd qwen_worker && uv run uvicorn server:app --host 127.0.0.1 --port 8022

POST /design  {text, language, instruct}            -> WAV: a new voice speaking `text`
POST /clone   {text, language, ref_audio, ref_text} -> WAV: `text` in the voice of ref_audio (base64)
GET  /health
"""
import base64
import hashlib
import io
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

log = logging.getLogger("qwen_worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODELS_DIR = Path(os.environ.get("T2V_QWEN_MODELS_DIR", Path(__file__).resolve().parent.parent / "pretrained_models"))
BASE_DIR = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base"
DESIGN_DIR = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
LANGUAGES = ("Auto", "Chinese", "English")
MAX_TEXT = 1000

state: dict = {"status": "loading", "error": None}
lock = threading.Lock()  # one generation at a time on the GPU
prompt_cache: dict[str, object] = {}


def _load() -> None:
    try:
        from qwen_tts import Qwen3TTSModel
        if torch.version.hip:
            torch.backends.cudnn.enabled = False  # MIOpen convs are slow on gfx1151, as in the main service
        kw = dict(device_map="cuda:0" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16,
                  attn_implementation="sdpa")
        t0 = time.time()
        state["base"] = Qwen3TTSModel.from_pretrained(str(BASE_DIR), **kw)
        state["design"] = Qwen3TTSModel.from_pretrained(str(DESIGN_DIR), **kw)
        state["status"] = "ok"
        log.info("loaded Qwen3-TTS Base + VoiceDesign in %.1fs", time.time() - t0)
    except Exception as e:  # keep serving /health so the main service can report why
        log.exception("failed to load Qwen models")
        state.update(status="error", error=f"{type(e).__name__}: {e}")


threading.Thread(target=_load, name="load-models", daemon=True).start()
app = FastAPI(title="text2voice Qwen worker")


class DesignRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT)
    language: str = "Chinese"
    instruct: str = Field(..., min_length=1, max_length=1000)


class CloneRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT)
    language: str = "Auto"
    ref_audio: str = Field(..., description="base64 WAV of the reference clip")
    ref_text: str = Field(..., min_length=1)


def _model(name: str):
    if state["status"] != "ok":
        raise HTTPException(503, f"Qwen models are {state['status']}" + (f": {state['error']}" if state["error"] else ""))
    return state[name]


def _language(language: str) -> str:
    if language not in LANGUAGES:
        raise HTTPException(422, f"language must be one of {LANGUAGES}")
    return language


def _max_tokens(text: str) -> int:
    """Cap generation by text length (the codec runs at 12 tokens/s of audio).

    Qwen occasionally never emits its end-of-speech token and would otherwise run to 2048 tokens
    (~170 s of babble), which can take many minutes. The cap allows ~3 CJK characters/s, slower
    than anyone speaks.
    """
    cjk = sum(1 for c in text if "　" <= c <= "鿿")
    return min(2048, 96 + 4 * cjk + (len(text) - cjk))


def _wav(audio: np.ndarray, sr: int) -> Response:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


@app.get("/health")
def health():
    return {"status": state["status"], "error": state["error"], "models": [BASE_DIR.name, DESIGN_DIR.name]}


@app.post("/design")
def design(req: DesignRequest):
    model = _model("design")
    with lock, torch.inference_mode():
        wavs, sr = model.generate_voice_design(text=req.text, language=_language(req.language), instruct=req.instruct,
                                               max_new_tokens=_max_tokens(req.text))
    return _wav(wavs[0], sr)


@app.post("/clone")
def clone(req: CloneRequest):
    model = _model("base")
    try:
        ref = base64.b64decode(req.ref_audio, validate=True)
        audio, sr = sf.read(io.BytesIO(ref), dtype="float32")
    except Exception as e:
        raise HTTPException(422, f"ref_audio must be a base64 WAV ({e})") from e
    key = hashlib.sha256(ref + req.ref_text.encode()).hexdigest()
    with lock, torch.inference_mode():
        prompt = prompt_cache.get(key)
        if prompt is None:
            prompt = model.create_voice_clone_prompt(ref_audio=(audio, sr), ref_text=req.ref_text,
                                                     x_vector_only_mode=False)
            prompt_cache[key] = prompt
        wavs, out_sr = model.generate_voice_clone(text=req.text, language=_language(req.language),
                                                  voice_clone_prompt=prompt, max_new_tokens=_max_tokens(req.text))
    return _wav(wavs[0], out_sr)
