import io

import numpy as np
import soundfile as sf

# format -> (media type, soundfile format, subtype)
FORMATS = {
    "mp3": ("audio/mpeg", "MP3", "MPEG_LAYER_III"),
    "wav": ("audio/wav", "WAV", "PCM_16"),
    "flac": ("audio/flac", "FLAC", "PCM_16"),
    "opus": ("audio/ogg", "OGG", "OPUS"),
    "pcm": ("audio/L16", "RAW", "PCM_16"),
}


def encode(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    _, container, subtype = FORMATS[fmt]
    audio = np.clip(audio, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format=container, subtype=subtype)
    return buf.getvalue()


def media_type(fmt: str, sample_rate: int) -> str:
    mt = FORMATS[fmt][0]
    return f"{mt};rate={sample_rate};channels=1" if fmt == "pcm" else mt
