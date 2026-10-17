"""Download model weights into pretrained_models/.

    uv run python scripts/download_models.py            # Fun-CosyVoice3 + text normaliser
    uv run python scripts/download_models.py --cosyvoice2
"""
import argparse
import time
from pathlib import Path

from modelscope import snapshot_download

ROOT = Path(__file__).resolve().parent.parent / "pretrained_models"
MODEL_FILES = ["llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "CosyVoice-BlankEN/model.safetensors"]
MODELS = {
    "cosyvoice3": ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "Fun-CosyVoice3-0.5B",
                   MODEL_FILES + ["cosyvoice3.yaml", "speech_tokenizer_v3.onnx"]),
    "cosyvoice2": ("iic/CosyVoice2-0.5B", "CosyVoice2-0.5B",
                   MODEL_FILES + ["cosyvoice2.yaml", "speech_tokenizer_v2.onnx"]),
    "wetext": ("pengzhendong/wetext", "wetext",
               [f"{lang}/tn/{f}.fst" for lang in ("en", "zh") for f in ("tagger", "verbalizer")]),
}


def fetch(repo: str, name: str, required: list[str], attempts: int = 6) -> None:
    for i in range(attempts):
        try:
            snapshot_download(repo, local_dir=str(ROOT / name))
        except Exception as e:
            print(f"{name}: {e}")
        # snapshot_download skips files ModelScope rate-limits (HTTP 403) instead of raising.
        missing = [f for f in required if not (ROOT / name / f).exists()]
        if not missing:
            print(f"ok: {name}")
            return
        wait = 30 * (i + 1)
        print(f"{name}: missing {missing}; retrying in {wait}s")
        time.sleep(wait)
    raise SystemExit(f"failed to download {repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosyvoice2", action="store_true", help="also download CosyVoice2-0.5B")
    args = ap.parse_args()
    fetch(*MODELS["wetext"])
    fetch(*MODELS["cosyvoice3"])
    if args.cosyvoice2:
        fetch(*MODELS["cosyvoice2"])
