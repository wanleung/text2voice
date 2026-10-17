"""Download model weights into pretrained_models/.

    uv run python scripts/download_models.py            # Fun-CosyVoice3 + text normaliser
    uv run python scripts/download_models.py --cantonese  # + CosyVoice2-Yue-ZoengJyutGaai
    uv run python scripts/download_models.py --cosyvoice2
"""
import argparse
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download as hf_snapshot_download
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


YUE_REPO, YUE_DIR = "ASLP-lab/Cosyvoice2-Yue-ZoengJyutGaai", "CosyVoice2-Yue-ZoengJyutGaai"


def fetch_cantonese() -> None:
    """Download the Cantonese fine-tune from Hugging Face and make it loadable by current CosyVoice."""
    d = ROOT / YUE_DIR
    hf_snapshot_download(YUE_REPO, local_dir=str(d), ignore_patterns=["*.plan", "vllm/*", "flow.encoder.*.zip"])
    # Its hift.pt uses the old weight_norm key names; hift.new.pt is the same vocoder in the current
    # format. Compare contents: a re-download restores the original hift.pt.
    new = d / "hift.new.pt"
    if new.exists() and (d / "hift.pt").read_bytes() != new.read_bytes():
        (d / "hift.pt").replace(d / "hift.old-format.pt")
        (d / "hift.pt").write_bytes(new.read_bytes())
    # llm.pt is a training checkpoint with extra "epoch"/"step" entries that load_state_dict rejects.
    sd = torch.load(d / "llm.pt", map_location="cpu", weights_only=True)
    if "epoch" in sd or "step" in sd:
        (d / "llm.pt").replace(d / "llm.training-ckpt.pt")
        torch.save({k: v for k, v in sd.items() if k not in ("epoch", "step")}, d / "llm.pt")
    print(f"ok: {YUE_DIR} (set T2V_YUE_MODEL_DIR=pretrained_models/{YUE_DIR})")


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
    ap.add_argument("--cantonese", action="store_true", help="also download the CosyVoice2-Yue-ZoengJyutGaai Cantonese model")
    args = ap.parse_args()
    fetch(*MODELS["wetext"])
    fetch(*MODELS["cosyvoice3"])
    if args.cosyvoice2:
        fetch(*MODELS["cosyvoice2"])
    if args.cantonese:
        fetch_cantonese()
