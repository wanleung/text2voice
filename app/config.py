from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="T2V_", env_file=ROOT / ".env", extra="ignore")

    # Fun-CosyVoice3 by default; point at pretrained_models/CosyVoice2-0.5B to use CosyVoice 2.
    model_dir: Path = ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B"
    wetext_dir: Path = ROOT / "pretrained_models" / "wetext"
    cosyvoice_dir: Path = ROOT / "third_party" / "CosyVoice"
    voices_dir: Path = ROOT / "voices"
    jobs_dir: Path = ROOT / "data" / "jobs"

    # Optional bearer token. Empty means no auth.
    api_key: str = ""
    fp16: bool = False
    # ROCm only: leave False unless a future MIOpen release is faster than the native kernels.
    use_miopen: bool = False

    default_voice: str = "default"
    # Language used for CJK text when language="auto".
    auto_chinese_language: str = "yue"
    # Synchronous endpoints reject longer input; use /v1/jobs for whole articles.
    max_sync_chars: int = 5000
    max_job_chars: int = 200_000
    paragraph_pause_ms: int = 500
    job_ttl_hours: int = 72


settings = Settings()
