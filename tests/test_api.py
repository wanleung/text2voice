"""API tests with the model replaced by a sine-wave generator (no GPU needed)."""
import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import main
from app.config import settings
from app.voices import VoiceStore


class FakeEngine:
    kind = "Fake"
    sample_rate = 24000

    def __init__(self, voices):
        self.voices = voices
        self.calls = []

    def synthesize(self, segments, voice, speed=1.0, pause_ms=None, progress=None):
        self.calls.append((segments, voice.name, speed))
        for i in range(len(segments)):
            if progress:
                progress(i + 1, len(segments))
        t = np.arange(int(0.5 * self.sample_rate * len(segments))) / self.sample_rate
        return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def wav_bytes(seconds: float, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sr), dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    voices_dir = tmp_path / "voices"
    VoiceStore(voices_dir).add("default", wav_bytes(3), "測試。", "zh")
    monkeypatch.setattr(settings, "voices_dir", voices_dir)
    monkeypatch.setattr(settings, "jobs_dir", tmp_path / "jobs")
    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(main, "TTSEngine", FakeEngine)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_tts_returns_audio_and_languages(client):
    r = client.post("/v1/tts", json={"text": "Hello.\n\n你好嗎？", "format": "wav"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers["x-languages"] == "en,yue"
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 24000 and len(data) == 24000


def test_tts_mp3_default(client):
    r = client.post("/v1/tts", json={"text": "Hello."})
    assert r.headers["content-type"] == "audio/mpeg" and len(r.content) > 100


def test_tts_rejects_long_text_and_unknown_voice(client):
    assert client.post("/v1/tts", json={"text": "a" * (settings.max_sync_chars + 1)}).status_code == 413
    assert client.post("/v1/tts", json={"text": "Hi.", "voice": "nobody"}).status_code == 404
    assert client.post("/v1/tts", json={"text": "```\ncode\n```"}).status_code == 422


def test_openai_endpoint_maps_unknown_voice_to_default(client):
    r = client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Hi.", "voice": "alloy"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"


def test_job_lifecycle(client):
    r = client.post("/v1/jobs", json={"text": "One.\n\n二。\n\nThree.", "format": "wav", "metadata": {"id": 7}})
    assert r.status_code == 202
    job_id = r.json()["id"]
    for _ in range(100):
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] == "done":
            break
        time.sleep(0.05)
    assert job["status"] == "done" and job["progress"] == 1 and job["metadata"] == {"id": 7}
    audio = client.get(job["audio_url"])
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert client.delete(f"/v1/jobs/{job_id}").status_code == 204
    assert client.get(f"/v1/jobs/{job_id}").status_code == 404


def test_voice_upload_validation_and_delete(client):
    files = {"audio": ("v.wav", wav_bytes(5), "audio/wav")}
    r = client.post("/v1/voices", data={"name": "anna", "language": "en", "transcript": "Hi there."}, files=files)
    assert r.status_code == 201
    assert any(v["name"] == "anna" for v in client.get("/v1/voices").json()["voices"])
    short = {"audio": ("v.wav", wav_bytes(1), "audio/wav")}
    assert client.post("/v1/voices", data={"name": "b", "language": "en", "transcript": "x"}, files=short).status_code == 422
    assert client.post("/v1/voices", data={"name": "anna", "language": "en", "transcript": "x"}, files=files).status_code == 422
    assert client.delete("/v1/voices/anna").status_code == 204
    assert client.delete("/v1/voices/default").status_code == 409


def test_api_key(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "s3cret")
    assert client.post("/v1/tts", json={"text": "Hi."}).status_code == 401
    assert client.post("/v1/tts", json={"text": "Hi."}, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.post("/v1/tts", json={"text": "Hi."}, headers={"X-API-Key": "s3cret"}).status_code == 200


def wait_done(client, job_id):
    for _ in range(100):
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_library_title_text_rename_and_search(client):
    job = client.post("/v1/jobs", json={"text": "# 標題\n\nBody text.", "source_url": "https://example.com/a"}).json()
    assert job["title"] == "標題" and job["languages"] == ["en", "yue"]
    wait_done(client, job["id"])
    assert client.get(f"/v1/jobs/{job['id']}/text").json()["text"] == "# 標題\n\nBody text."
    assert client.patch(f"/v1/jobs/{job['id']}", json={"title": "Renamed"}).json()["title"] == "Renamed"
    assert [j["id"] for j in client.get("/v1/jobs?q=renam").json()["jobs"]] == [job["id"]]
    assert [j["id"] for j in client.get("/v1/jobs?q=example.com").json()["jobs"]] == [job["id"]]
    assert client.get("/v1/jobs?q=nomatch").json()["jobs"] == []
    r = client.get(f"/v1/jobs/{job['id']}/audio?download=true")
    assert r.headers["content-disposition"].startswith("attachment") and "Renamed.mp3" in r.headers["content-disposition"]


def test_job_request_validation(client):
    assert client.post("/v1/jobs", json={"text": "a", "url": "https://x.y"}).status_code == 422
    assert client.post("/v1/jobs", json={}).status_code == 422
    assert client.post("/v1/jobs", json={"text": "a", "source_url": "javascript:alert(1)"}).status_code == 422


def test_job_from_url(client, monkeypatch):
    from app.fetch import Article

    async def fake_fetch(url):
        return Article(title="Fetched title", text="# Fetched title\n\n第一段。", url=url)

    monkeypatch.setattr(main, "fetch_article", fake_fetch)
    preview = client.post("/v1/fetch", json={"url": "https://news.example/x"}).json()
    assert preview["title"] == "Fetched title"
    job = client.post("/v1/jobs", json={"url": "https://news.example/x"}).json()
    assert job["title"] == "Fetched title" and job["source_url"] == "https://news.example/x"


def test_fetch_refuses_private_addresses(client):
    for url in ("http://127.0.0.1:8020/health", "http://localhost/", "file:///etc/passwd"):
        r = client.post("/v1/fetch", json={"url": url})
        assert r.status_code == 422, url


def test_share_link(client, monkeypatch):
    job = client.post("/v1/jobs", json={"text": "Share me.", "title": "Share <me> & you"}).json()
    wait_done(client, job["id"])
    shared = client.patch(f"/v1/jobs/{job['id']}", json={"shared": True}).json()
    link = shared["share_url"]
    monkeypatch.setattr(settings, "api_key", "s3cret")  # share pages work without the key
    page = client.get(link)
    assert page.status_code == 200 and "Share &lt;me&gt; &amp; you" in page.text and "<me>" not in page.text
    assert client.get(link + "/audio").status_code == 200
    assert client.get(f"/v1/jobs/{job['id']}/audio").status_code == 401
    monkeypatch.setattr(settings, "api_key", "")
    client.patch(f"/v1/jobs/{job['id']}", json={"shared": False})
    assert client.get(link).status_code == 404


def test_portal_login_cookie(client, monkeypatch):
    assert "text2voice" in client.get("/").text
    monkeypatch.setattr(settings, "api_key", "s3cret")
    assert client.get("/v1/session").json() == {"auth_required": True, "authenticated": False}
    assert client.post("/v1/login", json={"api_key": "wrong"}).status_code == 401
    assert client.post("/v1/login", json={"api_key": "s3cret"}).status_code == 200
    assert client.get("/v1/session").json()["authenticated"] is True
    assert client.get("/v1/jobs").status_code == 200  # cookie authenticates
    client.post("/v1/logout")
    assert client.get("/v1/jobs").status_code == 401


def test_voice_preview_audio(client):
    r = client.get("/v1/voices/default/audio")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
