"""API tests with the model replaced by a sine-wave generator (no GPU needed)."""
import io
import threading
import time
from types import SimpleNamespace

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
    main = SimpleNamespace(name="fake-model")
    yue = None
    lock = threading.Lock()

    def __init__(self, voices):
        self.voices = voices
        self.calls = []

    def synthesize(self, segments, voice, speed=1.0, pause_ms=None, progress=None):
        self.calls.append((segments, voice.name, speed))
        for i in range(len(segments)):
            if progress:
                progress(i + 1, len(segments), (i + 1) / len(segments))
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
    monkeypatch.setattr(settings, "default_voice", "default")  # independent of the local .env
    monkeypatch.setattr(settings, "qwen_url", "")
    monkeypatch.setattr(main, "TTSEngine", FakeEngine)
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def settings_voices(client):
    return settings.voices_dir  # the temporary folder the client fixture patched in


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


def encoded_bytes(container: str, codec: str, seconds: float = 4.0, sr: int = 48000) -> bytes:
    """A tone encoded with PyAV, like a phone (AAC/ALAC in M4A) or browser (Opus in WebM) recording."""
    import av
    buf = io.BytesIO()
    with av.open(buf, "w", format=container) as out:
        stream = out.add_stream(codec, rate=sr, layout="mono")
        t = np.arange(int(seconds * sr)) / sr
        pcm = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        fmt = "fltp" if codec in ("aac", "libopus") else "s16p"
        samples = pcm if fmt == "fltp" else (pcm * 32767).astype(np.int16)
        for i in range(0, len(samples), 1024):
            frame = av.AudioFrame.from_ndarray(samples[None, i:i + 1024], format=fmt, layout="mono")
            frame.sample_rate = sr
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buf.getvalue()


@pytest.mark.parametrize("filename, container, codec", [
    ("memo.m4a", "mp4", "aac"),        # iPhone Voice Memos / Android recorders
    ("memo-lossless.m4a", "mp4", "alac"),
    ("browser.webm", "webm", "libopus"),  # MediaRecorder in browsers
])
def test_voice_upload_compressed_formats(client, settings_voices, filename, container, codec):
    data = encoded_bytes(container, codec)
    r = client.post("/v1/voices", data={"name": f"v-{codec}", "language": "yue", "transcript": "測試。"},
                    files={"audio": (filename, data, "audio/mp4")})
    assert r.status_code == 201, r.text
    stored, sr = sf.read(settings_voices / f"v-{codec}" / "prompt.wav")
    assert sr == 24000 and abs(len(stored) / sr - 4.0) < 0.1  # decoded fully, resampled to 24 kHz


def test_voice_upload_rejects_garbage_and_huge_files(client, monkeypatch):
    form = {"name": "bad", "language": "yue", "transcript": "測試。"}
    r = client.post("/v1/voices", data=form, files={"audio": ("x.m4a", b"not audio at all" * 100, "audio/mp4")})
    assert r.status_code == 422 and "M4A" in r.json()["detail"]
    from app import voices
    monkeypatch.setattr(voices, "MAX_UPLOAD_BYTES", 1000)
    r = client.post("/v1/voices", data=form, files={"audio": ("big.wav", wav_bytes(3), "audio/wav")})
    assert r.status_code == 422 and "larger than" in r.json()["detail"]



class FakeQwenClient:
    """Stands in for the Qwen worker; records calls."""

    def __init__(self, url):
        self.calls = []

    def health(self):
        return {"status": "ok"}

    def _tone(self, seconds=3.0):
        t = np.arange(int(seconds * 24000)) / 24000
        return (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    def design(self, text, language, description):
        self.calls.append(("design", text, language, description))
        return self._tone()

    def clone(self, text, language, ref_wav, ref_text):
        self.calls.append(("clone", text, language, ref_text))
        return self._tone(0.5)


@pytest.fixture
def qwen_client(tmp_path, monkeypatch):
    voices_dir = tmp_path / "voices"
    VoiceStore(voices_dir).add("default", wav_bytes(3), "測試。", "yue")
    monkeypatch.setattr(settings, "voices_dir", voices_dir)
    monkeypatch.setattr(settings, "jobs_dir", tmp_path / "jobs")
    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(settings, "default_voice", "default")
    monkeypatch.setattr(settings, "qwen_url", "http://qwen.test")
    monkeypatch.setattr(main, "TTSEngine", FakeEngine)
    monkeypatch.setattr(main, "QwenClient", FakeQwenClient)
    with TestClient(main.app) as c:
        yield c


def test_qwen_engine_needs_the_worker(client):
    r = client.post("/v1/tts", json={"text": "Hi.", "engine": "qwen"})
    assert r.status_code == 422 and "T2V_QWEN_URL" in r.json()["detail"]
    assert client.post("/v1/voices/design", json={"description": "a voice"}).status_code == 422
    assert client.get("/health").json()["qwen"] == {"status": "not configured"}


def test_qwen_engine_speaks_and_maps_languages(qwen_client):
    r = qwen_client.post("/v1/tts", json={"text": "今日天氣好好。\n\nHello there.", "engine": "qwen", "format": "wav"})
    assert r.status_code == 200 and r.headers["x-engine"] == "qwen"
    calls = main.state["qwen"].calls
    assert [c[2] for c in calls] == ["yue", "en"]  # per-segment language; the client maps it for Qwen
    assert calls[0][3] == "測試。"  # the voice's transcript is the clone prompt text
    assert qwen_client.post("/v1/tts", json={"text": "Hi.", "engine": "qwen", "speed": 1.2}).status_code == 422
    assert qwen_client.get("/health").json()["qwen"] == {"status": "ok"}


def test_qwen_job_and_openai_model(qwen_client):
    job = qwen_client.post("/v1/jobs", json={"text": "One.\n\nTwo.", "engine": "qwen"}).json()
    assert job["engine"] == "qwen" and wait_done(qwen_client, job["id"])["status"] == "done"
    r = qwen_client.post("/v1/audio/speech", json={"model": "qwen3-tts", "input": "Hi.", "speed": 1.5})
    assert r.status_code == 200 and r.headers["x-engine"] == "qwen"  # speed is ignored for qwen


def test_voice_design_preview_then_save(qwen_client):
    d = qwen_client.post("/v1/voices/design", json={"description": "Hong Kong woman, warm storyteller"}).json()
    assert d["language"] == "yue" and "嘅" in d["sample_text"] and d["duration_seconds"] == 3.0
    preview = qwen_client.get(d["audio_url"])
    assert preview.status_code == 200 and preview.headers["content-type"] == "audio/wav"
    assert qwen_client.post(f"/v1/voices/design/{d['draft_id']}/save", json={"name": "bad name!"}).status_code == 422
    v = qwen_client.post(f"/v1/voices/design/{d['draft_id']}/save", json={"name": "storyteller"}).json()
    assert v["name"] == "storyteller" and v["language"] == "yue" and "Qwen3-TTS" in v["description"]
    assert v["transcript"] == d["sample_text"]
    stored, sr = sf.read(settings.voices_dir / "storyteller" / "prompt.wav")
    assert abs(len(stored) / sr - 3.0) < 0.05  # the saved voice is the previewed clip, not a new sample
    assert qwen_client.post(f"/v1/voices/design/{d['draft_id']}/save", json={"name": "x"}).status_code == 404


def test_design_asks_for_cantonese_when_missing():
    from app.qwen import QwenClient
    c = QwenClient("http://unused")
    sent = []
    c._post = lambda path, body: (sent.append(body), (np.zeros(10, dtype=np.float32), 24000))[1]
    c.design("你好。", "yue", "Hong Kong man, calm")
    c.design("你好。", "yue", "香港男性，讲粤语")
    c.design("Hi.", "en", "British man")
    c.clone("你好。", "yue", b"RIFF", "參考。")
    c.clone("Hi.", "en", b"RIFF", "Ref.")
    assert "粤语" in sent[0]["instruct"] and sent[0]["language"] == "Chinese"
    assert sent[1]["instruct"] == "香港男性，讲粤语"  # already asks for Cantonese: unchanged
    assert sent[2] == {"text": "Hi.", "language": "English", "instruct": "British man"}
    # Qwen has no Cantonese tag: cloning a Cantonese voice uses Auto so the reference carries the language.
    assert (sent[3]["language"], sent[4]["language"]) == ("Auto", "English")


def test_qwen_chunking_keeps_sentences_whole():
    from app.qwen import chunk_text
    chunks = chunk_text("第一句。" * 30 + "Second sentence here. " * 3, limit=40)
    assert all(len(c) <= 40 for c in chunks[:-1]) and "".join(chunks).replace(" ", "").startswith("第一句。第一句。")


def test_library_filters_by_engine(qwen_client):
    cv = qwen_client.post("/v1/jobs", json={"text": "One."}).json()
    qw = qwen_client.post("/v1/jobs", json={"text": "Two.", "engine": "qwen"}).json()
    ids = lambda q: [j["id"] for j in qwen_client.get(f"/v1/jobs{q}").json()["jobs"]]
    assert ids("?engine=qwen") == [qw["id"]] and ids("?engine=cosyvoice") == [cv["id"]]
    assert set(ids("")) == {cv["id"], qw["id"]}
    assert qwen_client.get("/v1/jobs?engine=other").status_code == 422
