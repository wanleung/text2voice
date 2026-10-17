# text2voice

HTTP text-to-speech service for reading articles aloud in **English** and **Cantonese** (Mandarin
also works), built on [Fun-CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) (CosyVoice 2 is a
drop-in alternative). Made to be called by an AI agent: plain JSON in, audio out, with background jobs for
long articles.

Why CosyVoice and not Qwen3-TTS: the open Qwen3-TTS models support 10 languages plus Beijing and
Sichuan dialect speakers, but not Cantonese. CosyVoice 2 and 3 both speak Cantonese, and in testing
Fun-CosyVoice3 was the more accurate of the two.

## Setup

Tested on an AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151) with ROCm PyTorch wheels from AMD's
`rocm.nightlies.amd.com/v2/gfx1151` index, which `pyproject.toml` pins.

```bash
git submodule update --init --recursive
uv sync
uv run python scripts/download_models.py          # add --cosyvoice2 to also fetch CosyVoice2-0.5B
uv run uvicorn app.main:app --host 127.0.0.1 --port 8020
```

On NVIDIA hardware, replace the ROCm pins in `pyproject.toml` (torch, torchaudio, triton, rocm*) with
CUDA builds of torch/torchaudio.

Run it permanently with the systemd user unit in `deploy/text2voice.service`. Settings go in `.env`
(see `.env.example`). Interactive API docs are at `http://localhost:8020/docs`, and the
machine-readable schema is at `/openapi.json`.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/tts` | Synchronous TTS for text up to 5,000 chars. Returns audio. |
| `POST` | `/v1/audio/speech` | OpenAI-compatible version of the above. |
| `POST` | `/v1/jobs` | Queue a whole article (up to 200k chars). Returns `202` with a job. |
| `GET` | `/v1/jobs/{id}` | Job status and progress. |
| `GET` | `/v1/jobs/{id}/audio` | Download the finished audio. |
| `DELETE` | `/v1/jobs/{id}` | Delete a finished job. |
| `GET/POST/DELETE` | `/v1/voices` | List, upload or delete reference voices. |
| `GET` | `/health` | Model and device info. |

Common request fields:

- `text`: plain text or Markdown. Headings, lists, links (only the link text is read), bare URLs,
  images, code blocks and HTML tags are cleaned up before speaking.
- `language`: `auto` (the default), `en`, `yue` or `zh`. `auto` decides per sentence: text that is
  mostly CJK is read as Cantonese (`T2V_AUTO_CHINESE_LANGUAGE`), anything else as English. A
  Cantonese article that quotes an English sentence switches language for that sentence.
- `voice`: a registered voice name. Defaults to `default`.
- `speed`: 0.5 to 2.0.
- `format`: `mp3` (the default), `wav`, `flac`, `opus` or `pcm` (16-bit mono, 24 kHz).

### Short text

```bash
curl -s localhost:8020/v1/tts -H 'content-type: application/json' \
  -d '{"text": "今日天氣好好，我哋一齊去飲茶啦。"}' -o out.mp3
```

Each response includes an `X-Audio-Duration` header and an `X-Languages` header listing the
languages that were used.

### Long articles

```bash
curl -s localhost:8020/v1/jobs -H 'content-type: application/json' \
  -d '{"text": "# Title\n\nFirst paragraph...", "format": "mp3",
       "callback_url": "http://my-agent/tts-done", "metadata": {"article_id": 42}}'
# -> {"id": "…", "status": "queued", "progress": 0.0, ...}

curl -s localhost:8020/v1/jobs/<id>            # status: queued | running | done | failed
curl -s localhost:8020/v1/jobs/<id>/audio -o article.mp3
```

If you pass `callback_url`, the service POSTs the final job JSON to it when the job finishes, so
you don't have to poll. Jobs and their audio are deleted after `T2V_JOB_TTL_HOURS` (72 by default).

### OpenAI-compatible

Any OpenAI TTS client works if you point its base URL at `http://host:8020/v1`. Voice names the
service doesn't know, such as `alloy`, fall back to the default voice. This endpoint also accepts
an extra `language` field.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8020/v1", api_key="unused-or-T2V_API_KEY")
client.audio.speech.create(model="cosyvoice", voice="default", input="Hello!").write_to_file("hi.mp3")
```

## MCP server

`mcp_server/text2voice_mcp.py` exposes the service as MCP tools. It's a thin client: the model
stays in the web service, so the script only needs `mcp` and `httpx`. `uv` installs those
automatically from the script's inline metadata. It can run on any machine that can reach the
service.

| Tool | What it does |
|---|---|
| `text_to_speech` | Reads `text` or a local `text_file` aloud, saves the audio file and returns its path, duration and the languages used. Short text is synthesized directly. Longer text runs as a background job that reports MCP progress while you wait. With `wait=false` it returns a `job_id` immediately. |
| `get_tts_job` | Checks a job started with `wait=false`. Once it's done, saves the audio and returns its path. |
| `list_tts_voices` | Lists the available voices. |
| `add_tts_voice` | Registers a new voice from a local 3–30 s recording plus its transcript. |

Register it with Claude Code:

```bash
claude mcp add text2voice --scope user \
  -e T2V_URL=http://127.0.0.1:8020 -e T2V_OUTPUT_DIR=$HOME/text2voice-audio \
  -- uv run --script $HOME/Projects/text2voice/mcp_server/text2voice_mcp.py
```

For other MCP clients (Claude Desktop, Cursor, custom agents), add an entry like this to their
JSON config:

```json
{
  "mcpServers": {
    "text2voice": {
      "command": "uv",
      "args": ["run", "--script", "/home/you/Projects/text2voice/mcp_server/text2voice_mcp.py"],
      "env": {"T2V_URL": "http://127.0.0.1:8020", "T2V_OUTPUT_DIR": "/home/you/text2voice-audio"}
    }
  }
}
```

If your agent connects over HTTP instead of launching a process, run
`uv run --script mcp_server/text2voice_mcp.py --http --port 8021` and point the agent at
`http://host:8021/mcp`.

MCP server settings:

- `T2V_URL`: the service URL.
- `T2V_API_KEY`: the service's API key, if it has one.
- `T2V_OUTPUT_DIR`: where audio files are saved. Defaults to `~/text2voice-audio`.
- `T2V_WAIT_TIMEOUT`: how long to wait for a job before returning its `job_id` instead.
  Defaults to 3600 s.
- `T2V_SYNC_LIMIT`: texts longer than this go through a background job. Defaults to 5000 chars.

Audio files are saved on the machine where the MCP server runs.

## Voices

CosyVoice clones a voice from a 3–30 second reference clip together with its exact transcript.
How each sentence is spoken depends on the voice:

| Target language | Voice recorded in… | Mode used |
|---|---|---|
| Cantonese | any language | instruct mode: "speak Cantonese" plus the voice's timbre |
| English or Mandarin | the same language | zero-shot clone: best quality, keeps the speaker's accent and prosody |
| English or Mandarin | a different language | cross-lingual: the voice's timbre only |

Cantonese always uses instruct mode. In testing, zero-shot cloning from a Cantonese clip drifted
toward Mandarin and garbled words, while instruct mode stayed accurate (checked by Whisper
large-v3 transcription).

The bundled `default` voice is a Mandarin-speaking female sample from CosyVoice. For the most
natural results, register a clean native-speaker clip for each language:

```bash
curl -s localhost:8020/v1/voices -F name=cantonese-f -F language=yue \
  -F transcript='呢段錄音入面講嘅每一隻字。' -F audio=@cantonese.wav
curl -s localhost:8020/v1/voices -F name=english-m -F language=en \
  -F transcript='The exact words spoken in the clip.' -F audio=@english.wav
```

Then pass `"voice": "cantonese-f"` for Cantonese articles and `"voice": "english-m"` for English
ones. Only clone voices you have permission to use.

## Performance

These numbers were measured on a Radeon 8060S with Fun-CosyVoice3 in fp32. The real-time factor
(RTF, compute time ÷ audio length) is about 0.7–0.9, so the service runs a little faster than real
time. A 10-minute article takes roughly 7–9 minutes to synthesize.

- On ROCm the service disables MIOpen (`torch.backends.cudnn.enabled = False`). MIOpen's
  convolution kernels made the vocoder about 4x slower on gfx1151. `T2V_USE_MIOPEN=true` turns it
  back on.
- `T2V_FP16=true` gains roughly 5–10%.
- One GPU worker handles all requests. Synchronous requests and jobs take turns at paragraph
  boundaries.

## Development

```bash
uv run --group dev pytest       # API tests swap the model for a fake engine, so no GPU is needed
```

Layout: `app/text.py` (markup cleanup, sentence splitting, language detection), `app/engine.py`
(CosyVoice wrapper), `app/jobs.py` (background queue), `app/voices.py` (voice registry),
`app/main.py` (FastAPI routes).
