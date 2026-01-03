# AItuber Example (AIAvatarKit + Local Stack)

This example runs AIAvatarKit as a local speech-to-speech backend with:
- Local LLM (OpenAI-compatible server)
- Local STT (faster-whisper)
- Local TTS (VOICEVOX)
- Optional Twitch/YouTube chat ingestion
- Local vector memory with periodic summarization

## Requirements
- Python 3.10+
- Local LLM server (OpenAI-compatible HTTP API)
  - Example: llama.cpp server, vLLM, or any local OpenAI-compatible endpoint
- VOICEVOX running locally
- faster-whisper installed

## Install
```bash
pip install -e .
pip install faster-whisper
pip install twitchio pytchat  # optional, for chat ingestion
pip install chromadb sentence-transformers  # optional, for vector memory
pip install faiss-cpu fastembed  # optional, alternative vector backend/embeddings
```

## Config
Copy `config.example.env` and set values:

```bash
LLM_PRESET=ollama
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=local-model
LLM_API_KEY=local

TTS_PROVIDER=voicevox
TTS_VOICEVOX_URL=http://127.0.0.1:50021
TTS_VOICEVOX_SPEAKER=46

STT_PRESET=small
STT_MODEL=small
STT_DEVICE=cuda
STT_COMPUTE_TYPE=int8
DEBUG=false
```

Presets:
- `LLM_PRESET`: `llama.cpp`, `vllm`, or `ollama` (fills `LLM_BASE_URL` if empty).
- `STT_PRESET`: `tiny`, `small`, or `medium` (overrides model/compute defaults).
- `TTS_PROVIDER`: `voicevox`, `style-bert-vits2`, `coqui`.

Local TTS providers:
- Style-Bert-VITS2: set `STYLE_BERT_VITS2_URL`, optional `STYLE_BERT_VITS2_ENDPOINT`, `STYLE_BERT_VITS2_SPEAKER`.
- Coqui TTS server: set `COQUI_TTS_URL`, optional `COQUI_TTS_ENDPOINT`, `COQUI_SPEAKER_ID`, `COQUI_LANGUAGE_ID`.

Optional chat ingestion:
```bash
TWITCH_ENABLED=true
TWITCH_TOKEN=oauth:xxxx
TWITCH_CHANNEL=your_channel

YOUTUBE_ENABLED=true
YOUTUBE_VIDEO_ID=your_live_video_id

CHAT_SESSION_ID=airi_chat
CHAT_MIN_INTERVAL_SEC=2.0
CHAT_MAX_QUEUE_SIZE=200
CHAT_MAX_LENGTH=200
CHAT_MIN_LENGTH=1
CHAT_ALLOWED_LANGS=ja,en
CHAT_BLOCKLIST=
CHAT_BLOCKLIST_PATH=
CHAT_REPEAT_RATIO_THRESHOLD=0.6
CHAT_REPEAT_RUN_THRESHOLD=8
CHAT_MENTION_KEYWORDS=airi
CHAT_BOT_NAME=airi
```

Chat filters drop spammy messages by length, repeated characters, or blocklist entries.
Mentions (via `CHAT_BOT_NAME` or `CHAT_MENTION_KEYWORDS`) are prioritized over paid chat,
which is prioritized over normal messages.

## Run
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

The WebSocket endpoint is:
```
ws://localhost:8000/ws
```

## Observability
- Health: `http://localhost:8000/healthz`
- Metrics (Prometheus text): `http://localhost:8000/metrics`

## Process supervisor (systemd / pm2)
Example configs are in `ops/`:
- `ops/systemd/aiavatarkit.service`
- `ops/pm2/aiavatarkit.json`

Adjust paths and env files for your deployment.

## Frontend (AIRI Live2D)
1) Run stage-web:
```bash
cd ../../../../airi
pnpm i
pnpm dev
```
2) Open: Settings -> System -> Developer -> "AIAvatarKit Bridge"
3) Connect to `ws://localhost:8000/ws`
4) Enable Mic / Screen / Camera as needed

If you want chat responses to speak through the Live2D avatar, set the same
`CHAT_SESSION_ID` on the backend and `Session ID` in the devtools page.

## Notes
- The memory tool is simple and local; it stores dialog lines and retrieves by text match.
- The Live2D client should connect with the same `CHAT_SESSION_ID` if you want chat responses
  to be spoken by the avatar.

## Memory
The memory store defaults to `MEMORY_BACKEND=auto`:
- Uses Chroma if installed, otherwise FAISS if available, else falls back to SQLite text search.
- Summaries are refreshed in the background (see `MEMORY_SUMMARY_*` env vars).
