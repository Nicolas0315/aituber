# AItuber Example (AIAvatarKit + Local Stack)

This example runs AIAvatarKit as a local speech-to-speech backend with:
- Local LLM (OpenAI-compatible server)
- Local STT (faster-whisper)
- Local TTS (VOICEVOX)
- Optional Twitch/YouTube chat ingestion
- Simple local memory tool

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
```

## Config
Copy `config.example.env` and set values:

```bash
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=local-model
LLM_API_KEY=local
VOICEVOX_URL=http://127.0.0.1:50021
VOICEVOX_SPEAKER=46
STT_MODEL=small
STT_DEVICE=cuda
STT_COMPUTE_TYPE=int8
DEBUG=false
```

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
