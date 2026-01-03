## AIAvatarKit + AIRI Integration (Local Stack)

This workspace uses:
- AIAvatarKit as the speech-to-speech backend (local STT/TTS/LLM)
- AIRI as the Live2D frontend (WebSocket client)

### Backend (AIAvatarKit)
1) Start a local OpenAI-compatible LLM server.
2) Start VOICEVOX.
3) From `aiavatarkit/examples/aituber`:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Set env vars (see `aiavatarkit/examples/aituber/config.example.env`):
- `LLM_BASE_URL`
- `LLM_MODEL`
- `TTS_PROVIDER`
- `TTS_VOICEVOX_URL`
- `STT_PRESET` or `STT_MODEL`

Optional presets:
- `LLM_PRESET` (llama.cpp/vllm/ollama)

Optional:
- Twitch/YouTube chat ingestion (set `TWITCH_*` / `YOUTUBE_*`)
- Vector memory (see `MEMORY_*` env vars)

### Frontend (AIRI)
1) Install and run stage-web:
```bash
cd airi
pnpm i
pnpm dev
```
2) Open Settings -> System -> Developer -> "AIAvatarKit Bridge"
3) Connect to `ws://localhost:8000/ws`
4) Enable Mic / Screen / Camera as needed

### Notes
- If you want chat messages to be spoken by the avatar, use the same `CHAT_SESSION_ID`
  in the backend and `Session ID` in the devtools page.
- Screen capture is only sent when the backend replies with `[vision:screenshot]`
  and the Screen toggle is on.
