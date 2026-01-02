# Integration Issues / TODO

## Core stability
- Add process supervisor config (pm2/systemd) and restart policies for backend services.
- Add backpressure and queue length limits for chat ingestion to avoid voice spam.
- Add metrics export (latency, queue depth, dropped frames) and a health endpoint.

## Local model path
- Standardize local LLM server choices (llama.cpp/vLLM/Ollama) and provide presets.
- Add local TTS options beyond VOICEVOX (Style-Bert-VITS2/Coqui) with config toggles.
- Add local STT model size selector and benchmark presets (tiny/small/medium).

## Live2D control
- Map AIAvatarKit face tags to Live2D parameters (eyes/mouth/eyebrows) instead of motion only.
- Add motion map UI for custom Live2D motions per emotion.

## Vision
- Add screen/camera sampling rate control and difference detection for low overhead.
- Add per-game toggle or hotkey integration for vision enable/disable.

## Memory
- Replace the simple SQLite store with vector DB (Chroma/FAISS) and embeddings.
- Add long-term memory summarization pipeline with periodic refresh.

## Streaming chat
- Add Twitch/YouTube message filtering (spam, length, language).
- Add priority rules (mentions > superchat > normal).
