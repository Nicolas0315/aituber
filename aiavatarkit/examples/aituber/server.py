import asyncio
import os
import time

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.sts.models import STSRequest
from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.tts import create_instant_synthesizer
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
from aiavatar.sts.llm.chatgpt import ChatGPTService
from aiavatar.sts.stt.openai import OpenAISpeechRecognizer
from aiavatar.sts.stt.faster_whisper import FasterWhisperSpeechRecognizer

from chat_sources import (
    ChatFilterConfig,
    ChatPriorityPolicy,
    ChatRouter,
    TwitchChatSource,
    YouTubeChatSource,
)
from memory import SimpleMemoryStore


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_blocklist(path: str) -> set[str]:
    if not path:
        return set()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return {line.strip().casefold() for line in handle if line.strip() and not line.startswith("#")}
    except OSError:
        return set()


SYSTEM_PROMPT = """
You are a Live2D VTuber. Keep responses short and spoken-friendly.

Face tags:
- Use [face:joy], [face:angry], [face:sorrow], [face:fun], [face:surprised], or [face:neutral]
- Insert tags directly in the response when you want an expression.

Vision:
- If you need a screenshot, respond with [vision:screenshot]
- If you need the camera, respond with [vision:camera]

Memory:
- Use the tool "search_memory" when past information is needed.
"""


LLM_PRESETS = {
    "llama.cpp": "http://127.0.0.1:8080/v1",
    "vllm": "http://127.0.0.1:8000/v1",
    "ollama": "http://127.0.0.1:11434/v1",
}


def join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def resolve_llm_base_url() -> str:
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    preset = os.getenv("LLM_PRESET", "").strip().lower()
    if not base_url and preset:
        base_url = LLM_PRESETS.get(preset, "")
    return base_url


def resolve_stt_settings() -> tuple[str, str, str]:
    model = os.getenv("STT_MODEL", "small").strip()
    device = os.getenv("STT_DEVICE", "").strip() or "cuda"
    compute_type = os.getenv("STT_COMPUTE_TYPE", "").strip()
    preset = os.getenv("STT_PRESET", "").strip().lower()
    if preset in ("tiny", "small", "medium"):
        model = preset
        if not compute_type:
            compute_type = "int8_float16" if preset == "medium" else "int8"
    if not compute_type:
        compute_type = "int8"
    return model, device, compute_type


def create_tts_provider(debug: bool):
    provider = os.getenv("TTS_PROVIDER", "voicevox").strip().lower()
    if provider in ("", "voicevox"):
        base_url = os.getenv("TTS_VOICEVOX_URL", os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021"))
        speaker = int(os.getenv("TTS_VOICEVOX_SPEAKER", os.getenv("VOICEVOX_SPEAKER", "46")))
        return VoicevoxSpeechSynthesizer(
            base_url=base_url,
            speaker=speaker,
            debug=debug,
        )
    if provider in ("style-bert-vits2", "style_bert_vits2"):
        base_url = os.getenv("STYLE_BERT_VITS2_URL", "http://127.0.0.1:5000").strip()
        endpoint = os.getenv("STYLE_BERT_VITS2_ENDPOINT", "/voice")
        speaker = os.getenv("STYLE_BERT_VITS2_SPEAKER", "0").strip()
        language = os.getenv("STYLE_BERT_VITS2_LANGUAGE", "").strip()
        payload = {"text": "{text}", "speaker": speaker}
        if language:
            payload["language"] = language
        return create_instant_synthesizer(
            method="POST",
            url=join_url(base_url, endpoint),
            json=payload,
            debug=debug,
        )
    if provider == "coqui":
        base_url = os.getenv("COQUI_TTS_URL", "http://127.0.0.1:5002").strip()
        endpoint = os.getenv("COQUI_TTS_ENDPOINT", "/api/tts")
        speaker_id = os.getenv("COQUI_SPEAKER_ID", "").strip()
        language_id = os.getenv("COQUI_LANGUAGE_ID", "").strip()
        payload = {"text": "{text}"}
        if speaker_id:
            payload["speaker_id"] = speaker_id
        if language_id:
            payload["language_id"] = language_id
        use_json = env_bool("COQUI_USE_JSON", "false")
        return create_instant_synthesizer(
            method="POST",
            url=join_url(base_url, endpoint),
            json=payload if use_json else None,
            params=None if use_json else payload,
            debug=debug,
        )
    raise RuntimeError(f"Unknown TTS_PROVIDER: {provider}")


DEBUG = env_bool("DEBUG", "false")
LLM_BASE_URL = resolve_llm_base_url()
LLM_API_KEY = os.getenv("LLM_API_KEY", "local")
LLM_MODEL = os.getenv("LLM_MODEL", "local-model")
STT_MODEL, STT_DEVICE, STT_COMPUTE_TYPE = resolve_stt_settings()
APP_START_TIME = time.time()
CHAT_ROUTER: ChatRouter | None = None


if not LLM_BASE_URL:
    raise RuntimeError("LLM_BASE_URL is required for local LLM server")


# VAD
vad = SileroSpeechDetector(
    silence_duration_threshold=0.5,
    use_vad_iterator=True,
)

# STT (local by default)
if env_bool("USE_OPENAI_STT", "false"):
    stt = OpenAISpeechRecognizer(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        language=os.getenv("STT_LANGUAGE", "ja"),
    )
else:
    stt = FasterWhisperSpeechRecognizer(
        model_size=STT_MODEL,
        device=STT_DEVICE,
        compute_type=STT_COMPUTE_TYPE,
        language=os.getenv("STT_LANGUAGE", "ja"),
        debug=DEBUG,
    )

# LLM (local OpenAI-compatible server)
llm = ChatGPTService(
    openai_api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    model=LLM_MODEL,
    system_prompt=os.getenv("SYSTEM_PROMPT", SYSTEM_PROMPT).strip(),
    use_dynamic_tools=True,
    enable_tool_filtering=True,
    debug=DEBUG,
)

# TTS (local)
tts = create_tts_provider(DEBUG)

# AIAvatar server
aiavatar_app = AIAvatarWebSocketServer(
    vad=vad,
    stt=stt,
    llm=llm,
    tts=tts,
    merge_request_threshold=1.5,
    use_invoke_queue=True,
    debug=DEBUG,
)


# Memory tool (simple local store)
memory_store = SimpleMemoryStore(db_path=os.getenv("MEMORY_DB", "memory.db"))
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))

search_memory_spec = {
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": "Search local memory records by query text.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}


@llm.tool(search_memory_spec)
async def search_memory(query: str, metadata: dict = None):
    results = memory_store.search(query, limit=MEMORY_TOP_K)
    return {
        "results": [
            {
                "id": r.id,
                "text": r.text,
                "kind": r.kind,
                "metadata": r.metadata,
                "created_at": r.created_at,
            }
            for r in results
        ]
    }


@aiavatar_app.sts.on_finish
async def store_memory(request, response):
    if request.text:
        memory_store.add(f"user: {request.text}", kind="dialog")
    if response.voice_text:
        memory_store.add(f"assistant: {response.voice_text}", kind="dialog")


async def chat_ingest_worker():
    global CHAT_ROUTER
    blocklist = {word.casefold() for word in parse_csv(os.getenv("CHAT_BLOCKLIST", ""))}
    blocklist |= load_blocklist(os.getenv("CHAT_BLOCKLIST_PATH", "").strip())
    allowed_langs = {lang.casefold() for lang in parse_csv(os.getenv("CHAT_ALLOWED_LANGS", "ja,en"))}

    filter_config = ChatFilterConfig(
        max_length=int(os.getenv("CHAT_MAX_LENGTH", "200")),
        min_length=int(os.getenv("CHAT_MIN_LENGTH", "1")),
        allowed_langs=allowed_langs,
        blocklist=blocklist,
        repeat_ratio_threshold=float(os.getenv("CHAT_REPEAT_RATIO_THRESHOLD", "0.6")),
        repeat_run_threshold=int(os.getenv("CHAT_REPEAT_RUN_THRESHOLD", "8")),
    )

    priority_policy = ChatPriorityPolicy(
        mention_keywords=parse_csv(os.getenv("CHAT_MENTION_KEYWORDS", "")),
        bot_name=os.getenv("CHAT_BOT_NAME", "").strip(),
    )

    router = ChatRouter(
        min_interval_sec=float(os.getenv("CHAT_MIN_INTERVAL_SEC", "2.0")),
        max_queue_size=int(os.getenv("CHAT_MAX_QUEUE_SIZE", "200")),
        filter_config=filter_config,
        priority_policy=priority_policy,
    )
    CHAT_ROUTER = router
    if env_bool("TWITCH_ENABLED", "false"):
        router.add_source(TwitchChatSource(os.getenv("TWITCH_TOKEN", ""), os.getenv("TWITCH_CHANNEL", "")))
    if env_bool("YOUTUBE_ENABLED", "false"):
        router.add_source(YouTubeChatSource(os.getenv("YOUTUBE_VIDEO_ID", "")))

    if not router.sources:
        return

    await router.start()
    session_id = os.getenv("CHAT_SESSION_ID", "airi_chat")
    context_id = os.getenv("CHAT_CONTEXT_ID", "") or None
    user_prefix = os.getenv("CHAT_USER_PREFIX", "")

    while True:
        msg = await router.next_message()
        now_ts = time.time()
        if not router.can_send_now(now_ts):
            continue
        router.mark_sent(now_ts)

        user_id = f"{user_prefix}{msg.platform}:{msg.user}"
        async for response in aiavatar_app.sts.invoke(
            STSRequest(
                type="invoke",
                session_id=session_id,
                user_id=user_id,
                context_id=context_id,
                text=msg.text,
                allow_merge=False,
            )
        ):
            if response.type == "start" and response.context_id:
                context_id = response.context_id
            await aiavatar_app.sts.handle_response(response)


app = FastAPI()
app.include_router(aiavatar_app.get_websocket_router())


def build_metrics_text() -> str:
    metrics = CHAT_ROUTER.get_metrics() if CHAT_ROUTER else {
        "raw_queue_depth": 0,
        "queue_depth": 0,
        "max_queue_size": 0,
        "dropped_total": 0,
        "filtered_total": 0,
        "enqueued_total": 0,
        "processed_total": 0,
        "last_message_at": 0.0,
        "last_sent_at": 0.0,
        "last_drop_at": 0.0,
    }
    uptime = time.time() - APP_START_TIME

    lines = [
        "# HELP aitatuber_uptime_seconds Uptime in seconds.",
        "# TYPE aitatuber_uptime_seconds gauge",
        f"aitatuber_uptime_seconds {uptime:.0f}",
        "# HELP aitatuber_chat_queue_depth Current priority queue depth.",
        "# TYPE aitatuber_chat_queue_depth gauge",
        f"aitatuber_chat_queue_depth {metrics['queue_depth']}",
        "# HELP aitatuber_chat_raw_queue_depth Current raw queue depth.",
        "# TYPE aitatuber_chat_raw_queue_depth gauge",
        f"aitatuber_chat_raw_queue_depth {metrics['raw_queue_depth']}",
        "# HELP aitatuber_chat_queue_max_size Configured queue max size.",
        "# TYPE aitatuber_chat_queue_max_size gauge",
        f"aitatuber_chat_queue_max_size {metrics['max_queue_size']}",
        "# HELP aitatuber_chat_dropped_total Dropped chat messages.",
        "# TYPE aitatuber_chat_dropped_total counter",
        f"aitatuber_chat_dropped_total {metrics['dropped_total']}",
        "# HELP aitatuber_chat_filtered_total Filtered chat messages.",
        "# TYPE aitatuber_chat_filtered_total counter",
        f"aitatuber_chat_filtered_total {metrics['filtered_total']}",
        "# HELP aitatuber_chat_enqueued_total Enqueued chat messages.",
        "# TYPE aitatuber_chat_enqueued_total counter",
        f"aitatuber_chat_enqueued_total {metrics['enqueued_total']}",
        "# HELP aitatuber_chat_processed_total Processed chat messages.",
        "# TYPE aitatuber_chat_processed_total counter",
        f"aitatuber_chat_processed_total {metrics['processed_total']}",
        "# HELP aitatuber_chat_last_message_ts_seconds Last message enqueue timestamp.",
        "# TYPE aitatuber_chat_last_message_ts_seconds gauge",
        f"aitatuber_chat_last_message_ts_seconds {metrics['last_message_at']:.0f}",
        "# HELP aitatuber_chat_last_sent_ts_seconds Last message sent timestamp.",
        "# TYPE aitatuber_chat_last_sent_ts_seconds gauge",
        f"aitatuber_chat_last_sent_ts_seconds {metrics['last_sent_at']:.0f}",
        "# HELP aitatuber_chat_last_drop_ts_seconds Last drop timestamp.",
        "# TYPE aitatuber_chat_last_drop_ts_seconds gauge",
        f"aitatuber_chat_last_drop_ts_seconds {metrics['last_drop_at']:.0f}",
    ]
    return "\n".join(lines) + "\n"


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - APP_START_TIME),
        "chat_router_running": CHAT_ROUTER is not None,
    }


@app.get("/metrics")
def metrics():
    return PlainTextResponse(build_metrics_text())


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(chat_ingest_worker())
