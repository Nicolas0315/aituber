import asyncio
import os
import time

from fastapi import FastAPI

from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.sts.models import STSRequest
from aiavatar.sts.vad.silero import SileroSpeechDetector
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


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "local")
LLM_MODEL = os.getenv("LLM_MODEL", "local-model")
VOICEVOX_URL = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021")
VOICEVOX_SPEAKER = int(os.getenv("VOICEVOX_SPEAKER", "46"))

STT_MODEL = os.getenv("STT_MODEL", "small")
STT_DEVICE = os.getenv("STT_DEVICE", "cuda")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")

DEBUG = env_bool("DEBUG", "false")


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

# TTS (local Voicevox)
tts = VoicevoxSpeechSynthesizer(
    base_url=VOICEVOX_URL,
    speaker=VOICEVOX_SPEAKER,
    debug=DEBUG,
)

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


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(chat_ingest_worker())
