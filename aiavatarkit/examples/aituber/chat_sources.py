import asyncio
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    platform: str
    user: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatFilterConfig:
    max_length: int = 200
    min_length: int = 1
    allowed_langs: set[str] = field(default_factory=set)
    blocklist: set[str] = field(default_factory=set)
    repeat_ratio_threshold: float = 0.6
    repeat_run_threshold: int = 8


@dataclass
class ChatPriorityPolicy:
    mention_keywords: list[str] = field(default_factory=list)
    bot_name: str = ""
    superchat_priority: int = 1
    mention_priority: int = 2


def _normalize_list(values: list[str]) -> list[str]:
    return [value.strip() for value in values if value.strip()]


def _safe_float(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _max_run_length(text: str) -> int:
    if not text:
        return 0
    max_run = 1
    run = 1
    for idx in range(1, len(text)):
        if text[idx] == text[idx - 1]:
            run += 1
        else:
            max_run = max(max_run, run)
            run = 1
    return max(max_run, run)


def _repeat_ratio(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    return max(counts.values()) / max(1, len(text))


def _is_japanese_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana/Katakana
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0xFF66 <= code <= 0xFF9D  # Half-width Katakana
    )


def _is_language_allowed(text: str, allowed_langs: set[str]) -> bool:
    if not allowed_langs or "all" in allowed_langs:
        return True
    has_ja = any(_is_japanese_char(ch) for ch in text)
    has_en = any(ch.isascii() and ch.isalpha() for ch in text)
    if has_ja and "ja" in allowed_langs:
        return True
    if has_en and "en" in allowed_langs:
        return True
    if not (has_ja or has_en) and "other" in allowed_langs:
        return True
    return False


def _contains_blocked(text: str, blocklist: set[str]) -> bool:
    if not blocklist:
        return False
    lowered = text.casefold()
    return any(word in lowered for word in blocklist)


def _is_spam(text: str, repeat_ratio_threshold: float, repeat_run_threshold: int) -> bool:
    compact = "".join(ch for ch in text if not ch.isspace())
    if not compact:
        return True
    if repeat_run_threshold > 0 and _max_run_length(compact) >= repeat_run_threshold:
        return True
    if repeat_ratio_threshold > 0 and _repeat_ratio(compact) >= repeat_ratio_threshold:
        return True
    return False


class ChatFilter:
    def __init__(self, config: ChatFilterConfig):
        self.config = config

    def allows(self, message: ChatMessage) -> bool:
        text = message.text.strip()
        if not text:
            return False
        if self.config.min_length > 0 and len(text) < self.config.min_length:
            return False
        if self.config.max_length > 0 and len(text) > self.config.max_length:
            return False
        if _contains_blocked(text, self.config.blocklist):
            return False
        if not _is_language_allowed(text, self.config.allowed_langs):
            return False
        if _is_spam(text, self.config.repeat_ratio_threshold, self.config.repeat_run_threshold):
            return False
        return True


class ChatPriority:
    def __init__(self, policy: ChatPriorityPolicy):
        self.policy = policy
        self.policy.mention_keywords = _normalize_list(self.policy.mention_keywords)

    def _is_mention(self, text: str) -> bool:
        lowered = text.casefold()
        if self.policy.bot_name:
            handle = f"@{self.policy.bot_name.casefold()}"
            if handle in lowered:
                return True
        for keyword in self.policy.mention_keywords:
            if keyword.casefold() in lowered:
                return True
        return False

    def compute(self, message: ChatMessage) -> int:
        priority = 0
        is_superchat = bool(message.metadata.get("is_superchat"))
        if is_superchat:
            priority = max(priority, self.policy.superchat_priority)
        if self._is_mention(message.text):
            priority = max(priority, self.policy.mention_priority)
        return priority


class ChatSourceBase:
    async def start(self, queue: asyncio.Queue) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        return None


class TwitchChatSource(ChatSourceBase):
    def __init__(self, token: str, channel: str):
        self.token = token
        self.channel = channel
        self._client = None

    async def start(self, queue: asyncio.Queue) -> None:
        try:
            import twitchio
        except ImportError as exc:
            raise ImportError("twitchio is required for TwitchChatSource") from exc

        client = twitchio.Client(token=self.token)
        self._client = client

        @client.event()
        async def event_ready():
            logger.info(f"Twitch connected as {client.nick}")
            await client.join_channels([self.channel])

        @client.event()
        async def event_message(message):
            if message.echo:
                return
            tags = getattr(message, "tags", {}) or {}
            bits = 0
            try:
                bits = int(tags.get("bits", 0))
            except (TypeError, ValueError):
                bits = 0

            await queue.put(
                ChatMessage(
                    platform="twitch",
                    user=message.author.name,
                    text=message.content,
                    metadata={
                        "bits": bits,
                        "is_superchat": bits > 0,
                    },
                )
            )

        await client.connect()

    async def stop(self) -> None:
        if self._client:
            await self._client.close()


class YouTubeChatSource(ChatSourceBase):
    def __init__(self, video_id: str):
        self.video_id = video_id
        self._running = False
        self._thread: Optional[threading.Thread] = None

    async def start(self, queue: asyncio.Queue) -> None:
        try:
            import pytchat
        except ImportError as exc:
            raise ImportError("pytchat is required for YouTubeChatSource") from exc

        loop = asyncio.get_running_loop()
        self._running = True

        def worker():
            chat = pytchat.create(video_id=self.video_id)
            while self._running and chat.is_alive():
                for item in chat.get().sync_items():
                    amount_value = getattr(item, "amountValue", None) or getattr(item, "amount", None)
                    amount_value = _safe_float(amount_value)
                    msg = ChatMessage(
                        platform="youtube",
                        user=item.author.name,
                        text=item.message,
                        metadata={
                            "amount_value": amount_value,
                            "currency": getattr(item, "currency", None),
                            "is_superchat": amount_value > 0,
                            "type": getattr(item, "type", None),
                        },
                    )
                    loop.call_soon_threadsafe(queue.put_nowait, msg)
            chat.terminate()

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)


class ChatRouter:
    def __init__(
        self,
        min_interval_sec: float = 2.0,
        filter_config: Optional[ChatFilterConfig] = None,
        priority_policy: Optional[ChatPriorityPolicy] = None,
    ):
        self.raw_queue: asyncio.Queue[ChatMessage] = asyncio.Queue()
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.sources: list[ChatSourceBase] = []
        self.min_interval_sec = min_interval_sec
        self._last_sent_at: float = 0.0
        self._tasks: list[asyncio.Task] = []
        self._filter_task: Optional[asyncio.Task] = None
        self._counter = 0
        self._filter = ChatFilter(filter_config or ChatFilterConfig())
        self._priority = ChatPriority(priority_policy or ChatPriorityPolicy())

    def add_source(self, source: ChatSourceBase) -> None:
        self.sources.append(source)

    async def start(self) -> None:
        for source in self.sources:
            self._tasks.append(asyncio.create_task(source.start(self.raw_queue)))
        self._filter_task = asyncio.create_task(self._filter_worker())

    async def stop(self) -> None:
        for source in self.sources:
            await source.stop()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._filter_task and not self._filter_task.done():
            self._filter_task.cancel()

    async def next_message(self) -> ChatMessage:
        _, _, _, message = await self.queue.get()
        return message

    async def _filter_worker(self) -> None:
        while True:
            message = await self.raw_queue.get()
            if not self._filter.allows(message):
                continue
            priority = self._priority.compute(message)
            self._counter += 1
            await self.queue.put((-priority, time.time(), self._counter, message))

    def can_send_now(self, now_ts: float) -> bool:
        if self._last_sent_at == 0.0:
            return True
        return (now_ts - self._last_sent_at) >= self.min_interval_sec

    def mark_sent(self, now_ts: float) -> None:
        self._last_sent_at = now_ts
