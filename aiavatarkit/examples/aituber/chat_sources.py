import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    platform: str
    user: str
    text: str


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
            await queue.put(ChatMessage(platform="twitch", user=message.author.name, text=message.content))

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
                    msg = ChatMessage(platform="youtube", user=item.author.name, text=item.message)
                    loop.call_soon_threadsafe(queue.put_nowait, msg)
            chat.terminate()

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)


class ChatRouter:
    def __init__(self, min_interval_sec: float = 2.0):
        self.queue: asyncio.Queue[ChatMessage] = asyncio.Queue()
        self.sources: list[ChatSourceBase] = []
        self.min_interval_sec = min_interval_sec
        self._last_sent_at: float = 0.0
        self._tasks: list[asyncio.Task] = []

    def add_source(self, source: ChatSourceBase) -> None:
        self.sources.append(source)

    async def start(self) -> None:
        for source in self.sources:
            self._tasks.append(asyncio.create_task(source.start(self.queue)))

    async def stop(self) -> None:
        for source in self.sources:
            await source.stop()
        for task in self._tasks:
            if not task.done():
                task.cancel()

    async def next_message(self) -> ChatMessage:
        return await self.queue.get()

    def can_send_now(self, now_ts: float) -> bool:
        if self._last_sent_at == 0.0:
            return True
        return (now_ts - self._last_sent_at) >= self.min_interval_sec

    def mark_sent(self, now_ts: float) -> None:
        self._last_sent_at = now_ts
