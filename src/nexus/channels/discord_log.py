"""Discord logging handler — pipes Python log records to #logs channel.

Buffers log lines and flushes them periodically or when the buffer is
full so we don't spam Discord with one message per log line.

Usage::

    from nexus.channels.discord_log import DiscordLogHandler

    handler = DiscordLogHandler(channel)
    handler.start()  # begin background flush loop
    logging.getLogger("nexus").addHandler(handler)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Final

import discord

# Max characters per Discord message (leave room for code block markers)
_MAX_MSG: Final[int] = 1900
# Flush interval in seconds
_FLUSH_INTERVAL: Final[float] = 5.0
# Maximum buffered lines before forcing a flush
_MAX_BUFFER: Final[int] = 30


class DiscordLogHandler(logging.Handler):
    """Async-safe logging handler that sends records to a Discord channel.

    Log records are buffered and flushed in batches wrapped in code blocks
    so they're readable in Discord.  Only WARNING+ and selected INFO
    loggers are forwarded to avoid flooding the channel.
    """

    # Loggers whose INFO messages are interesting enough to forward.
    _INFO_LOGGERS: Final[frozenset[str]] = frozenset({
        "nexus.bot",
        "nexus.orchestrator.loop",
        "nexus.integrations.c2_engine",
        "nexus.integrations.pieces",
        "nexus.memory.store",
    })

    def __init__(self, channel: discord.TextChannel) -> None:
        super().__init__(level=logging.INFO)
        self._channel = channel
        self._buffer: list[str] = []
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Start the background flush loop. Call after bot is ready."""
        self._loop = asyncio.get_event_loop()
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Flush remaining buffer and cancel the background task."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._flush()

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer a log record for async delivery to Discord."""
        # Filter: WARNING+ always, INFO only from interesting loggers.
        if record.levelno < logging.WARNING:
            if record.name not in self._INFO_LOGGERS:
                return

        line = self.format(record)
        # Truncate very long lines
        if len(line) > 200:
            line = line[:197] + "..."

        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._buffer.append, line)
        else:
            self._buffer.append(line)

    async def _flush_loop(self) -> None:
        """Periodically flush buffered log lines to Discord."""
        while True:
            await asyncio.sleep(_FLUSH_INTERVAL)
            await self._flush()

    async def _flush(self) -> None:
        """Send buffered lines to the #logs channel."""
        async with self._lock:
            if not self._buffer:
                return

            lines = self._buffer[:]
            self._buffer.clear()

        # Build messages within Discord's character limit
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for line in lines:
            # +1 for newline
            if current_len + len(line) + 1 > _MAX_MSG and current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += len(line) + 1

        if current:
            chunks.append("\n".join(current))

        for chunk in chunks:
            try:
                await self._channel.send(f"```\n{chunk}\n```")
            except Exception:
                pass  # Don't let logging errors crash the bot
