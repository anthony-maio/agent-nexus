"""Background email monitor for Agent Nexus.

Periodically polls an IMAP mailbox and ingests new emails into
Continuity Core.  New emails are also announced in ``#nexus`` so the
swarm is aware of incoming communications.

Usage::

    from nexus.integrations.email_monitor import EmailMonitor

    monitor = EmailMonitor(bot)
    await monitor.start()
    # ... later ...
    await monitor.stop()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from nexus.integrations.email_reader import EmailReader

log = logging.getLogger(__name__)


class EmailMonitor:
    """Background task that polls IMAP for new emails.

    Emails are written to C2 as events with intent ``email_received``
    and optionally announced in ``#nexus``.

    Args:
        bot: The ``NexusBot`` instance.
    """

    def __init__(self, bot: Any) -> None:
        self.bot = bot
        settings = bot.settings
        self.reader = EmailReader(
            host=settings.EMAIL_IMAP_HOST,
            port=settings.EMAIL_IMAP_PORT,
            address=settings.EMAIL_ADDRESS,
            password=settings.EMAIL_PASSWORD,
            folder=settings.EMAIL_FOLDER,
        )
        self._interval: int = settings.EMAIL_POLL_INTERVAL
        self._max_messages: int = settings.EMAIL_MAX_MESSAGES
        self._task: asyncio.Task | None = None
        self._running = False
        self._emails_ingested: int = 0

    @property
    def is_configured(self) -> bool:
        return self.reader.is_configured

    @property
    def emails_ingested(self) -> int:
        return self._emails_ingested

    async def start(self) -> None:
        """Start the background email polling loop."""
        if not self.is_configured:
            log.info("Email monitor not configured — skipping.")
            return
        if self._running:
            return

        # Test connection first
        ok = await self.reader.check_connection()
        if not ok:
            log.warning("Email IMAP connection test failed — monitor disabled.")
            return

        self._running = True
        self._task = asyncio.create_task(self._loop())
        log.info(
            "Email monitor started (interval=%ds, folder=%s)",
            self._interval,
            self.reader.folder,
        )

    async def stop(self) -> None:
        """Stop the email polling loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("Email monitor stopped (%d emails ingested)", self._emails_ingested)

    async def _loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._poll()
            except Exception:
                log.warning("Email poll cycle failed.", exc_info=True)
            await asyncio.sleep(self._interval)

    async def _poll(self) -> None:
        """Fetch unread emails and ingest them."""
        emails = await self.reader.fetch_unread(limit=self._max_messages)
        if not emails:
            return

        log.info("Email monitor: %d new email(s) found.", len(emails))

        for msg in emails:
            await self._ingest_email(msg)

    async def _ingest_email(self, msg: Any) -> None:
        """Write a single email to C2 and announce in #nexus."""
        self._emails_ingested += 1

        # Write to C2
        if self.bot.c2.is_running:
            try:
                c2_text = msg.to_c2_text()
                await self.bot.c2.write_event(
                    actor="email",
                    intent="email_received",
                    inp=f"From: {msg.sender} | Subject: {msg.subject}"[:500],
                    out=c2_text[:1500],
                    tags=["email", "ingestion"],
                )
            except Exception:
                log.debug("Failed to write email to C2.", exc_info=True)

        # Store in vector memory for semantic search
        if self.bot.memory_store.is_connected:
            try:
                text = f"Email from {msg.sender}: {msg.subject}\n{msg.body[:2000]}"
                vector = await self.bot.embeddings.embed_one(text)
                await self.bot.memory_store.store(
                    content=text,
                    vector=vector,
                    source="email",
                    channel="nexus",
                    metadata={
                        "type": "email",
                        "sender": msg.sender[:100],
                        "subject": msg.subject[:200],
                    },
                )
            except Exception:
                log.debug("Failed to store email in memory.", exc_info=True)

        # Announce in #nexus
        try:
            import discord

            embed = discord.Embed(
                title=f"New Email: {msg.subject[:200]}",
                color=0xE91E63,
            )
            embed.add_field(name="From", value=msg.sender[:200], inline=True)
            embed.add_field(name="Date", value=msg.date[:100], inline=True)
            if msg.attachment_names:
                embed.add_field(
                    name="Attachments",
                    value=", ".join(msg.attachment_names[:5]),
                    inline=False,
                )
            # Preview of body
            preview = msg.body[:500].replace("\n", " ")
            embed.add_field(name="Preview", value=preview or "(empty)", inline=False)
            await self.bot.router.nexus.send(embed=embed)
        except Exception:
            log.debug("Failed to announce email in #nexus.", exc_info=True)
