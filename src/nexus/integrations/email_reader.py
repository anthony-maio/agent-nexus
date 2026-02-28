"""Email ingestion for Agent Nexus via IMAP.

Polls an IMAP mailbox for unread messages, extracts text content (plain
text and HTML-stripped), and feeds them into C2 via the event log.

Uses only Python standard library modules (``imaplib``, ``email``) so no
extra dependencies are required.

Usage::

    from nexus.integrations.email_reader import EmailReader

    reader = EmailReader(
        host="imap.gmail.com",
        port=993,
        address="user@gmail.com",
        password="app-specific-password",
    )
    emails = await reader.fetch_unread(limit=10)
"""

from __future__ import annotations

import asyncio
import email
import email.header
import email.utils
import imaplib
import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EmailMessage:
    """Parsed email message."""

    uid: str
    subject: str
    sender: str
    date: str
    body: str
    has_attachments: bool = False
    attachment_names: list[str] = field(default_factory=list)

    def to_c2_text(self) -> str:
        """Format for C2 event log ingestion."""
        parts = [
            f"From: {self.sender}",
            f"Subject: {self.subject}",
            f"Date: {self.date}",
        ]
        if self.attachment_names:
            parts.append(f"Attachments: {', '.join(self.attachment_names)}")
        parts.append("")
        parts.append(self.body[:5000])
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML stripper (lightweight, no dependencies)
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    """Simple HTML-to-text converter."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("br", "p", "div", "li", "tr", "h1", "h2", "h3"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse whitespace
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def _strip_html(html: str) -> str:
    """Convert HTML to plain text."""
    stripper = _HTMLStripper()
    try:
        stripper.feed(html)
        return stripper.get_text()
    except Exception:
        # Fallback: crude tag removal
        return re.sub(r"<[^>]+>", "", html).strip()


# ---------------------------------------------------------------------------
# Email parser
# ---------------------------------------------------------------------------


def _decode_header(raw: str | None) -> str:
    """Decode an RFC 2047 encoded header."""
    if not raw:
        return ""
    parts = email.header.decode_header(raw)
    decoded: list[str] = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return " ".join(decoded)


def _extract_body(msg: email.message.Message) -> str:
    """Extract the best text body from a MIME message."""
    if msg.is_multipart():
        plain_parts: list[str] = []
        html_parts: list[str] = []
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    plain_parts.append(payload.decode(charset, errors="replace"))
            elif ctype == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    html_parts.append(payload.decode(charset, errors="replace"))
        if plain_parts:
            return "\n".join(plain_parts)
        if html_parts:
            return _strip_html("\n".join(html_parts))
        return ""
    else:
        payload = msg.get_payload(decode=True)
        if not payload:
            return ""
        charset = msg.get_content_charset() or "utf-8"
        text = payload.decode(charset, errors="replace")
        if msg.get_content_type() == "text/html":
            return _strip_html(text)
        return text


def _get_attachment_names(msg: email.message.Message) -> list[str]:
    """List attachment filenames in a MIME message."""
    names: list[str] = []
    if not msg.is_multipart():
        return names
    for part in msg.walk():
        disp = part.get("Content-Disposition", "")
        if "attachment" in disp:
            fname = part.get_filename()
            if fname:
                names.append(_decode_header(fname))
    return names


def _parse_email(uid: str, raw_bytes: bytes) -> EmailMessage:
    """Parse raw email bytes into an EmailMessage."""
    msg = email.message_from_bytes(raw_bytes)
    subject = _decode_header(msg.get("Subject"))
    sender = _decode_header(msg.get("From"))
    date = msg.get("Date", "")
    body = _extract_body(msg)
    attachments = _get_attachment_names(msg)
    return EmailMessage(
        uid=uid,
        subject=subject,
        sender=sender,
        date=date,
        body=body,
        has_attachments=bool(attachments),
        attachment_names=attachments,
    )


# ---------------------------------------------------------------------------
# IMAP reader
# ---------------------------------------------------------------------------


class EmailReader:
    """Async IMAP email reader.

    All IMAP operations are synchronous and wrapped in
    ``asyncio.to_thread()`` to avoid blocking the event loop.

    Args:
        host: IMAP server hostname.
        port: IMAP server port (993 for SSL).
        address: Email address / username.
        password: Password or app-specific password.
        folder: IMAP folder to read (default ``INBOX``).
    """

    def __init__(
        self,
        host: str,
        port: int = 993,
        address: str = "",
        password: str = "",
        folder: str = "INBOX",
    ) -> None:
        self.host = host
        self.port = port
        self.address = address
        self.password = password
        self.folder = folder

    @property
    def is_configured(self) -> bool:
        """Return True if email settings are provided."""
        return bool(self.host and self.address and self.password)

    async def fetch_unread(self, limit: int = 10) -> list[EmailMessage]:
        """Fetch unread emails from the configured IMAP mailbox.

        Args:
            limit: Maximum number of unread emails to fetch.

        Returns:
            List of parsed :class:`EmailMessage` objects.
        """
        if not self.is_configured:
            return []
        return await asyncio.to_thread(self._fetch_sync, limit)

    def _fetch_sync(self, limit: int) -> list[EmailMessage]:
        """Synchronous IMAP fetch (runs in thread pool)."""
        messages: list[EmailMessage] = []
        conn: imaplib.IMAP4_SSL | None = None
        try:
            conn = imaplib.IMAP4_SSL(self.host, self.port)
            conn.login(self.address, self.password)
            conn.select(self.folder, readonly=True)

            status, data = conn.search(None, "UNSEEN")
            if status != "OK" or not data[0]:
                return []

            uids = data[0].split()
            # Take the most recent N
            uids = uids[-limit:]

            for uid_bytes in uids:
                uid = uid_bytes.decode("ascii")
                status, msg_data = conn.fetch(uid_bytes, "(RFC822)")
                if status != "OK" or not msg_data or not msg_data[0]:
                    continue
                raw = msg_data[0]
                if isinstance(raw, tuple) and len(raw) >= 2:
                    messages.append(_parse_email(uid, raw[1]))

        except Exception:
            log.warning("IMAP fetch failed.", exc_info=True)
        finally:
            if conn is not None:
                try:
                    conn.close()
                    conn.logout()
                except Exception:
                    pass

        return messages

    async def check_connection(self) -> bool:
        """Test the IMAP connection. Returns True on success."""
        if not self.is_configured:
            return False
        try:
            return await asyncio.to_thread(self._test_sync)
        except Exception:
            return False

    def _test_sync(self) -> bool:
        """Synchronous connection test."""
        conn = imaplib.IMAP4_SSL(self.host, self.port)
        conn.login(self.address, self.password)
        conn.select(self.folder, readonly=True)
        conn.close()
        conn.logout()
        return True
