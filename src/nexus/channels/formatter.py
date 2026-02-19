"""Discord message formatting with model identity branding.

Every response posted by the swarm is rendered through this module so that
users can visually distinguish which model authored a message.  Embeds carry
the model's emoji, display name, colour, and role as defined in
:mod:`nexus.personality.identities`.

Usage::

    from nexus.channels.formatter import MessageFormatter

    embed = MessageFormatter.format_response("minimax/minimax-m2.5", "Hello!")
    await channel.send(embed=embed)
"""

from __future__ import annotations

import discord

from nexus.personality.identities import ModelIdentity, get_identity

# Discord hard limits
_EMBED_DESCRIPTION_LIMIT: int = 4096
_MESSAGE_CHAR_LIMIT: int = 2000


class MessageFormatter:
    """Format model responses for Discord with identity branding.

    All methods are static so the formatter can be used without instantiation.
    A shared instance is not required because the class carries no state.
    """

    @staticmethod
    def format_response(
        model_id: str,
        content: str,
        *,
        is_consensus: bool = False,
    ) -> discord.Embed:
        """Create a Discord embed for a model's response.

        Args:
            model_id: OpenRouter model identifier used to look up the
                model's :class:`~nexus.personality.identities.ModelIdentity`.
            content: The text body of the response.  Truncated to 4 096
                characters to respect the Discord embed description limit.
            is_consensus: When ``True`` a ``(consensus)`` tag is appended
                to the author line to indicate the response represents
                agreement across multiple models.

        Returns:
            A :class:`discord.Embed` styled with the model's colour and
            identity metadata.
        """
        identity: ModelIdentity = get_identity(model_id)

        embed = discord.Embed(
            description=content[:_EMBED_DESCRIPTION_LIMIT],
            color=identity.color,
        )

        title = f"{identity.emoji} {identity.name}"
        if is_consensus:
            title += " (consensus)"
        embed.set_author(name=title)
        embed.set_footer(text=f"{identity.role} | {model_id}")

        return embed

    @staticmethod
    def format_plain(model_id: str, content: str) -> str:
        """Format as plain text with an identity prefix.

        Useful for longer messages or contexts where embeds are not
        appropriate (e.g. logging, CLI output).

        Args:
            model_id: OpenRouter model identifier.
            content: The message body.

        Returns:
            A Markdown-formatted string with the model's emoji, bolded
            name, and role.
        """
        identity: ModelIdentity = get_identity(model_id)
        return f"{identity.emoji} **{identity.name}** ({identity.role}):\n{content}"

    @staticmethod
    def format_memory_log(action: str, detail: str) -> discord.Embed:
        """Format a memory audit-log entry for the ``#memory`` channel.

        Args:
            action: Short label describing the memory operation (e.g.
                ``"write"``, ``"delete"``, ``"daily_summary"``).
            detail: Full description of what was stored or changed.

        Returns:
            A neutral-grey :class:`discord.Embed`.
        """
        return discord.Embed(
            title=f"Memory: {action}",
            description=detail[:_EMBED_DESCRIPTION_LIMIT],
            color=0x555555,
        )

    @staticmethod
    def format_alert(title: str, message: str) -> discord.Embed:
        """Format an alert for the ``#human`` channel.

        Alerts use a red accent colour to attract attention for items that
        require human review or approval.

        Args:
            title: Short summary of the alert.
            message: Detailed explanation.

        Returns:
            A red-accented :class:`discord.Embed`.
        """
        return discord.Embed(
            title=f"Alert: {title}",
            description=message[:_EMBED_DESCRIPTION_LIMIT],
            color=0xFF4444,
        )

    @staticmethod
    def truncate_for_discord(text: str, limit: int = _MESSAGE_CHAR_LIMIT) -> str:
        """Truncate *text* to fit within Discord's message character limit.

        Args:
            text: The raw text to truncate.
            limit: Maximum allowed length.  Defaults to 2 000 (Discord's
                standard message limit).

        Returns:
            The original text if it fits, otherwise the text trimmed with a
            ``...(truncated)`` suffix.
        """
        if len(text) <= limit:
            return text
        return text[: limit - 20] + "\n\n...(truncated)"
