"""Discord channel routing and message formatting.

Public API:
    :class:`ChannelRouter` -- manages the 3-channel architecture.
    :class:`MessageFormatter` -- formats model responses for Discord.
    :data:`CHANNEL_SPECS` -- channel name/topic definitions.
"""

from nexus.channels.formatter import MessageFormatter
from nexus.channels.router import CHANNEL_SPECS, ChannelRouter

__all__ = [
    "CHANNEL_SPECS",
    "ChannelRouter",
    "MessageFormatter",
]
