"""Memory interaction commands for Agent Nexus.

Provides Discord commands for searching, storing, and deleting entries in the
Qdrant-backed swarm memory: ``!memory``, ``!remember``, and ``!forget``.

The memory layer uses vector embeddings for semantic search, so queries are
matched by meaning rather than exact keyword.

Usage::

    # In bot startup:
    await bot.load_extension("nexus.commands.memory_cmds")
"""

from __future__ import annotations

import logging

import discord
from discord.ext import commands

from nexus.channels.formatter import MessageFormatter

log = logging.getLogger(__name__)


class MemoryCommands(commands.Cog):
    """Memory interaction commands.

    All commands in this cog require a connected Qdrant memory store and a
    functioning embedding provider. If either is unavailable the commands
    report the failure gracefully rather than raising.

    Attributes:
        bot: The parent bot instance that owns the memory store and
            embedding provider.
    """

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ------------------------------------------------------------------
    # !memory -- semantic search over stored memories
    # ------------------------------------------------------------------

    @commands.command(name="memory")
    @commands.cooldown(rate=5, per=60, type=commands.BucketType.user)
    async def memory_search(self, ctx: commands.Context, *, query: str) -> None:
        """Search swarm memory.

        Usage: !memory what were the main decisions about authentication?
        """
        if not self.bot.memory_store.is_connected:
            await ctx.send("Memory store not connected.")
            return

        async with ctx.typing():
            try:
                query_vector = await self.bot.embeddings.embed_one(query)
                memories = await self.bot.memory_store.search(
                    query_vector, limit=5
                )

                if not memories:
                    await ctx.send("No relevant memories found.")
                    return

                embed = discord.Embed(
                    title=f"Memory Search: {query[:100]}",
                    color=0x555555,
                )
                for i, mem in enumerate(memories, 1):
                    embed.add_field(
                        name=(
                            f"#{i} (relevance: {mem.score:.2f}, "
                            f"source: {mem.source})"
                        ),
                        value=mem.content[:1024],
                        inline=False,
                    )
                await ctx.send(embed=embed)

            except Exception as exc:
                log.exception("!memory search failed for query: %s", query[:80])
                await ctx.send("Memory search failed. Check bot logs for details.")

    # ------------------------------------------------------------------
    # !remember -- store a new memory
    # ------------------------------------------------------------------

    @commands.command(name="remember")
    @commands.cooldown(rate=5, per=60, type=commands.BucketType.user)
    async def remember(self, ctx: commands.Context, *, text: str) -> None:
        """Store something in swarm memory.

        Usage: !remember The team decided to use PostgreSQL for the main database.
        """
        if not self.bot.memory_store.is_connected:
            await ctx.send("Memory store not connected.")
            return

        async with ctx.typing():
            try:
                vector = await self.bot.embeddings.embed_one(text)
                memory_id = await self.bot.memory_store.store(
                    content=text,
                    vector=vector,
                    source="human",
                    channel="human",
                    metadata={"author": str(ctx.author)},
                )
                await ctx.send(f"Stored in memory (id: `{memory_id[:8]}...`)")

                # Log to #memory channel for audit trail
                log_embed = MessageFormatter.format_memory_log(
                    "Store",
                    f"**From:** {ctx.author}\n**Content:** {text[:500]}",
                )
                await self.bot.router.memory.send(embed=log_embed)

            except Exception as exc:
                log.exception("!remember failed for text: %s", text[:80])
                await ctx.send("Failed to store memory. Check bot logs for details.")

    # ------------------------------------------------------------------
    # !forget -- delete a memory by ID
    # ------------------------------------------------------------------

    @commands.command(name="forget")
    async def forget(self, ctx: commands.Context, memory_id: str) -> None:
        """Delete a memory by ID.

        Usage: !forget abc12345
        """
        if not self.bot.memory_store.is_connected:
            await ctx.send("Memory store not connected.")
            return

        try:
            await self.bot.memory_store.delete(memory_id)
            await ctx.send(f"Memory `{memory_id[:8]}...` deleted.")

            # Log deletion to #memory channel for audit trail
            log_embed = MessageFormatter.format_memory_log(
                "Delete",
                f"**Deleted by:** {ctx.author}\n**Memory ID:** {memory_id}",
            )
            await self.bot.router.memory.send(embed=log_embed)

        except Exception as exc:
            log.exception("!forget failed for memory_id: %s", memory_id)
            await ctx.send("Failed to delete memory. Check bot logs for details.")


async def setup(bot: commands.Bot) -> None:
    """Load the MemoryCommands cog into the bot."""
    await bot.add_cog(MemoryCommands(bot))
