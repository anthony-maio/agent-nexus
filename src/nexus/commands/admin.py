"""Admin and configuration commands for Agent Nexus.

Provides Discord commands for inspecting and managing the running swarm:
model listing (``!models``), cost tracking (``!costs``), configuration
display (``!config``), and crosstalk toggling (``!crosstalk``).

These commands are informational and do not modify swarm memory or
conversation state.

Usage::

    # In bot startup:
    await bot.load_extension("nexus.commands.admin")
"""

from __future__ import annotations

import logging

import discord
from discord.ext import commands

from nexus.personality.identities import get_identity

log = logging.getLogger(__name__)


class AdminCommands(commands.Cog):
    """Admin and monitoring commands.

    This cog provides commands for operators to inspect swarm configuration,
    review active models and their pricing, monitor session costs, and toggle
    runtime behaviours like crosstalk.

    Attributes:
        bot: The parent bot instance that owns the swarm infrastructure.
    """

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ------------------------------------------------------------------
    # !models -- list active swarm models
    # ------------------------------------------------------------------

    @commands.command(name="models")
    async def list_models(self, ctx: commands.Context) -> None:
        """List active swarm models and their roles."""
        embed = discord.Embed(title="Active Models", color=0x3498DB)

        for model_id, spec in self.bot.swarm_models.items():
            identity = get_identity(model_id)
            embed.add_field(
                name=f"{identity.emoji} {identity.name} ({identity.role})",
                value=(
                    f"**Model:** `{model_id}`\n"
                    f"**Context:** {spec.context_window:,} tokens\n"
                    f"**Cost:** ${spec.cost_input_per_m:.2f}/"
                    f"${spec.cost_output_per_m:.2f} per M"
                ),
                inline=True,
            )

        await ctx.send(embed=embed)

    # ------------------------------------------------------------------
    # !costs -- session cost breakdown
    # ------------------------------------------------------------------

    @commands.command(name="costs")
    async def costs(self, ctx: commands.Context) -> None:
        """Show session cost breakdown."""
        embed = discord.Embed(title="Session Costs", color=0xE67E22)

        total = self.bot.openrouter.session_cost
        embed.add_field(name="Total", value=f"${total:.4f}", inline=False)

        # Cost limit warning
        from nexus.config import get_settings

        settings = get_settings()
        if total > settings.SESSION_COST_LIMIT * 0.8:
            embed.add_field(
                name="Warning",
                value=(
                    f"Approaching session cost limit "
                    f"(${settings.SESSION_COST_LIMIT:.2f})"
                ),
                inline=False,
            )

        await ctx.send(embed=embed)

    # ------------------------------------------------------------------
    # !config -- show non-sensitive configuration
    # ------------------------------------------------------------------

    @commands.command(name="config")
    async def show_config(self, ctx: commands.Context) -> None:
        """Show current configuration (non-sensitive)."""
        from nexus.config import get_settings

        settings = get_settings()

        embed = discord.Embed(title="Configuration", color=0x555555)
        embed.add_field(
            name="Swarm Models",
            value=str(len(settings.SWARM_MODELS)),
            inline=True,
        )
        embed.add_field(
            name="Embedding",
            value=settings.EMBEDDING_MODEL,
            inline=True,
        )
        embed.add_field(
            name="Qdrant",
            value=settings.QDRANT_URL,
            inline=True,
        )
        embed.add_field(
            name="Orchestrator Interval",
            value=f"{settings.ORCHESTRATOR_INTERVAL}s",
            inline=True,
        )
        embed.add_field(
            name="Crosstalk Probability",
            value=f"{settings.CROSSTALK_PROBABILITY:.0%}",
            inline=True,
        )
        embed.add_field(
            name="Consensus Threshold",
            value=f"{settings.CONSENSUS_THRESHOLD:.0%}",
            inline=True,
        )
        embed.add_field(
            name="Pieces MCP",
            value="Enabled" if settings.PIECES_MCP_ENABLED else "Disabled",
            inline=True,
        )

        await ctx.send(embed=embed)

    # ------------------------------------------------------------------
    # !crosstalk -- toggle spontaneous cross-model responses
    # ------------------------------------------------------------------

    @commands.command(name="crosstalk")
    async def toggle_crosstalk(
        self, ctx: commands.Context, state: str = ""
    ) -> None:
        """Toggle crosstalk on/off.

        Usage: !crosstalk on/off
        """
        if state.lower() == "on":
            self.bot.crosstalk.enable()
            await ctx.send("Crosstalk enabled.")
        elif state.lower() == "off":
            self.bot.crosstalk.disable()
            await ctx.send("Crosstalk disabled.")
        else:
            current = "enabled" if self.bot.crosstalk.is_enabled else "disabled"
            await ctx.send(
                f"Crosstalk is {current}. "
                f"Use `!crosstalk on` or `!crosstalk off`."
            )


async def setup(bot: commands.Bot) -> None:
    """Load the AdminCommands cog into the bot."""
    await bot.add_cog(AdminCommands(bot))
