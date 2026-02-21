"""Admin and configuration commands for Agent Nexus.

Provides Discord commands for inspecting and managing the running swarm:
model listing (``!models``), cost tracking (``!costs``), configuration
display (``!config``), crosstalk toggling (``!crosstalk``), autonomy mode
(``!autonomy``), curiosity scan (``!curiosity``), C2 backend health
(``!c2status``), C2 event log (``!c2events``), and status (``!status``).

Usage::

    # In bot startup:
    await bot.load_extension("nexus.commands.admin")
"""

from __future__ import annotations

import logging

import discord
from discord.ext import commands

from nexus.orchestrator.autonomy import AutonomyMode
from nexus.personality.identities import get_identity

log = logging.getLogger(__name__)


class AdminCommands(commands.Cog):
    """Admin and monitoring commands.

    This cog provides commands for operators to inspect swarm configuration,
    review active models and their pricing, monitor session costs, toggle
    runtime behaviours, manage autonomy mode, trigger curiosity scans,
    check C2 backend health, and browse C2 event logs.

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
            name="Autonomy Mode",
            value=self.bot.autonomy_gate.mode.value,
            inline=True,
        )
        embed.add_field(
            name="Crosstalk",
            value="Enabled" if self.bot.crosstalk.is_enabled else "Disabled",
            inline=True,
        )
        embed.add_field(
            name="Pieces MCP",
            value="Enabled" if settings.PIECES_MCP_ENABLED else "Disabled",
            inline=True,
        )
        embed.add_field(
            name="C2",
            value="Connected" if self.bot.c2.is_running else "Offline",
            inline=True,
        )

        await ctx.send(embed=embed)

    # ------------------------------------------------------------------
    # !crosstalk -- toggle spontaneous cross-model responses
    # ------------------------------------------------------------------

    @commands.command(name="crosstalk")
    @commands.has_permissions(administrator=True)
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

    # ------------------------------------------------------------------
    # !autonomy -- set autonomy mode
    # ------------------------------------------------------------------

    @commands.command(name="autonomy")
    @commands.has_permissions(administrator=True)
    async def set_autonomy(
        self, ctx: commands.Context, mode: str = ""
    ) -> None:
        """Set the orchestrator autonomy mode.

        Usage: !autonomy observe|escalate|autopilot
        """
        valid_modes = {m.value for m in AutonomyMode}
        mode_lower = mode.lower()

        if mode_lower in valid_modes:
            self.bot.autonomy_gate.set_mode(mode_lower)

            # Log mode change to C2
            self.bot._spawn(self.bot._log_to_c2(
                actor="human", intent="config",
                inp=f"autonomy={mode_lower}",
                tags=["autonomy", mode_lower],
            ))

            await ctx.send(f"Autonomy mode set to **{mode_lower}**.")
        else:
            current = self.bot.autonomy_gate.mode.value
            await ctx.send(
                f"Current mode: **{current}**\n"
                f"Usage: `!autonomy observe|escalate|autopilot`\n"
                f"- **observe**: All actions proposed in #human first\n"
                f"- **escalate**: Low-risk auto-executes, high-risk asks\n"
                f"- **autopilot**: Everything auto-executes"
            )

    # ------------------------------------------------------------------
    # !curiosity -- trigger a C2 curiosity scan
    # ------------------------------------------------------------------

    @commands.command(name="curiosity")
    @commands.has_permissions(administrator=True)
    async def curiosity_scan(self, ctx: commands.Context) -> None:
        """Trigger a C2 curiosity scan and display results."""
        if not self.bot.c2.is_running:
            await ctx.send("C2 is not running. Curiosity scan unavailable.")
            return

        await ctx.send("Scanning for epistemic tensions...")

        result = await self.bot.c2.curiosity()
        if result is None:
            await ctx.send("Curiosity scan returned no results.")
            return

        embed = discord.Embed(
            title="Curiosity Scan Results",
            color=0x9B59B6,
        )
        embed.add_field(
            name="Stress Level",
            value=f"{result.get('stress_level', 0):.3f}",
            inline=True,
        )
        embed.add_field(
            name="Suggested Action",
            value=result.get("suggested_action", "none"),
            inline=True,
        )

        contradictions = result.get("contradictions", [])
        if contradictions:
            lines = []
            for c in contradictions[:3]:
                lines.append(f"- {c.get('s1', '')[:60]} vs {c.get('s2', '')[:60]}")
            embed.add_field(
                name=f"Contradictions ({len(contradictions)})",
                value="\n".join(lines),
                inline=False,
            )

        tensions = result.get("deep_tensions", [])
        if tensions:
            lines = []
            for t in tensions[:3]:
                lines.append(f"- {t.get('s1', '')[:60]} vs {t.get('s2', '')[:60]}")
            embed.add_field(
                name=f"Deep Tensions ({len(tensions)})",
                value="\n".join(lines),
                inline=False,
            )

        questions = result.get("bridging_questions", [])
        if questions:
            lines = [f"- {q[:100]}" for q in questions[:3]]
            embed.add_field(
                name=f"Bridging Questions ({len(questions)})",
                value="\n".join(lines),
                inline=False,
            )

        await ctx.send(embed=embed)

    # ------------------------------------------------------------------
    # !c2status -- C2 backend health
    # ------------------------------------------------------------------

    @commands.command(name="c2status")
    async def c2_status(self, ctx: commands.Context) -> None:
        """Show Continuity Core backend health and system metrics."""
        if not self.bot.c2.is_running:
            await ctx.send("C2 is not running.")
            return

        result = await self.bot.c2.status()
        if result is None:
            await ctx.send("C2 status unavailable.")
            return

        embed = discord.Embed(title="Continuity Core Status", color=0x3498DB)
        embed.add_field(
            name="Neo4j",
            value=f"{result.get('neo4j', 'unknown')} ({result.get('neo4j_nodes', 0)} nodes)",
            inline=True,
        )
        embed.add_field(name="Qdrant (C2)", value=result.get("qdrant", "unknown"), inline=True)
        embed.add_field(name="Redis", value=result.get("redis", "unknown"), inline=True)
        evt_backend = result.get("event_backend", "unknown")
        evt_count = result.get("event_count", 0)
        embed.add_field(
            name="Event Log",
            value=f"{evt_backend} ({evt_count} events)",
            inline=True,
        )
        embed.add_field(
            name="Embeddings",
            value=result.get("embedding_backend", "unknown"),
            inline=True,
        )
        embed.add_field(
            name="MRA Stress",
            value=f"{result.get('stress_level', 0):.3f}",
            inline=True,
        )
        if result.get("fallback_memory_count", 0) > 0:
            embed.add_field(
                name="Fallback Memory",
                value=f"{result['fallback_memory_count']} items (in-memory)",
                inline=True,
            )

        await ctx.send(embed=embed)

    # ------------------------------------------------------------------
    # !c2events -- recent C2 event log
    # ------------------------------------------------------------------

    @commands.command(name="c2events")
    async def c2_events(self, ctx: commands.Context, limit: int = 10) -> None:
        """Show recent C2 events. Usage: !c2events [count]"""
        if not self.bot.c2.is_running:
            await ctx.send("C2 is not running.")
            return

        limit = max(1, min(limit, 20))
        result = await self.bot.c2.events(limit=limit)
        if result is None or not result.get("events"):
            await ctx.send("No C2 events found.")
            return

        from datetime import datetime, timezone

        lines = []
        for evt in result["events"]:
            ts = datetime.fromtimestamp(evt["timestamp"], tz=timezone.utc).strftime("%H:%M:%S")
            actor = evt.get("actor", "?")
            intent = evt.get("intent", "?")
            output = evt.get("output", "")[:80]
            lines.append(f"`{ts}` **{actor}** [{intent}] {output}")

        embed = discord.Embed(
            title=f"C2 Events (last {result.get('count', 0)})",
            description="\n".join(lines),
            color=0x555555,
        )
        await ctx.send(embed=embed)

    # ------------------------------------------------------------------
    # !discuss -- trigger curiosity-driven swarm discussion
    # ------------------------------------------------------------------

    @commands.command(name="discuss")
    async def discuss_curiosity(self, ctx: commands.Context) -> None:
        """Trigger the swarm to discuss C2 curiosity findings."""
        if not self.bot.c2.is_running:
            await ctx.send("C2 is not running. Cannot trigger discussion.")
            return

        await ctx.send("Querying C2 for epistemic tensions...")

        result = await self.bot.c2.curiosity()
        if result is None:
            await ctx.send("C2 returned no curiosity data.")
            return

        has_signals = (
            result.get("stress_level", 0) > 0.1
            or result.get("contradictions")
            or result.get("deep_tensions")
            or result.get("bridging_questions")
        )

        if not has_signals:
            await ctx.send("No epistemic tensions detected — nothing to discuss.")
            return

        await ctx.send(
            f"Found signals (stress={result.get('stress_level', 0):.3f}, "
            f"{len(result.get('contradictions', []))} contradiction(s), "
            f"{len(result.get('deep_tensions', []))} tension(s)). "
            f"Triggering swarm discussion..."
        )

        await self.bot.orchestrator._trigger_curiosity_discussion(result)

    # ------------------------------------------------------------------
    # !pieces -- query PiecesOS activity context
    # ------------------------------------------------------------------

    @commands.command(name="pieces")
    async def pieces_query(
        self, ctx: commands.Context, *, query: str = "recent activity and context"
    ) -> None:
        """Query PiecesOS LTM for recent user activity.

        Usage: !pieces [query]
        """
        if not self.bot.pieces.is_connected:
            await ctx.send(
                "PiecesOS is not connected. Check PIECES_MCP_ENABLED "
                "and PIECES_MCP_URL in config/.env."
            )
            return

        await ctx.send(f"Querying PiecesOS: *{query[:100]}*...")

        result = await self.bot.pieces.get_recent_activity(query=query)
        if result is None:
            await ctx.send("PiecesOS returned no results.")
            return

        # Truncate to Discord embed limit
        text = result[:4000] if len(result) > 4000 else result
        embed = discord.Embed(
            title="PiecesOS Activity",
            description=text,
            color=0x00BCD4,
        )
        await ctx.send(embed=embed)


async def setup(bot: commands.Bot) -> None:
    """Load the AdminCommands cog into the bot."""
    await bot.add_cog(AdminCommands(bot))
