"""Core swarm interaction commands for Agent Nexus.

Provides the primary Discord commands that humans use to interact with the
model swarm: directed questions (``!ask``), multi-perspective queries
(``!think``), system status (``!status``), and a help overview
(``!help_nexus``).

All commands are implemented as a :class:`discord.ext.commands.Cog` and
loaded via the standard ``setup()`` entrypoint.

Usage::

    # In bot startup:
    await bot.load_extension("nexus.commands.core")
"""

from __future__ import annotations

import asyncio
import logging

import discord
from discord.ext import commands

from nexus.channels.formatter import MessageFormatter
from nexus.personality.identities import IDENTITIES, format_name, get_identity

log = logging.getLogger(__name__)


class CoreCommands(commands.Cog):
    """Core swarm interaction commands.

    This cog provides the everyday commands that the human operator uses to
    direct questions at individual models or the entire swarm, inspect system
    health, and discover available commands.

    Attributes:
        bot: The parent bot instance that owns the swarm infrastructure
            (OpenRouter client, conversation manager, channel router, etc.).
    """

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # ------------------------------------------------------------------
    # !ask -- direct a question to a single model
    # ------------------------------------------------------------------

    @commands.command(name="ask")
    async def ask(self, ctx: commands.Context, model_name: str, *, prompt: str) -> None:
        """Ask a specific model a question.

        Usage: !ask atlas What is the best approach for this?
        Model names: atlas, sage, nova, cipher (or full model ID)
        """
        model_id = self._resolve_model(model_name)
        if not model_id:
            await ctx.send(
                f"Unknown model: `{model_name}`. Use `!models` to see available models."
            )
            return

        async with ctx.typing():
            # Build context with conversation history
            system_prompt = self.bot.get_system_prompt(model_id)
            messages = self.bot.conversation.build_messages_for_model(
                model_id, system_prompt, limit=10
            )
            messages.append({"role": "user", "content": prompt})

            try:
                response = await self.bot.openrouter.chat(
                    model=model_id,
                    messages=messages,
                )

                # Record in conversation
                await self.bot.conversation.add_message("human", prompt, is_human=True)
                await self.bot.conversation.add_message(model_id, response.content)

                # Post to #nexus
                embed = MessageFormatter.format_response(model_id, response.content)
                footer_text = embed.footer.text or ""
                embed.set_footer(
                    text=(
                        f"{footer_text} | "
                        f"{response.input_tokens}+{response.output_tokens} tokens | "
                        f"${response.cost:.4f}"
                    )
                )
                await self.bot.router.nexus.send(embed=embed)

                # Trigger crosstalk -- other models may spontaneously respond
                responders = await self.bot.crosstalk.select_responders(
                    model_id, list(self.bot.swarm_models.keys())
                )
                for responder_id in responders:
                    asyncio.create_task(self._crosstalk_respond(responder_id))

            except Exception as exc:
                log.exception("!ask command failed for model %s", model_id)
                await ctx.send(f"Error: {exc}")

    # ------------------------------------------------------------------
    # !think -- ask all swarm models in parallel
    # ------------------------------------------------------------------

    @commands.command(name="think")
    async def think(self, ctx: commands.Context, *, prompt: str) -> None:
        """Ask all swarm models to respond. Multi-perspective analysis.

        Usage: !think Should we refactor this module?
        """
        async with ctx.typing():
            await self.bot.conversation.add_message("human", prompt, is_human=True)

            # Query all swarm models in parallel
            tasks = [
                self._query_model(model_id, prompt)
                for model_id in self.bot.swarm_models
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for model_id, result in zip(self.bot.swarm_models, results):
                if isinstance(result, Exception):
                    log.error("!think error from %s: %s", model_id, result)
                    continue
                if result:
                    await self.bot.conversation.add_message(model_id, result.content)
                    embed = MessageFormatter.format_response(model_id, result.content)
                    await self.bot.router.nexus.send(embed=embed)

    # ------------------------------------------------------------------
    # !status -- system health overview
    # ------------------------------------------------------------------

    @commands.command(name="status")
    async def status(self, ctx: commands.Context) -> None:
        """Show swarm status: connected models, memory health, costs."""
        embed = discord.Embed(
            title="Agent Nexus Status",
            color=0x3498DB,
        )

        # Models
        model_lines = [
            f"  {format_name(model_id)}"
            for model_id in self.bot.swarm_models
        ]
        embed.add_field(
            name=f"Swarm Models ({len(self.bot.swarm_models)})",
            value="\n".join(model_lines) or "None",
            inline=False,
        )

        # Memory
        mem_status = "Disconnected"
        if self.bot.memory_store.is_connected:
            count = await self.bot.memory_store.count()
            mem_status = f"Connected ({count} memories)"
        embed.add_field(name="Memory (Qdrant)", value=mem_status, inline=True)

        # Continuity Core (C2)
        c2 = getattr(self.bot, "c2", None)
        if c2 is not None:
            c2_status = "Connected" if c2.is_running else "Offline"
        else:
            c2_status = "Not configured"
        embed.add_field(name="Continuity Core", value=c2_status, inline=True)

        # Autonomy mode
        gate = getattr(self.bot, "autonomy_gate", None)
        if gate is not None:
            embed.add_field(name="Autonomy", value=gate.mode.value, inline=True)

        # Orchestrator
        orch_status = "Running" if self.bot.orchestrator.is_running else "Stopped"
        embed.add_field(
            name="Orchestrator",
            value=f"{orch_status} ({self.bot.orchestrator.cycles_completed} cycles)",
            inline=True,
        )

        # Costs
        cost = self.bot.openrouter.session_cost
        embed.add_field(name="Session Cost", value=f"${cost:.4f}", inline=True)

        # Conversation
        embed.add_field(
            name="Conversation",
            value=f"{self.bot.conversation.message_count} messages",
            inline=True,
        )

        await ctx.send(embed=embed)

    # ------------------------------------------------------------------
    # !help_nexus -- command reference
    # ------------------------------------------------------------------

    @commands.command(name="help_nexus")
    async def help_nexus(self, ctx: commands.Context) -> None:
        """Show available commands."""
        embed = discord.Embed(
            title="Agent Nexus Commands",
            color=0x3498DB,
        )
        embed.add_field(
            name="!ask <model> <prompt>",
            value="Ask a specific model",
            inline=False,
        )
        embed.add_field(
            name="!think <prompt>",
            value="Ask all swarm models (multi-perspective)",
            inline=False,
        )
        embed.add_field(
            name="!memory <query>",
            value="Search swarm memory",
            inline=False,
        )
        embed.add_field(
            name="!remember <text>",
            value="Store something in memory",
            inline=False,
        )
        embed.add_field(name="!models", value="List available models", inline=False)
        embed.add_field(name="!costs", value="Show session costs", inline=False)
        embed.add_field(name="!status", value="Show swarm status", inline=False)

        await ctx.send(embed=embed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _query_model(self, model_id: str, prompt: str):
        """Query a single model with conversation context.

        Args:
            model_id: OpenRouter model identifier.
            prompt: The user's prompt text.

        Returns:
            A :class:`~nexus.models.openrouter.ChatResponse` from the model.
        """
        system_prompt = self.bot.get_system_prompt(model_id)
        messages = self.bot.conversation.build_messages_for_model(
            model_id, system_prompt, limit=10
        )
        messages.append({"role": "user", "content": prompt})
        return await self.bot.openrouter.chat(model=model_id, messages=messages)

    async def _crosstalk_respond(self, model_id: str) -> None:
        """Generate a crosstalk response from a model.

        Called as a fire-and-forget task when another model's response triggers
        spontaneous cross-model conversation.

        Args:
            model_id: The model that should produce a follow-up response.
        """
        try:
            system_prompt = self.bot.get_system_prompt(model_id)
            messages = self.bot.conversation.build_messages_for_model(
                model_id, system_prompt, limit=10
            )
            response = await self.bot.openrouter.chat(
                model=model_id, messages=messages
            )
            await self.bot.conversation.add_message(model_id, response.content)
            embed = MessageFormatter.format_response(model_id, response.content)
            await self.bot.router.nexus.send(embed=embed)
        except Exception as exc:
            log.error("Crosstalk error from %s: %s", model_id, exc)

    def _resolve_model(self, name: str) -> str | None:
        """Resolve a friendly model name (like 'atlas') to its model ID.

        Performs two lookups:
        1. Checks if *name* is already a valid model ID in the active swarm.
        2. Searches registered identities by display name (case-insensitive).

        Args:
            name: A display name or model ID string.

        Returns:
            The resolved OpenRouter model ID, or ``None`` if no match is found.
        """
        # Check if it is already a model ID
        if name in self.bot.swarm_models:
            return name

        # Check identity display names (case-insensitive)
        name_lower = name.lower()
        for model_id, identity in IDENTITIES.items():
            if identity.name.lower() == name_lower:
                if model_id in self.bot.swarm_models:
                    return model_id

        return None


async def setup(bot: commands.Bot) -> None:
    """Load the CoreCommands cog into the bot."""
    await bot.add_cog(CoreCommands(bot))
