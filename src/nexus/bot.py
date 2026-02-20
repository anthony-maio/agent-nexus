"""NexusBot: The main Discord bot that orchestrates the AI swarm."""

from __future__ import annotations

import asyncio
import logging

import discord
from discord.ext import commands

from nexus.channels.router import ChannelRouter
from nexus.config import get_settings
from nexus.integrations.c2_client import C2Client
from nexus.memory.context import ContextBuilder
from nexus.memory.store import MemoryStore
from nexus.models.embeddings import EmbeddingProvider
from nexus.models.ollama import OllamaClient
from nexus.models.openrouter import OpenRouterClient
from nexus.models.registry import get_active_swarm, get_embedding, ModelSpec
from nexus.orchestrator.activity import ActivityMonitor
from nexus.orchestrator.autonomy import AutonomyGate, AutonomyMode
from nexus.orchestrator.dispatch import TaskDispatcher
from nexus.orchestrator.loop import OrchestratorLoop
from nexus.orchestrator.state import StateGatherer
from nexus.personality.prompts import build_system_prompt
from nexus.swarm.consensus import ConsensusProtocol
from nexus.swarm.conversation import ConversationManager
from nexus.swarm.crosstalk import CrosstalkManager

log = logging.getLogger(__name__)


class NexusBot(commands.Bot):
    """Multi-model AI swarm orchestrated through Discord.

    Wires together all subsystems:
    - OpenRouter + Ollama model clients
    - Qdrant memory store + embedding provider
    - 3-channel routing (human, nexus, memory)
    - Swarm conversation + crosstalk + consensus
    - Background orchestrator loop
    - Continuity Core (C2) cognitive memory
    - Activity monitor (Pieces polling)
    - Autonomy gate (observe/escalate/autopilot)
    - PiecesOS integration (optional)
    """

    def __init__(self) -> None:
        settings = get_settings()

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True

        super().__init__(
            command_prefix="!",
            intents=intents,
            help_command=None,  # We provide our own !help_nexus
        )

        # --- Config ---
        self.settings = settings

        # --- Models ---
        self.openrouter = OpenRouterClient(api_key=settings.OPENROUTER_API_KEY)
        self.ollama: OllamaClient | None = OllamaClient(base_url=settings.OLLAMA_BASE_URL)

        # Active swarm models (Tier 1)
        swarm_specs = get_active_swarm(settings.SWARM_MODELS)
        self.swarm_models: dict[str, ModelSpec] = {s.id: s for s in swarm_specs}

        # --- Embeddings ---
        embedding_spec = get_embedding(settings.EMBEDDING_MODEL)
        self.embeddings = EmbeddingProvider(
            model_id=settings.EMBEDDING_MODEL,
            openrouter_client=self.openrouter,
            ollama_client=self.ollama,
        )

        # --- Memory ---
        self.memory_store = MemoryStore(
            url=settings.QDRANT_URL,
            collection=settings.QDRANT_COLLECTION,
            dimensions=embedding_spec.dimensions,
        )

        # --- Context ---
        self.context_builder = ContextBuilder(
            memory_store=self.memory_store,
            embedding_provider=self.embeddings,
        )

        # --- Channels ---
        self.router = ChannelRouter()

        # --- Swarm ---
        self.conversation = ConversationManager()
        self.crosstalk = CrosstalkManager(probability=settings.CROSSTALK_PROBABILITY)
        self.consensus = ConsensusProtocol(threshold=settings.CONSENSUS_THRESHOLD)

        # --- Orchestrator ---
        self.state_gatherer = StateGatherer(self)
        self.dispatcher = TaskDispatcher(self)
        self.orchestrator = OrchestratorLoop(self, interval=settings.ORCHESTRATOR_INTERVAL)

        # --- Continuity Core (C2) ---
        self.c2 = C2Client()

        # --- Autonomy Gate ---
        self.autonomy_gate = AutonomyGate(AutonomyMode(settings.AUTONOMY_MODE))

        # --- Activity Monitor ---
        self.activity_monitor = ActivityMonitor(
            self, poll_interval=settings.ACTIVITY_POLL_INTERVAL,
        )

        # --- Integrations ---
        self.pieces = None

        # --- System prompt cache ---
        self._system_prompts: dict[str, str] = {}

        # --- Background task tracking ---
        self._background_tasks: set[asyncio.Task] = set()

    async def setup_hook(self) -> None:
        """Called after login, before the bot starts processing events."""
        # Load command cogs
        await self.load_extension("nexus.commands.core")
        await self.load_extension("nexus.commands.memory_cmds")
        await self.load_extension("nexus.commands.admin")
        log.info("Command cogs loaded")

    async def on_ready(self) -> None:
        """Called when the bot is fully connected to Discord."""
        log.info(f"Logged in as {self.user} (ID: {self.user.id})")

        # Resolve guild
        guild = self._resolve_guild()
        if not guild:
            log.error("No guild found. Invite the bot to a server first.")
            return

        log.info(f"Connected to guild: {guild.name} (ID: {guild.id})")

        # Ensure channels exist
        await self.router.ensure_channels(guild)

        # Initialize memory store
        try:
            await self.memory_store.initialize()
            log.info("Memory store initialized")
        except Exception as e:
            log.warning(f"Memory store initialization failed: {e}")

        # Check Ollama availability
        if self.ollama:
            available = await self.ollama.is_available()
            if available:
                log.info("Ollama is available for local task agents")
            else:
                log.info("Ollama not available - task agents will use OpenRouter only")
                self.ollama = None

        # Connect PiecesOS if enabled
        if self.settings.PIECES_MCP_ENABLED:
            from nexus.integrations.pieces import PiecesMCPClient

            self.pieces = PiecesMCPClient(base_url=self.settings.PIECES_MCP_URL)
            connected = await self.pieces.connect()
            if connected:
                log.info("PiecesOS MCP connected")
            else:
                log.info("PiecesOS not available")
                self.pieces = None

        # Start Continuity Core subprocess
        c2_started = await self.c2.start()
        if c2_started:
            log.info("Continuity Core (C2) subprocess started")
        else:
            log.info("C2 not available - cognitive memory features disabled")

        # Build system prompts for all swarm models
        model_ids = list(self.swarm_models.keys())
        for model_id in model_ids:
            self._system_prompts[model_id] = build_system_prompt(model_id, model_ids)

        # Start orchestrator
        await self.orchestrator.start()

        # Start activity monitor (if Pieces is available)
        if self.pieces is not None:
            await self.activity_monitor.start()

        # Announce in #nexus
        from nexus.personality.identities import format_name

        model_names = [format_name(mid) for mid in model_ids]
        await self.router.nexus.send(
            f"**Agent Nexus online.** Swarm: {', '.join(model_names)}"
        )
        log.info(
            "Agent Nexus ready with %d swarm models (autonomy=%s, c2=%s)",
            len(model_ids),
            self.autonomy_gate.mode.value,
            "connected" if self.c2.is_running else "offline",
        )

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming Discord messages."""
        # Ignore our own messages
        if message.author == self.user:
            return

        # Guard: router not yet initialized (message arrived before on_ready)
        if not self.router._ready:
            await self.process_commands(message)
            return

        # Ignore messages outside our channels
        if not self.router.is_bot_channel(message.channel.id):
            await self.process_commands(message)
            return

        # Track in context builder
        self.context_builder.add_message(
            author=str(message.author),
            content=message.content,
            channel=message.channel.name,
            timestamp=message.created_at,
        )

        # Handle messages in #human - forward to swarm
        if self.router.is_human_channel(message.channel.id):
            if not message.content.startswith("!"):
                await self._handle_human_message(message)

        # Process commands regardless of channel
        await self.process_commands(message)

    async def _handle_human_message(self, message: discord.Message) -> None:
        """Handle a non-command message from #human. Forward to swarm."""
        from nexus.channels.formatter import MessageFormatter
        from nexus.swarm.crosstalk import CrosstalkManager

        # Record human message in conversation
        await self.conversation.add_message("human", message.content, is_human=True)

        # Log to C2
        self._spawn(self._log_to_c2(
            actor="human", intent="message",
            inp=message.content[:500], tags=["human", "input"],
        ))

        model_ids = list(self.swarm_models.keys())
        if not model_ids:
            return

        # Pick a random primary responder (so it's not always the same model)
        import random
        primary_model = random.choice(model_ids)

        try:
            # --- Primary response ---
            system_prompt = self.get_system_prompt(primary_model)
            messages = self.conversation.build_messages_for_model(
                primary_model, system_prompt, limit=15
            )

            response = await self.openrouter.chat(
                model=primary_model,
                messages=messages,
            )

            await self.conversation.add_message(primary_model, response.content)

            embed = MessageFormatter.format_response(primary_model, response.content)
            last_msg = await self.router.nexus.send(embed=embed)

            # Store primary response in memory (background)
            if self.memory_store.is_connected:
                self._spawn(self._store_in_memory(response.content, primary_model))

            # Log model response to C2
            self._spawn(self._log_to_c2(
                actor=primary_model, intent="response",
                out=response.content[:500], tags=["swarm", "nexus"],
            ))

            # --- Reaction round: sequential + organic, but non-blocking ---
            if self.crosstalk.is_enabled:
                self._spawn(self._run_reaction_round(primary_model, model_ids, last_msg))

        except Exception as e:
            log.error("Error handling human message: %s", e, exc_info=True)
            try:
                await self.router.human.send("An error occurred processing your message. Check bot logs for details.")
            except Exception:
                pass

    async def _run_reaction_round(
        self, primary_model: str, model_ids: list[str], last_msg: discord.Message
    ) -> None:
        """Run the organic reaction round in the background.

        Each model sees what came before and decides to contribute or pass.
        Capped at 2 reactions so the channel doesn't get flooded.
        """
        from nexus.channels.formatter import MessageFormatter
        from nexus.swarm.crosstalk import CrosstalkManager

        reaction_order = self.crosstalk.build_reaction_order(primary_model, model_ids)
        reaction_suffix = CrosstalkManager.get_reaction_suffix()
        reactions_posted = 0
        max_reactions = 2  # Cap to keep conversations tight

        for reactor_id in reaction_order:
            if reactions_posted >= max_reactions:
                break
            try:
                reactor_prompt = self.get_system_prompt(reactor_id) + reaction_suffix
                reactor_messages = self.conversation.build_messages_for_model(
                    reactor_id, reactor_prompt, limit=15
                )

                reaction = await asyncio.wait_for(
                    self.openrouter.chat(model=reactor_id, messages=reactor_messages),
                    timeout=30.0,
                )

                if CrosstalkManager.is_pass(reaction.content):
                    log.debug(f"{reactor_id} passed in reaction round")
                    continue

                await self.conversation.add_message(reactor_id, reaction.content)
                embed = MessageFormatter.format_response(reactor_id, reaction.content)
                last_msg = await last_msg.reply(embed=embed, mention_author=False)
                reactions_posted += 1

                if self.memory_store.is_connected:
                    self._spawn(self._store_in_memory(reaction.content, reactor_id))

                # Log reaction to C2
                self._spawn(self._log_to_c2(
                    actor=reactor_id, intent="response",
                    out=reaction.content[:500], tags=["swarm", "nexus"],
                ))

            except asyncio.TimeoutError:
                log.warning(f"Reaction from {reactor_id} timed out (30s)")
            except Exception as e:
                log.error(f"Reaction error from {reactor_id}: {e}")

    async def _store_in_memory(self, content: str, source: str) -> None:
        """Store a response in vector memory (background task)."""
        try:
            vector = await self.embeddings.embed_one(content)
            await self.memory_store.store(
                content=content,
                vector=vector,
                source=source,
                channel="nexus",
            )
        except Exception as e:
            log.warning(f"Failed to store response in memory: {e}")

    async def _log_to_c2(
        self,
        actor: str,
        intent: str,
        inp: str = "",
        out: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Log an event to C2 if available.  Failures are silently ignored."""
        if not self.c2.is_running:
            return
        try:
            await self.c2.write_event(actor=actor, intent=intent, inp=inp, out=out, tags=tags)
        except Exception:
            pass

    def get_system_prompt(self, model_id: str) -> str:
        """Get the system prompt for a model."""
        if model_id not in self._system_prompts:
            self._system_prompts[model_id] = build_system_prompt(
                model_id, list(self.swarm_models.keys())
            )
        return self._system_prompts[model_id]

    def _spawn(self, coro) -> asyncio.Task:
        """Create a tracked background task with error logging."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        """Callback for completed background tasks."""
        self._background_tasks.discard(task)
        if not task.cancelled() and task.exception():
            log.error("Background task failed: %s", task.exception(), exc_info=task.exception())

    def _resolve_guild(self) -> discord.Guild | None:
        """Resolve the target guild."""
        if self.settings.DISCORD_GUILD_ID:
            return self.get_guild(self.settings.DISCORD_GUILD_ID)
        # Auto-detect: use the first guild
        if self.guilds:
            return self.guilds[0]
        return None

    async def close(self) -> None:
        """Clean shutdown."""
        log.info("Shutting down Agent Nexus...")
        await self.activity_monitor.stop()
        await self.orchestrator.stop()
        await self.c2.stop()
        await self.openrouter.close()
        if self.ollama:
            await self.ollama.close()
        if self.pieces:
            await self.pieces.close()
        # Cancel remaining background tasks
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        await super().close()
