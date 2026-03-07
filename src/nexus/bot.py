"""NexusBot: The main Discord bot that orchestrates the AI swarm."""

from __future__ import annotations

import asyncio
import logging

import discord
from discord.ext import commands

from nexus.channels.router import ChannelRouter
from nexus.config import get_settings
from nexus.integrations.c2_engine import C2Engine
from nexus.integrations.email_monitor import EmailMonitor
from nexus.memory.context import ContextBuilder
from nexus.memory.store import MemoryStore
from nexus.models.embeddings import EmbeddingProvider
from nexus.models.ollama import OllamaClient
from nexus.models.openrouter import OpenRouterClient
from nexus.models.registry import ModelSpec, get_active_swarm, get_embedding
from nexus.orchestrator.activity import ActivityMonitor
from nexus.orchestrator.autonomy import AutonomyGate, AutonomyMode
from nexus.orchestrator.dispatch import TaskDispatcher
from nexus.orchestrator.goals import GoalStore
from nexus.orchestrator.health import HealthMonitor
from nexus.orchestrator.loop import OrchestratorLoop
from nexus.orchestrator.state import StateGatherer
from nexus.orchestrator.triggers import (
    ActivityTrigger,
    GoalStaleTrigger,
    MessageRateTrigger,
    ScheduledTrigger,
    TriggerManager,
)
from nexus.personality.prompts import build_system_prompt
from nexus.swarm.consensus import ConsensusProtocol
from nexus.swarm.conversation import ConversationManager
from nexus.swarm.crosstalk import CrosstalkManager
from nexus.swarm.initiative import SwarmInitiative
from nexus.swarm.sentiment import SentimentTracker
from nexus.swarm.session import SessionManager
from nexus.synthesis.tdd_engine import NexusLLMAdapter, TDDEngine

log = logging.getLogger(__name__)


class NexusBot(commands.Bot):
    """Multi-model AI swarm orchestrated through Discord.

    Wires together all subsystems:
    - OpenRouter + Ollama model clients
    - Qdrant memory store + embedding provider
    - 3-channel routing (human, nexus, memory)
    - Swarm conversation + crosstalk + consensus + initiative
    - Background orchestrator loop
    - Persistent goal store (Redis-backed)
    - Trigger manager (activity, message rate, goal staleness, scheduled)
    - Health monitor (periodic self-checks)
    - Continuity Core (C2) cognitive memory
    - Activity monitor (Pieces polling — legacy, subsumed by triggers)
    - Autonomy gate (observe/escalate/autopilot) with dynamic risk scoring
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
        self.initiative = SwarmInitiative(
            self,
            cooldown_minutes=settings.INITIATIVE_COOLDOWN_MINUTES,
            enabled=settings.INITIATIVE_ENABLED,
        )

        # --- Orchestrator ---
        self.state_gatherer = StateGatherer(self)
        self.dispatcher = TaskDispatcher(self)
        self.orchestrator = OrchestratorLoop(self, interval=settings.ORCHESTRATOR_INTERVAL)

        # --- Goal Store (Redis-backed persistence) ---
        self.goal_store = GoalStore(
            redis_url=settings.REDIS_URL,
            max_active_goals=settings.GOAL_MAX_ACTIVE,
            default_max_age_hours=settings.GOAL_MAX_AGE_HOURS,
        )

        # --- Continuity Core (C2) ---
        self.c2 = C2Engine(settings)

        # --- Synthesis TDD Engine ---
        self.tdd = TDDEngine(
            llm=NexusLLMAdapter(self.openrouter, model="qwen/qwen3-coder-next"),
        )

        # --- Autonomy Gate (with dynamic risk scoring) ---
        self.autonomy_gate = AutonomyGate(
            AutonomyMode(settings.AUTONOMY_MODE),
            bot=self,
        )

        # --- Activity Monitor (legacy — kept for backward compat) ---
        self.activity_monitor = ActivityMonitor(
            self,
            poll_interval=settings.ACTIVITY_POLL_INTERVAL,
        )

        # --- Trigger Manager (replaces sole reliance on ActivityMonitor) ---
        self.trigger_manager = TriggerManager(
            self,
            check_interval=settings.TRIGGER_CHECK_INTERVAL,
        )

        # --- Health Monitor ---
        self.health_monitor = HealthMonitor(
            self,
            check_interval=settings.HEALTH_CHECK_INTERVAL,
        )

        # --- Integrations ---
        self.pieces = None
        self.email_monitor = EmailMonitor(self)

        # --- Sentiment tracking ---
        self.sentiment = SentimentTracker()

        # --- Session lifecycle ---
        self.session = SessionManager(self)

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

        # Initialize goal store (Redis)
        await self.goal_store.connect()

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
                log.info("PiecesOS not available at startup — will retry on use")

        # Start Continuity Core (direct integration)
        c2_started = await self.c2.start()
        if c2_started:
            log.info("Continuity Core (C2) engine started (direct)")
        else:
            log.info("C2 not available - cognitive memory features disabled")

        # Restore previous session context from C2
        await self.session.on_startup()

        # Initialize LangGraph orchestrator (behind feature flag)
        self._orchestrator_graph = None
        if self.settings.LANGGRAPH_ENABLED:
            await self._init_langgraph()

        # Build system prompts for all swarm models
        model_ids = list(self.swarm_models.keys())
        for model_id in model_ids:
            self._system_prompts[model_id] = build_system_prompt(model_id, model_ids)

        # Start orchestrator
        await self.orchestrator.start()

        # Start activity monitor (legacy — if Pieces is available)
        if self.pieces is not None:
            await self.activity_monitor.start()

        # Register and start trigger manager
        self._setup_triggers()
        await self.trigger_manager.start()

        # Start health monitor
        await self.health_monitor.start()

        # Attach Discord log handler to pipe logs to #logs channel
        from nexus.channels.discord_log import DiscordLogHandler

        self._discord_log_handler = DiscordLogHandler(self.router.logs)
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        self._discord_log_handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logging.getLogger("nexus").addHandler(self._discord_log_handler)
        self._discord_log_handler.start()

        # Auto-ingest configured paths into C2 (background)
        if self.c2.is_running and self.settings.INGEST_PATHS:
            self._spawn(self._auto_ingest(self.settings.INGEST_PATHS))

        # Start email monitor (if configured)
        await self.email_monitor.start()

        # Announce in #nexus
        from nexus.personality.identities import format_name

        model_names = [format_name(mid) for mid in model_ids]
        features = []
        if self.goal_store.is_connected:
            features.append("goals")
        features.append(f"autonomy={self.autonomy_gate.mode.value}")
        features.append(f"c2={'on' if self.c2.is_running else 'off'}")
        features.append("tdd=on")
        features.append(f"triggers={len(self.trigger_manager._triggers)}")
        if self.email_monitor.is_configured:
            features.append("email=on")

        await self.router.nexus.send(
            f"**Agent Nexus online.** Swarm: {', '.join(model_names)} [{', '.join(features)}]"
        )

        # Post command reference for discoverability.
        cmd_embed = discord.Embed(title="Available Commands", color=0x3498DB)
        cmd_embed.add_field(
            name="Interaction",
            value=(
                "`!ask <model> <prompt>` — Ask a specific model\n"
                "`!think <prompt>` — Multi-perspective from all models\n"
                "`!memory <query>` — Search swarm memory\n"
                "`!remember <text>` — Store in memory\n"
                "`!forget <id>` — Delete a memory"
            ),
            inline=False,
        )
        cmd_embed.add_field(
            name="Monitoring",
            value=(
                "`!status` — Swarm health overview\n"
                "`!models` — List active models\n"
                "`!costs` — Session cost breakdown\n"
                "`!config` — Show configuration\n"
                "`!goals` — List active goals\n"
                "`!session` — Session info + mood\n"
                "`!pieces [query]` — Query PiecesOS activity"
            ),
            inline=False,
        )
        cmd_embed.add_field(
            name="Admin",
            value=(
                "`!crosstalk on/off` — Toggle crosstalk\n"
                "`!autonomy observe|escalate|autopilot` — Set autonomy\n"
                "`!curiosity` — Trigger epistemic scan\n"
                "`!c2status` — C2 backend health\n"
                "`!c2events [n]` — Recent C2 events\n"
                "`!discuss` — Trigger curiosity discussion\n"
                "`!ingest [paths]` — Ingest files into C2\n"
                "`!email [poll]` — Email monitor status"
            ),
            inline=False,
        )
        await self.router.nexus.send(embed=cmd_embed)

        # Announce previous session context if available
        await self.session.announce_restore()

        log.info(
            "Agent Nexus ready with %d swarm models (autonomy=%s, c2=%s, "
            "goals=%s, triggers=%d, health=on)",
            len(model_ids),
            self.autonomy_gate.mode.value,
            "connected" if self.c2.is_running else "offline",
            "redis" if self.goal_store.is_connected else "memory",
            len(self.trigger_manager._triggers),
        )

    async def _init_langgraph(self) -> None:
        """Initialize the LangGraph orchestrator graph.

        Creates LangChain LLMs, builds tools, sets up checkpointing,
        and compiles the graph.  Falls back gracefully on failure.
        """
        try:
            from nexus.models.langchain_adapter import (
                create_agent_llm,
                create_orchestrator_llm,
            )
            from nexus.orchestrator.graph import build_orchestrator_graph
            from nexus.orchestrator.tools import build_tools

            orchestrator_llm = create_orchestrator_llm(
                api_key=self.settings.OPENROUTER_API_KEY,
                model=self.settings.ORCHESTRATOR_MODEL,
            )
            agent_llm = create_agent_llm(
                api_key=self.settings.OPENROUTER_API_KEY,
                model=self.settings.TASK_AGENT_MODEL,
            )
            tools = build_tools(self)

            # Redis checkpointer (reuse existing nexus-redis).
            checkpointer = None
            try:
                from langgraph.checkpoint.redis.aio import AsyncRedisSaver

                checkpointer = AsyncRedisSaver.from_conn_string(
                    self.settings.REDIS_URL,
                )
                await checkpointer.setup()
                log.info("LangGraph: Redis checkpointer initialized.")
            except Exception as exc:
                log.info(
                    "LangGraph: Redis checkpointer unavailable (%s), using in-memory fallback.",
                    exc,
                )
                from langgraph.checkpoint.memory import MemorySaver

                checkpointer = MemorySaver()

            self._orchestrator_graph = build_orchestrator_graph(
                bot=self,
                orchestrator_llm=orchestrator_llm,
                agent_llm=agent_llm,
                tools=tools,
                checkpointer=checkpointer,
            )
            log.info(
                "LangGraph orchestrator initialized (orchestrator=%s, agent=%s).",
                self.settings.ORCHESTRATOR_MODEL,
                self.settings.TASK_AGENT_MODEL,
            )
        except Exception:
            log.error(
                "LangGraph initialization failed -- falling back to manual orchestrator.",
                exc_info=True,
            )
            self._orchestrator_graph = None

    def _setup_triggers(self) -> None:
        """Register all trigger sources with the trigger manager."""
        settings = self.settings

        # Activity change trigger (PiecesOS)
        if self.pieces is not None:
            self.trigger_manager.add_trigger(ActivityTrigger())

        # Message rate trigger
        self.trigger_manager.add_trigger(
            MessageRateTrigger(
                threshold=settings.MESSAGE_RATE_TRIGGER_THRESHOLD,
                window_minutes=5.0,
            )
        )

        # Goal staleness trigger
        self.trigger_manager.add_trigger(
            GoalStaleTrigger(stale_hours=settings.GOAL_MAX_AGE_HOURS / 12)
        )

        # Scheduled reflection trigger (every 6 hours)
        self.trigger_manager.add_trigger(ScheduledTrigger(interval_hours=6.0))

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

        # Handle human messages in #human or #nexus - forward to swarm
        is_human_msg = self.router.is_human_channel(
            message.channel.id
        ) or self.router.is_nexus_channel(message.channel.id)
        if is_human_msg and not message.content.startswith("!"):
            await self._handle_human_message(message)

        # Process commands regardless of channel
        await self.process_commands(message)

    async def _handle_human_message(self, message: discord.Message) -> None:
        """Handle a non-command message from #human. Forward to swarm."""
        from nexus.channels.formatter import MessageFormatter

        # Extract text from attachments (PDFs, images, office docs).
        content = message.content
        attachment_text = await self._extract_attachments(message)
        if attachment_text:
            content = f"{message.content}\n\n---\n**Attached Document:**\n{attachment_text}"

        # Analyse sentiment so models can adapt their tone.
        self.sentiment.analyze(content)
        current_mood = self.sentiment.current_mood

        # Record human message in conversation (with extracted attachment text).
        await self.conversation.add_message("human", content, is_human=True)

        # Log to C2 (with mood tag for searchable mood history).
        self._spawn(
            self._log_to_c2(
                actor="human",
                intent="message",
                inp=content[:1500],
                tags=["human", "input", f"mood:{current_mood.value}"],
                metadata={"origin": "human"},
            )
        )

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

            # Flag potential fabrication in swarm output
            from nexus.orchestrator.guardrails import check_swarm_fabrication

            fab_warnings = check_swarm_fabrication(response.content)
            if fab_warnings:
                footer = embed.footer.text or ""
                embed.set_footer(text=f"{footer} | Unverified claims detected".strip(" |"))

            last_msg = await self.router.nexus.send(embed=embed)

            # Store primary response in memory (background)
            if self.memory_store.is_connected:
                self._spawn(self._store_in_memory(response.content, primary_model))

            # Log model response to C2 — split at sentence boundaries
            # to avoid truncation artifacts in the MRA stress monitor.
            from nexus.channels.formatter import MessageFormatter

            chunks = MessageFormatter.split_for_storage(response.content)
            for ci, chunk in enumerate(chunks):
                self._spawn(
                    self._log_to_c2(
                        actor=primary_model,
                        intent="response",
                        out=chunk,
                        tags=["swarm", "nexus"],
                        metadata={
                            "origin": "swarm",
                            "chunk": f"{ci + 1}/{len(chunks)}",
                        },
                    )
                )

            # --- Reaction round: sequential + organic, but non-blocking ---
            if self.crosstalk.is_enabled:
                self._spawn(self._run_reaction_round(primary_model, model_ids, last_msg))

        except Exception as e:
            log.error("Error handling human message: %s", e, exc_info=True)
            try:
                await self.router.human.send(
                    "An error occurred processing your message. Check bot logs for details."
                )
            except Exception:
                pass

    async def _extract_attachments(self, message: discord.Message) -> str | None:
        """Extract text from supported message attachments (PDFs, images, etc.).

        Downloads each attachment, runs text extraction via pymupdf (with
        docling OCR fallback for scanned documents), and returns the
        combined text.  Returns ``None`` if there are no supported
        attachments or extraction fails.
        """
        if not message.attachments:
            return None

        from pathlib import Path

        from nexus.integrations.ocr import SUPPORTED_EXTENSIONS, DocumentExtractor

        extractor = DocumentExtractor()
        parts: list[str] = []

        for attachment in message.attachments[:3]:  # Max 3 attachments
            ext = Path(attachment.filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            try:
                await message.channel.send(f"Reading attachment: *{attachment.filename}*...")
                text = await extractor.extract_from_url(
                    attachment.url,
                    attachment.filename,
                )
                if text:
                    parts.append(f"### {attachment.filename}\n{text}")
            except Exception:
                log.warning(
                    "Failed to extract attachment: %s",
                    attachment.filename,
                    exc_info=True,
                )

        return "\n\n".join(parts) if parts else None

    async def _run_reaction_round(
        self, primary_model: str, model_ids: list[str], last_msg: discord.Message
    ) -> None:
        """Run the organic reaction round in the background.

        Each model sees what came before and decides to contribute or pass.
        Capped at 2 reactions so the channel doesn't get flooded.
        """
        from nexus.channels.formatter import MessageFormatter
        from nexus.orchestrator.guardrails import check_swarm_fabrication
        from nexus.swarm.crosstalk import CrosstalkManager

        reaction_order = self.crosstalk.build_reaction_order(
            primary_model,
            model_ids,
            mood=self.sentiment.current_mood.value,
            model_specs=self.swarm_models,
        )
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

                fab_warnings = check_swarm_fabrication(reaction.content)
                if fab_warnings:
                    footer = embed.footer.text or ""
                    embed.set_footer(text=f"{footer} | Unverified claims detected".strip(" |"))

                last_msg = await last_msg.reply(embed=embed, mention_author=False)
                reactions_posted += 1

                if self.memory_store.is_connected:
                    self._spawn(self._store_in_memory(reaction.content, reactor_id))

                # Log reaction to C2 — sentence-boundary split
                from nexus.channels.formatter import MessageFormatter

                r_chunks = MessageFormatter.split_for_storage(reaction.content)
                for ci, chunk in enumerate(r_chunks):
                    self._spawn(
                        self._log_to_c2(
                            actor=reactor_id,
                            intent="response",
                            out=chunk,
                            tags=["swarm", "nexus"],
                            metadata={
                                "origin": "swarm",
                                "chunk": f"{ci + 1}/{len(r_chunks)}",
                            },
                        )
                    )

            except asyncio.TimeoutError:
                log.warning(f"Reaction from {reactor_id} timed out (30s)")
            except Exception as e:
                log.error(f"Reaction error from {reactor_id}: {e}")

    async def _store_in_memory(self, content: str, source: str) -> None:
        """Store a response in vector memory (background task)."""
        try:
            vector = await self.embeddings.embed_one(content)
            metadata: dict[str, str] = {}
            tracker = getattr(self, "sentiment", None)
            if tracker is not None:
                metadata["mood"] = tracker.current_mood.value
                metadata["mood_score"] = str(round(tracker.average_score, 3))
            await self.memory_store.store(
                content=content,
                vector=vector,
                source=source,
                channel="nexus",
                metadata=metadata if metadata else None,
            )
        except Exception as e:
            log.warning(f"Failed to store response in memory: {e}")

    async def _auto_ingest(self, paths: list[str]) -> None:
        """Background task: ingest configured filesystem paths into C2."""
        try:
            from continuity_core.ingest.pipeline import IngestPipeline

            pipeline = IngestPipeline(
                chunk_size=self.settings.INGEST_CHUNK_SIZE,
                max_bytes=self.settings.INGEST_MAX_FILE_BYTES,
            )
            result = await asyncio.to_thread(pipeline.ingest_paths, paths)
            log.info(
                "Auto-ingest complete: %d files, %d docs, %d chunks (%.1fs)",
                result.files_seen,
                result.docs_ingested,
                result.chunks_ingested,
                result.duration_sec,
            )
            if result.docs_ingested > 0:
                await self._log_to_c2(
                    actor="system",
                    intent="auto_ingest",
                    inp=", ".join(paths)[:500],
                    out=f"docs={result.docs_ingested} chunks={result.chunks_ingested}",
                    tags=["ingest", "startup"],
                    metadata={"origin": "ingest"},
                )
        except Exception:
            log.warning("Auto-ingest failed.", exc_info=True)

    async def _log_to_c2(
        self,
        actor: str,
        intent: str,
        inp: str = "",
        out: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Log an event to C2 if available.  Failures are silently ignored."""
        if not self.c2.is_running:
            return
        try:
            await self.c2.write_event(
                actor=actor,
                intent=intent,
                inp=inp,
                out=out,
                tags=tags,
                metadata=metadata,
            )
        except Exception:
            pass

    def get_system_prompt(self, model_id: str) -> str:
        """Get the system prompt for a model, with mood and session context."""
        if model_id not in self._system_prompts:
            self._system_prompts[model_id] = build_system_prompt(
                model_id, list(self.swarm_models.keys())
            )
        parts = [self._system_prompts[model_id]]

        # Inject previous session context for continuity
        prev = self.session.last_session_summary
        if prev:
            parts.append(f"\n\n## Previous Session\n{prev[:500]}\n")

        # Inject current user mood
        mood_ctx = self.sentiment.mood_context_for_prompt()
        if mood_ctx:
            parts.append(mood_ctx)

        return "".join(parts)

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
        # Persist session summary before stopping subsystems
        await self.session.on_shutdown()
        await self.email_monitor.stop()
        await self.health_monitor.stop()
        await self.trigger_manager.stop()
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
