#!/usr/bin/env python3
"""Interactive setup wizard for Agent Nexus.

Run this before ``docker compose up -d`` to generate a ``.env`` file
with validated configuration.

Usage::

    python setup/setup.py          # Interactive wizard
    python setup/setup.py --check  # Validate existing .env only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Resolve project root (one level up from setup/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

BANNER = r"""
  ╔═══════════════════════════════════════════╗
  ║          Agent Nexus Setup Wizard         ║
  ╚═══════════════════════════════════════════╝
"""

# Swarm model choices: (id, display_name, cost_hint, is_default)
SWARM_CHOICES: list[tuple[str, str, str, bool]] = [
    ("minimax/minimax-m2.5", "MiniMax M2.5", "Free - Programming #1, SWE-Bench 80.2%", True),
    ("z-ai/glm-5", "Z.ai GLM-5", "$0.30/$2.55 per M - Agentic planning, self-correction", True),
    ("moonshotai/kimi-k2.5", "Kimi K2.5", "$0.23/$3.00 per M - Multimodal, tool-calling", True),
    ("qwen/qwen3-coder-next", "Qwen3 Coder Next", "$0.12/$0.75 per M - MoE coding agent", True),
    ("anthropic/claude-sonnet-4-6", "Claude Sonnet 4.6", "$3.00/$15.00 per M - EXPENSIVE", False),
    ("google/gemini-3-flash", "Gemini 3 Flash", "$0.10/$0.40 per M - Fast multimodal", False),
    ("openai/chatgpt-5.2", "ChatGPT 5.2", "$2.50/$10.00 per M - EXPENSIVE", False),
]

# Embedding model choices: (id, display_name, cost_hint, is_default)
EMBEDDING_CHOICES: list[tuple[str, str, str, bool]] = [
    ("qwen/qwen3-embedding-8b", "Qwen3 Embedding 8B", "$0.01/M - Recommended (cheapest)", True),
    ("openai/text-embedding-3-small", "OpenAI Embedding 3 Small", "$0.02/M - Good alternative", False),
    ("openai/text-embedding-3-large", "OpenAI Embedding 3 Large", "$0.13/M - Best quality", False),
    ("google/gemini-embedding-001", "Gemini Embedding 001", "$0.15/M - Multilingual", False),
    ("mxbai-embed-large-v1", "Local (Ollama)", "Free - Requires Ollama installed", False),
]

# Styles
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_step(step: int, total: int, title: str) -> None:
    print(f"\n{CYAN}[{step}/{total}]{RESET} {BOLD}{title}{RESET}")
    print(f"{DIM}{'─' * 50}{RESET}")


def prompt_required(label: str, env_var: str, existing: dict[str, str]) -> str:
    """Prompt for a required value, showing existing if available."""
    current = existing.get(env_var, "")
    if current:
        masked = current[:8] + "..." if len(current) > 12 else current
        response = input(f"  {label} [{masked}]: ").strip()
        return response if response else current
    while True:
        response = input(f"  {label}: ").strip()
        if response:
            return response
        print(f"  {RED}This field is required.{RESET}")


def prompt_yes_no(label: str, default: bool = False) -> bool:
    """Prompt for a yes/no answer."""
    hint = "[Y/n]" if default else "[y/N]"
    response = input(f"  {label} {hint}: ").strip().lower()
    if not response:
        return default
    return response in ("y", "yes")


def prompt_choice(label: str, choices: list[tuple[str, str, str, bool]], multi: bool = False) -> list[str]:
    """Display numbered choices and let user select.

    For multi-select, defaults are pre-selected and user toggles.
    For single-select, default is highlighted.
    """
    print()
    defaults = [i for i, (_, _, _, is_default) in enumerate(choices) if is_default]

    for i, (model_id, name, hint, is_default) in enumerate(choices):
        marker = f"{GREEN}*{RESET}" if is_default else " "
        cost_warning = f" {YELLOW}WARNING: Expensive{RESET}" if "EXPENSIVE" in hint else ""
        print(f"  {marker} {i + 1}. {BOLD}{name}{RESET}")
        print(f"       {DIM}{hint}{RESET}{cost_warning}")

    if multi:
        default_str = ",".join(str(d + 1) for d in defaults)
        print(f"\n  {DIM}Enter numbers separated by commas. * = recommended defaults.{RESET}")
        response = input(f"  {label} [{default_str}]: ").strip()
        if not response:
            indices = defaults
        else:
            try:
                indices = [int(x.strip()) - 1 for x in response.split(",")]
                # Validate range
                for idx in indices:
                    if idx < 0 or idx >= len(choices):
                        print(f"  {RED}Invalid choice: {idx + 1}. Using defaults.{RESET}")
                        indices = defaults
                        break
            except ValueError:
                print(f"  {RED}Invalid input. Using defaults.{RESET}")
                indices = defaults

        if not indices:
            print(f"  {RED}Must select at least one. Using defaults.{RESET}")
            indices = defaults

        selected = [choices[i][0] for i in indices]
        print(f"  {GREEN}Selected: {', '.join(choices[i][1] for i in indices)}{RESET}")
        return selected
    else:
        # Single select
        default_idx = defaults[0] if defaults else 0
        response = input(f"  {label} [{default_idx + 1}]: ").strip()
        if not response:
            idx = default_idx
        else:
            try:
                idx = int(response) - 1
                if idx < 0 or idx >= len(choices):
                    print(f"  {RED}Invalid choice. Using default.{RESET}")
                    idx = default_idx
            except ValueError:
                print(f"  {RED}Invalid input. Using default.{RESET}")
                idx = default_idx

        selected_id = choices[idx][0]
        print(f"  {GREEN}Selected: {choices[idx][1]}{RESET}")
        return [selected_id]


def load_existing_env() -> dict[str, str]:
    """Load existing .env file if present."""
    env: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def write_env(config: dict[str, str]) -> None:
    """Write the .env file with all configuration."""
    lines = [
        "# Agent Nexus Configuration",
        "# Generated by setup wizard",
        "",
        "# === REQUIRED ===",
        f"DISCORD_TOKEN={config['DISCORD_TOKEN']}",
        f"OPENROUTER_API_KEY={config['OPENROUTER_API_KEY']}",
        "",
    ]

    if config.get("DISCORD_GUILD_ID"):
        lines.append(f"DISCORD_GUILD_ID={config['DISCORD_GUILD_ID']}")
        lines.append("")

    lines.extend([
        "# === MODELS ===",
        f"SWARM_MODELS={config['SWARM_MODELS']}",
        f"EMBEDDING_MODEL={config['EMBEDDING_MODEL']}",
        "",
    ])

    # Optional premium keys
    premium_keys = [
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("OPENAI_API_KEY", "OpenAI"),
        ("GOOGLE_API_KEY", "Google"),
    ]
    has_premium = any(config.get(k) for k, _ in premium_keys)
    if has_premium:
        lines.append("# === PREMIUM API KEYS ===")
        for key, _label in premium_keys:
            if config.get(key):
                lines.append(f"{key}={config[key]}")
        lines.append("")

    # Ollama
    if config.get("OLLAMA_ENABLED") == "true":
        lines.extend([
            "# === LOCAL MODELS ===",
            f"OLLAMA_BASE_URL={config.get('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')}",
            "",
        ])

    # PiecesOS
    if config.get("PIECES_MCP_ENABLED") == "true":
        lines.extend([
            "# === PIECESOS ===",
            "PIECES_MCP_ENABLED=true",
            f"PIECES_MCP_URL={config.get('PIECES_MCP_URL', 'http://host.docker.internal:39300')}",
            "",
        ])

    # Infrastructure defaults (include as comments so users know they exist)
    lines.extend([
        "# === INFRASTRUCTURE (defaults work with docker-compose) ===",
        "# QDRANT_URL=http://nexus-qdrant:6333",
        "# REDIS_URL=redis://nexus-redis:6379/0",
        "",
        "# === ORCHESTRATOR ===",
        "# ORCHESTRATOR_INTERVAL=3600",
        "# CROSSTALK_PROBABILITY=0.3",
        "# CONSENSUS_THRESHOLD=0.5",
        "",
    ])

    ENV_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  {GREEN}Wrote {ENV_FILE}{RESET}")


# ---------------------------------------------------------------------------
# Connection tests
# ---------------------------------------------------------------------------

def test_openrouter(api_key: str) -> bool:
    """Test OpenRouter API connectivity."""
    try:
        import urllib.request
        import json

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            model_count = len(data.get("data", []))
            print(f"  {GREEN}OpenRouter: Connected ({model_count} models available){RESET}")
            return True
    except Exception as e:
        print(f"  {RED}OpenRouter: Failed - {e}{RESET}")
        return False


def test_qdrant(url: str = "http://localhost:6333") -> bool:
    """Test Qdrant connectivity."""
    try:
        import urllib.request
        import json

        req = urllib.request.Request(f"{url}/collections")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            collections = len(data.get("result", {}).get("collections", []))
            print(f"  {GREEN}Qdrant: Connected ({collections} collections){RESET}")
            return True
    except Exception as e:
        print(f"  {YELLOW}Qdrant: Not reachable - {e}{RESET}")
        print(f"  {DIM}  (This is fine - Qdrant starts with docker compose){RESET}")
        return False


def test_ollama(url: str = "http://localhost:11434") -> bool:
    """Test Ollama connectivity."""
    try:
        import urllib.request

        req = urllib.request.Request(f"{url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json
            data = json.loads(resp.read())
            models = len(data.get("models", []))
            print(f"  {GREEN}Ollama: Connected ({models} models pulled){RESET}")
            return True
    except Exception as e:
        print(f"  {YELLOW}Ollama: Not reachable - {e}{RESET}")
        return False


# ---------------------------------------------------------------------------
# Wizard steps
# ---------------------------------------------------------------------------

def run_wizard() -> None:
    """Run the interactive setup wizard."""
    print(BANNER)
    print(f"  {DIM}This wizard will generate a .env file for Agent Nexus.{RESET}")
    print(f"  {DIM}Press Enter to accept defaults shown in [brackets].{RESET}")

    existing = load_existing_env()
    config: dict[str, str] = {}
    total_steps = 7

    # Step 1: Discord Token
    print_step(1, total_steps, "Discord Bot Token")
    print(f"  {DIM}Create a bot at https://discord.com/developers/applications{RESET}")
    print(f"  {DIM}Bot -> Token -> Copy{RESET}")
    config["DISCORD_TOKEN"] = prompt_required("Bot token", "DISCORD_TOKEN", existing)

    # Step 2: Invite URL
    print_step(2, total_steps, "Invite Bot to Your Server")
    print(f"  {DIM}Use this OAuth2 URL to invite your bot:{RESET}")
    print()

    # Extract client ID hint
    print(f"  {BOLD}Go to your bot's application page -> OAuth2 -> URL Generator{RESET}")
    print(f"  {DIM}Select scopes: bot{RESET}")
    print(f"  {DIM}Select permissions: Manage Channels, Send Messages, Embed Links,{RESET}")
    print(f"  {DIM}  Read Message History, Use External Emojis, Add Reactions{RESET}")
    print()

    guild_id = existing.get("DISCORD_GUILD_ID", "")
    response = input(f"  Guild ID (optional, auto-detects) [{guild_id or 'skip'}]: ").strip()
    if response:
        config["DISCORD_GUILD_ID"] = response
    elif guild_id:
        config["DISCORD_GUILD_ID"] = guild_id

    # Step 3: OpenRouter API Key
    print_step(3, total_steps, "OpenRouter API Key")
    print(f"  {DIM}Sign up free at https://openrouter.ai{RESET}")
    print(f"  {DIM}Keys -> Create Key -> Copy{RESET}")
    config["OPENROUTER_API_KEY"] = prompt_required("API key", "OPENROUTER_API_KEY", existing)

    # Step 4: Choose Swarm Models
    print_step(4, total_steps, "Choose Your Swarm (pick 2-4 models)")
    print(f"  {DIM}These models will converse and collaborate in your Discord server.{RESET}")
    print(f"  {DIM}All models see each other's messages and can build on each other.{RESET}")
    selected_swarm = prompt_choice("Your swarm", SWARM_CHOICES, multi=True)
    config["SWARM_MODELS"] = ",".join(selected_swarm)

    # Check if premium models need keys
    premium_models = {
        "anthropic/claude-sonnet-4-6": ("ANTHROPIC_API_KEY", "Anthropic"),
        "openai/chatgpt-5.2": ("OPENAI_API_KEY", "OpenAI"),
    }
    for model_id, (key_var, provider) in premium_models.items():
        if model_id in selected_swarm:
            print(f"\n  {YELLOW}{provider} model selected - API key required.{RESET}")
            config[key_var] = prompt_required(f"{provider} API key", key_var, existing)

    # Step 5: Choose Embedding Model
    print_step(5, total_steps, "Choose Embedding Model")
    print(f"  {RED}{BOLD}WARNING: This CANNOT be changed after first run!{RESET}")
    print(f"  {DIM}Changing embedding model invalidates all stored vectors.{RESET}")
    selected_embedding = prompt_choice("Embedding model", EMBEDDING_CHOICES, multi=False)
    config["EMBEDDING_MODEL"] = selected_embedding[0]

    # Step 6: Optional Integrations
    print_step(6, total_steps, "Optional Integrations")

    # Ollama
    print(f"\n  {BOLD}Local Models (Ollama){RESET}")
    print(f"  {DIM}Ollama runs LiquidAI task agents locally for free.{RESET}")
    print(f"  {DIM}Requires Ollama installed: https://ollama.com{RESET}")
    if prompt_yes_no("Enable Ollama integration?", default=False):
        config["OLLAMA_ENABLED"] = "true"
        ollama_url = existing.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        response = input(f"  Ollama URL [{ollama_url}]: ").strip()
        config["OLLAMA_BASE_URL"] = response if response else ollama_url

    # PiecesOS
    print(f"\n  {BOLD}PiecesOS Activity Tracking{RESET}")
    print(f"  {DIM}PiecesOS tracks your workflow and provides context to the swarm.{RESET}")
    print(f"  {DIM}Requires PiecesOS installed: https://pieces.app{RESET}")
    if prompt_yes_no("Enable PiecesOS integration?", default=False):
        config["PIECES_MCP_ENABLED"] = "true"
        pieces_url = existing.get("PIECES_MCP_URL", "http://host.docker.internal:39300")
        response = input(f"  PiecesOS MCP URL [{pieces_url}]: ").strip()
        config["PIECES_MCP_URL"] = response if response else pieces_url

    # Step 7: Test & Write
    print_step(7, total_steps, "Testing Connections")
    test_openrouter(config["OPENROUTER_API_KEY"])
    test_qdrant()
    if config.get("OLLAMA_ENABLED") == "true":
        test_ollama(config.get("OLLAMA_BASE_URL", "http://localhost:11434"))

    # Write .env
    print(f"\n{BOLD}Writing configuration...{RESET}")
    write_env(config)

    # Summary
    print(f"\n{GREEN}{'═' * 50}{RESET}")
    print(f"{GREEN}{BOLD}  Setup complete!{RESET}")
    print(f"{GREEN}{'═' * 50}{RESET}")
    print()
    print(f"  Swarm: {', '.join(s.split('/')[-1] for s in selected_swarm)}")
    print(f"  Embeddings: {config['EMBEDDING_MODEL'].split('/')[-1]}")
    print()
    print(f"  {BOLD}Next steps:{RESET}")
    print(f"  1. Make sure Docker is running")
    print(f"  2. Run: {CYAN}docker compose -f docker/docker-compose.yml up -d{RESET}")
    print(f"  3. The bot will auto-create #human, #nexus, and #memory channels")
    print(f"  4. Say hello in #human!")
    print()


# ---------------------------------------------------------------------------
# Check mode
# ---------------------------------------------------------------------------

def run_check() -> None:
    """Validate an existing .env file."""
    print(f"\n{BOLD}Checking existing configuration...{RESET}\n")

    if not ENV_FILE.exists():
        print(f"  {RED}No .env file found at {ENV_FILE}{RESET}")
        print(f"  {DIM}Run: python setup/setup.py{RESET}")
        sys.exit(1)

    env = load_existing_env()
    errors: list[str] = []
    warnings: list[str] = []

    # Required fields
    if not env.get("DISCORD_TOKEN"):
        errors.append("DISCORD_TOKEN is missing")
    if not env.get("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY is missing")

    # Models
    swarm = env.get("SWARM_MODELS", "")
    if swarm:
        models = [m.strip() for m in swarm.split(",")]
        if len(models) < 2:
            warnings.append(f"Only {len(models)} swarm model(s) configured. Recommend 2-4.")
        print(f"  Swarm models: {len(models)}")
    else:
        print(f"  Swarm models: defaults (4)")

    embedding = env.get("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
    print(f"  Embedding model: {embedding}")

    # Connectivity tests
    print()
    if env.get("OPENROUTER_API_KEY"):
        test_openrouter(env["OPENROUTER_API_KEY"])
    test_qdrant(env.get("QDRANT_URL", "http://localhost:6333"))
    if env.get("OLLAMA_BASE_URL"):
        test_ollama(env["OLLAMA_BASE_URL"])

    # Report
    print()
    if errors:
        for e in errors:
            print(f"  {RED}ERROR: {e}{RESET}")
        sys.exit(1)
    for w in warnings:
        print(f"  {YELLOW}WARNING: {w}{RESET}")
    print(f"  {GREEN}Configuration looks good!{RESET}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Nexus setup wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing .env without running the wizard",
    )
    args = parser.parse_args()

    if args.check:
        run_check()
    else:
        try:
            run_wizard()
        except KeyboardInterrupt:
            print(f"\n\n  {YELLOW}Setup cancelled.{RESET}")
            sys.exit(1)


if __name__ == "__main__":
    main()
