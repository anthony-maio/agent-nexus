"""Discord bot commands (!ask, !think, !memory, etc.).

This package contains the three command cogs that form the Agent Nexus
Discord interface:

- :mod:`nexus.commands.core` -- ``!ask``, ``!think``, ``!status``, ``!help_nexus``
- :mod:`nexus.commands.memory_cmds` -- ``!memory``, ``!remember``, ``!forget``
- :mod:`nexus.commands.admin` -- ``!models``, ``!costs``, ``!config``, ``!crosstalk``

Load all cogs during bot startup::

    COMMAND_EXTENSIONS = [
        "nexus.commands.core",
        "nexus.commands.memory_cmds",
        "nexus.commands.admin",
    ]
    for ext in COMMAND_EXTENSIONS:
        await bot.load_extension(ext)
"""

COMMAND_EXTENSIONS: list[str] = [
    "nexus.commands.core",
    "nexus.commands.memory_cmds",
    "nexus.commands.admin",
]
