"""Multi-model swarm conversation, crosstalk, and consensus."""

from nexus.swarm.consensus import ConsensusOutcome, ConsensusProtocol, ConsensusResult, Vote
from nexus.swarm.conversation import ConversationManager, SwarmMessage
from nexus.swarm.crosstalk import CrosstalkManager

__all__ = [
    "ConsensusOutcome",
    "ConsensusProtocol",
    "ConsensusResult",
    "ConversationManager",
    "CrosstalkManager",
    "SwarmMessage",
    "Vote",
]
