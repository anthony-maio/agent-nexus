"""Consciousness Pilot — the verification layer between MRA/C2 and execution.

The Pilot mediates between the generative "subconscious" (MRA signals, memory
retrieval, context composition) and the agent's outward actions.  It implements
the four-step protocol from the paper:

    1. Review Intent — does this align with known goals?
    2. Check Safety — does it violate hard constraints?
    3. Uncertainty Check — is confidence too low to act?
    4. Commit / Abort — explicit sign-off or downgrade to question.
"""

from .verification import (
    PilotVerdict,
    VerificationResult,
    ConsciousnessPilot,
    SafetyConstraint,
)

__all__ = [
    "PilotVerdict",
    "VerificationResult",
    "ConsciousnessPilot",
    "SafetyConstraint",
]
