"""Context composition for Continuity Core."""

from .composer import Candidate, ContextComposer
from .gather import CandidateGatherer
from .pipeline import ContextPipeline, ContextResult

__all__ = ["Candidate", "ContextComposer", "CandidateGatherer", "ContextPipeline", "ContextResult"]
