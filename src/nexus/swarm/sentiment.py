"""Lightweight sentiment analysis for user messages.

Detects user mood from message text using keyword/pattern matching and
tracks it over a sliding window.  The detected mood is injected into
the swarm system prompt so models can adapt their tone.

No external dependencies -- uses a curated word-list approach that is
fast and deterministic.

Usage::

    from nexus.swarm.sentiment import SentimentTracker

    tracker = SentimentTracker()
    mood = tracker.analyze("I'm frustrated, nothing is working!")
    # mood.label == "frustrated"
    # mood.score == -0.6
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from enum import Enum


class Mood(str, Enum):
    """Broad mood categories detected from user messages."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"
    CURIOUS = "curious"
    URGENT = "urgent"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class SentimentResult:
    """Result of analysing a single message."""

    label: Mood
    score: float  # -1.0 (very negative) to 1.0 (very positive)
    confidence: float  # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Word lists (lowercase, compiled once)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = frozenset(
    "thanks thank awesome great excellent amazing wonderful perfect love "
    "appreciate helpful good nice cool fantastic brilliant impressive "
    "beautiful happy glad excited yes absolutely agree".split()
)

_NEGATIVE_WORDS = frozenset(
    "bad awful terrible horrible wrong broken fail failed error crash "
    "hate sucks useless worst disappointing poor ugly messy".split()
)

_FRUSTRATED_WORDS = frozenset(
    "frustrated frustrating annoying annoyed stuck confused wtf ugh "
    "impossible nothing works broken again still why can't stop "
    "waste wasting pointless".split()
)

_CURIOUS_WORDS = frozenset(
    "how what why when where which explain describe curious wonder "
    "thinking interested learn understand clarify elaborate".split()
)

_URGENT_WORDS = frozenset(
    "urgent asap immediately emergency critical help now please hurry "
    "deadline important priority blocker".split()
)

_QUESTION_RE = re.compile(r"\?\s*$", re.MULTILINE)
_EXCLAIM_RE = re.compile(r"!{2,}")
_CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _score_message(text: str) -> SentimentResult:
    """Score a single message by keyword matching."""
    lower = text.lower()
    words = set(re.findall(r"[a-z']+", lower))

    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    frust = len(words & _FRUSTRATED_WORDS)
    curious = len(words & _CURIOUS_WORDS)
    urgent = len(words & _URGENT_WORDS)

    # Boost frustrated if caps or repeated exclamation
    if _EXCLAIM_RE.search(text) or len(_CAPS_RE.findall(text)) >= 2:
        frust += 1

    # Boost curious if question mark
    if _QUESTION_RE.search(text):
        curious += 1

    total = pos + neg + frust + curious + urgent
    if total == 0:
        return SentimentResult(label=Mood.NEUTRAL, score=0.0, confidence=0.3)

    # Pick the dominant mood
    scores = {
        Mood.POSITIVE: pos,
        Mood.NEGATIVE: neg,
        Mood.FRUSTRATED: frust,
        Mood.CURIOUS: curious,
        Mood.URGENT: urgent,
    }
    dominant = max(scores, key=scores.get)  # type: ignore[arg-type]
    dominant_count = scores[dominant]
    confidence = min(1.0, dominant_count / max(total, 1) * (1 + dominant_count * 0.15))

    # Numeric score: positive direction for positive/curious, negative for neg/frust
    if dominant in (Mood.POSITIVE, Mood.CURIOUS):
        numeric = min(1.0, dominant_count * 0.3)
    elif dominant in (Mood.NEGATIVE, Mood.FRUSTRATED):
        numeric = max(-1.0, -dominant_count * 0.3)
    else:  # urgent
        numeric = -0.2

    return SentimentResult(label=dominant, score=numeric, confidence=confidence)


# ---------------------------------------------------------------------------
# Tracker (sliding window)
# ---------------------------------------------------------------------------


class SentimentTracker:
    """Tracks user sentiment over a sliding window of recent messages.

    Args:
        window_size: Number of recent analyses to keep.
    """

    def __init__(self, window_size: int = 10) -> None:
        self._window: deque[SentimentResult] = deque(maxlen=window_size)

    def analyze(self, text: str) -> SentimentResult:
        """Analyse a message and add to the sliding window."""
        result = _score_message(text)
        self._window.append(result)
        return result

    @property
    def current_mood(self) -> Mood:
        """The dominant mood across the recent window."""
        if not self._window:
            return Mood.NEUTRAL
        counts: dict[Mood, int] = {}
        for r in self._window:
            counts[r.label] = counts.get(r.label, 0) + 1
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    @property
    def average_score(self) -> float:
        """Average sentiment score across the window (-1 to 1)."""
        if not self._window:
            return 0.0
        return sum(r.score for r in self._window) / len(self._window)

    def mood_context_for_prompt(self) -> str:
        """Build a brief context string about user mood for system prompts.

        Returns an empty string if the mood is neutral (no special handling
        needed).
        """
        mood = self.current_mood

        if mood == Mood.NEUTRAL:
            return ""

        tone_hints: dict[Mood, str] = {
            Mood.POSITIVE: (
                "The user seems to be in a positive mood. Match their energy and be enthusiastic."
            ),
            Mood.NEGATIVE: (
                "The user seems unhappy or dissatisfied. "
                "Be empathetic, acknowledge their concern, and focus on solutions."
            ),
            Mood.FRUSTRATED: (
                "The user appears frustrated. Be patient, avoid jargon, "
                "give clear and direct answers, and acknowledge the difficulty."
            ),
            Mood.CURIOUS: (
                "The user is in an exploratory/learning mood. "
                "Be thorough in explanations and offer additional context."
            ),
            Mood.URGENT: (
                "The user has an urgent need. Be concise, prioritise the "
                "most critical information first, and skip preamble."
            ),
        }

        hint = tone_hints.get(mood, "")
        if not hint:
            return ""

        return f"\n\n## Current User Mood\n{hint}\n"
