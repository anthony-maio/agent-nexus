"""Tests for the consensus protocol vote parsing."""

import pytest

from nexus.swarm.consensus import ConsensusProtocol


@pytest.fixture
def protocol():
    return ConsensusProtocol(threshold=0.5)


def test_parse_approve(protocol):
    response = "DECISION: approve\nCONFIDENCE: 0.9\nREASONING: Looks good"
    vote = protocol._parse_vote("model-a", response)
    assert vote.decision == "approve"
    assert vote.confidence == 0.9


def test_parse_reject(protocol):
    response = "DECISION: reject\nCONFIDENCE: 0.8\nREASONING: Not safe"
    vote = protocol._parse_vote("model-a", response)
    assert vote.decision == "reject"
    assert vote.confidence == 0.8


def test_parse_approved_variant(protocol):
    response = "DECISION: approved.\nCONFIDENCE: 0.7\nREASONING: OK"
    vote = protocol._parse_vote("model-a", response)
    assert vote.decision == "approve"


def test_parse_yes_as_approve(protocol):
    response = "DECISION: yes\nCONFIDENCE: 0.9\nREASONING: Fine"
    vote = protocol._parse_vote("model-a", response)
    assert vote.decision == "approve"


def test_parse_no_as_reject(protocol):
    response = "DECISION: no\nCONFIDENCE: 0.6\nREASONING: Bad idea"
    vote = protocol._parse_vote("model-a", response)
    assert vote.decision == "reject"


def test_dont_approve_does_not_match_approve(protocol):
    """Review #7: 'I don't approve' should NOT parse as approve."""
    response = "DECISION: I don't approve\nCONFIDENCE: 0.5\nREASONING: Nope"
    vote = protocol._parse_vote("model-a", response)
    assert vote.decision != "approve"


def test_abstain_on_garbage(protocol):
    response = "DECISION: maybe\nCONFIDENCE: 0.5\nREASONING: dunno"
    vote = protocol._parse_vote("model-a", response)
    assert vote.decision == "abstain"


def test_confidence_clamped(protocol):
    response = "DECISION: approve\nCONFIDENCE: 1.5\nREASONING: Very sure"
    vote = protocol._parse_vote("model-a", response)
    assert vote.confidence == 1.0


def test_confidence_invalid_defaults(protocol):
    response = "DECISION: approve\nCONFIDENCE: not-a-number\nREASONING: test"
    vote = protocol._parse_vote("model-a", response)
    assert vote.confidence == 0.5  # default
