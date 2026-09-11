"""
tests/conftest.py
─────────────────────────────────────────────────────────────────────────────
Pytest configuration and shared fixtures.

Fixtures here are available to all test files without import.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_audio_metadata():
    """Return a valid AudioMetadata instance for use in tests."""
    from src.models import AudioMetadata
    return AudioMetadata(
        file_path="/tmp/test_meeting.wav",
        original_file="/tmp/test_meeting.wav",
        duration_seconds=300.0,  # 5 minutes
        sample_rate=16000,
        channels=1,
        file_format="wav",
        file_size_bytes=9_600_000,
    )


@pytest.fixture
def sample_transcript_segments():
    """Return a realistic list of TranscriptSegment instances."""
    from src.models import TranscriptSegment
    return [
        TranscriptSegment(start=0.0, end=4.2, text="Good morning everyone.", confidence=0.95),
        TranscriptSegment(start=4.3, end=8.1, text="Let's start the sprint planning.", confidence=0.92),
        TranscriptSegment(start=8.2, end=14.5, text="I'll finish the API integration by Friday.", confidence=0.89),
        TranscriptSegment(start=14.6, end=20.0, text="We decided to deploy on Monday.", confidence=0.94),
    ]


@pytest.fixture
def sample_aligned_segments():
    """Return a list of AlignedSegment instances with speaker labels."""
    from src.models import AlignedSegment
    return [
        AlignedSegment(speaker="Speaker 1", start=0.0, end=4.2, text="Good morning everyone."),
        AlignedSegment(speaker="Speaker 2", start=4.3, end=8.1, text="Let's start the sprint planning."),
        AlignedSegment(speaker="Speaker 1", start=8.2, end=14.5, text="I'll finish the API integration by Friday."),
        AlignedSegment(speaker="Speaker 2", start=14.6, end=20.0, text="We decided to deploy on Monday."),
    ]
