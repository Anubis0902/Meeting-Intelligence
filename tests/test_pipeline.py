"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the speaker alignment algorithm in src/pipeline.py.

These tests do NOT require an audio file, ASR model, or LLM API call.
They test the pure alignment logic with mock data.
"""

import pytest
from src.models import (
    AlignedSegment,
    DiarizationResult,
    DiarizationSegment,
    TranscriptSegment,
    TranscriptionResult,
)
from src.pipeline import align_speakers, _compute_overlap


class TestComputeOverlap:
    def test_no_overlap(self):
        assert _compute_overlap(1.0, 3.0, 5.0, 7.0) == 0.0

    def test_partial_overlap(self):
        assert _compute_overlap(1.0, 5.0, 3.0, 7.0) == pytest.approx(2.0)

    def test_full_containment(self):
        assert _compute_overlap(1.0, 10.0, 3.0, 7.0) == pytest.approx(4.0)

    def test_exact_match(self):
        assert _compute_overlap(1.0, 5.0, 1.0, 5.0) == pytest.approx(4.0)

    def test_adjacent_no_overlap(self):
        # End of A == Start of B → no overlap
        assert _compute_overlap(1.0, 3.0, 3.0, 5.0) == 0.0


class TestAlignSpeakers:
    def _make_transcription(self, segments: list[tuple]) -> TranscriptionResult:
        """Helper to build a TranscriptionResult from (start, end, text) tuples."""
        segs = [
            TranscriptSegment(start=s, end=e, text=t)
            for s, e, t in segments
        ]
        return TranscriptionResult(
            language="en",
            segments=segs,
            full_text=" ".join(t for _, _, t in segments),
        )

    def _make_diarization(self, segments: list[tuple]) -> DiarizationResult:
        """Helper to build a DiarizationResult from (speaker, start, end) tuples."""
        segs = [
            DiarizationSegment(speaker=sp, start=s, end=e)
            for sp, s, e in segments
        ]
        speakers = sorted(set(sp for sp, _, _ in segments))
        return DiarizationResult(
            segments=segs,
            num_speakers=len(speakers),
            speaker_labels=speakers,
        )

    def test_no_diarization_gives_none_speaker(self):
        transcription = self._make_transcription([(0.0, 5.0, "Hello world")])
        aligned = align_speakers(transcription, diarization=None)
        assert len(aligned) == 1
        assert aligned[0].speaker is None

    def test_single_speaker_assignment(self):
        transcription = self._make_transcription([(1.0, 4.0, "Let's begin")])
        diarization = self._make_diarization([("SPEAKER_00", 0.0, 5.0)])
        aligned = align_speakers(transcription, diarization)
        assert aligned[0].speaker == "Speaker 1"

    def test_two_speakers(self):
        transcription = self._make_transcription([
            (0.0, 4.0, "Good morning"),
            (4.5, 8.0, "How are you"),
        ])
        diarization = self._make_diarization([
            ("SPEAKER_00", 0.0, 4.2),
            ("SPEAKER_01", 4.3, 9.0),
        ])
        aligned = align_speakers(transcription, diarization)
        assert aligned[0].speaker == "Speaker 1"
        assert aligned[1].speaker == "Speaker 2"

    def test_unmatched_segment_gets_none(self):
        # ASR segment at 20–25s, but diarization only covers 0–10s
        transcription = self._make_transcription([(20.0, 25.0, "Late segment")])
        diarization = self._make_diarization([("SPEAKER_00", 0.0, 10.0)])
        aligned = align_speakers(transcription, diarization)
        assert aligned[0].speaker is None

    def test_majority_speaker_wins(self):
        # ASR segment spans 0–10s
        # SPEAKER_00 covers 0–7s (7s overlap)
        # SPEAKER_01 covers 7–10s (3s overlap)
        # → SPEAKER_00 should win
        transcription = self._make_transcription([(0.0, 10.0, "Mixed overlap segment")])
        diarization = self._make_diarization([
            ("SPEAKER_00", 0.0, 7.0),
            ("SPEAKER_01", 7.0, 10.0),
        ])
        aligned = align_speakers(transcription, diarization)
        assert aligned[0].speaker == "Speaker 1"  # SPEAKER_00 → Speaker 1
