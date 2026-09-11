"""
tests/test_models.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for Pydantic data models (src/models.py).
"""

import pytest
from src.models import (
    AudioMetadata,
    SpeechSegment,
    TranscriptSegment,
    AlignedSegment,
    ActionItem,
    Decision,
    MeetingResult,
    TranscriptionResult,
    WERResult,
    LatencyResult,
)


class TestAudioMetadata:
    def _make(self, **kwargs) -> AudioMetadata:
        defaults = dict(
            file_path="/tmp/test.wav",
            original_file="/tmp/test.wav",
            duration_seconds=120.0,
            sample_rate=16000,
            channels=1,
            file_format="wav",
            file_size_bytes=1_000_000,
        )
        defaults.update(kwargs)
        return AudioMetadata(**defaults)

    def test_duration_formatted(self):
        m = self._make(duration_seconds=3661.0)
        assert m.duration_formatted == "01:01:01"

    def test_file_size_mb(self):
        m = self._make(file_size_bytes=2_097_152)  # 2 MB exactly
        assert m.file_size_mb == 2.0

    def test_short_meeting(self):
        m = self._make(duration_seconds=0.5)
        assert m.duration_formatted == "00:00:00"


class TestSpeechSegment:
    def test_duration(self):
        seg = SpeechSegment(start=1.0, end=5.5)
        assert seg.duration == pytest.approx(4.5)


class TestTranscriptSegment:
    def test_timestamp_str(self):
        seg = TranscriptSegment(start=90.0, end=95.0, text="Hello world")
        assert seg.timestamp_str == "[00:01:30]"


class TestAlignedSegment:
    def test_to_display_line_with_speaker(self):
        seg = AlignedSegment(
            speaker="Speaker 1",
            start=10.0,
            end=14.0,
            text="Let's discuss the deployment.",
        )
        line = seg.to_display_line()
        assert "[00:00:10]" in line
        assert "Speaker 1" in line
        assert "deployment" in line

    def test_to_display_line_no_speaker(self):
        seg = AlignedSegment(start=0.0, end=5.0, text="Good morning.")
        line = seg.to_display_line()
        assert "Good morning." in line


class TestActionItem:
    def test_null_owner_allowed(self):
        item = ActionItem(task="Fix the bug", owner=None, deadline=None)
        assert item.owner is None
        assert item.deadline is None

    def test_confidence_range(self):
        with pytest.raises(Exception):
            ActionItem(task="Fix bug", confidence=1.5)  # out of range


class TestMeetingResult:
    def _make_result(self) -> MeetingResult:
        from src.models import AlignedSegment, TranscriptionResult, TranscriptSegment
        return MeetingResult(
            meeting_id="test_001",
            audio_metadata=AudioMetadata(
                file_path="/tmp/t.wav",
                original_file="/tmp/t.wav",
                duration_seconds=60.0,
                sample_rate=16000,
                channels=1,
                file_format="wav",
                file_size_bytes=500_000,
            ),
            transcription=TranscriptionResult(
                language="en",
                segments=[
                    TranscriptSegment(start=0.0, end=3.0, text="Hello world"),
                    TranscriptSegment(start=3.0, end=6.0, text="Good morning"),
                ],
                full_text="Hello world Good morning",
            ),
            aligned_transcript=[
                AlignedSegment(speaker="Speaker 1", start=0.0, end=3.0, text="Hello world"),
                AlignedSegment(speaker="Speaker 2", start=3.0, end=6.0, text="Good morning"),
            ],
        )

    def test_search_transcript_found(self):
        result = self._make_result()
        matches = result.search_transcript("hello")
        assert len(matches) == 1
        assert matches[0].text == "Hello world"

    def test_search_transcript_not_found(self):
        result = self._make_result()
        matches = result.search_transcript("deployment")
        assert matches == []

    def test_search_case_insensitive(self):
        result = self._make_result()
        matches = result.search_transcript("GOOD")
        assert len(matches) == 1


class TestLatencyResult:
    def test_rtf(self):
        lr = LatencyResult(
            audio_duration_seconds=600.0,
            total_processing_seconds=120.0,
        )
        assert lr.real_time_factor == pytest.approx(0.2)

    def test_rtf_zero_audio(self):
        lr = LatencyResult(
            audio_duration_seconds=0.0,
            total_processing_seconds=5.0,
        )
        assert lr.real_time_factor == 0.0


class TestModelRobustness:
    def test_topic_none_and_defaults(self):
        from src.models import Topic
        t = Topic(topic=None)
        assert t.topic == ""
        t2 = Topic()
        assert t2.topic == ""

    def test_action_item_none_task(self):
        from src.models import ActionItem
        a = ActionItem(task=None)
        assert a.task == ""

    def test_decision_none_decision(self):
        from src.models import Decision
        d = Decision(decision=None)
        assert d.decision == ""
