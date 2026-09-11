"""
src/models.py
─────────────────────────────────────────────────────────────────────────────
Pydantic data models shared across the entire pipeline.

Why Pydantic?
  • Every model is type-checked at runtime → impossible to pass a string where
    a float is expected and silently corrupt downstream stages.
  • Models are JSON-serialisable out of the box (.model_dump_json()).
  • They act as living documentation — any developer can read these classes and
    immediately understand the data contract between pipeline stages.

Design principles:
  • Keep models flat and easy to serialise.
  • Optional fields default to None rather than empty strings → callers can
    reliably distinguish "unknown" from "empty".
  • No business logic in models — that lives in the processing modules.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Audio
# ─────────────────────────────────────────────────────────────────────────────

class AudioMetadata(BaseModel):
    """Metadata extracted from an audio file before any processing."""

    file_path: str = Field(description="Absolute path to the (possibly converted) audio file.")
    original_file: str = Field(description="Path to the original uploaded file.")
    duration_seconds: float = Field(description="Total duration in seconds.")
    sample_rate: int = Field(description="Sample rate in Hz (e.g. 16000).")
    channels: int = Field(description="Number of audio channels (1=mono, 2=stereo).")
    file_format: str = Field(description="File extension / format (e.g. 'wav', 'mp3').")
    file_size_bytes: int = Field(description="File size in bytes.")

    @property
    def duration_formatted(self) -> str:
        """Return duration as HH:MM:SS string."""
        total = int(self.duration_seconds)
        h, remainder = divmod(total, 3600)
        m, s = divmod(remainder, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def file_size_mb(self) -> float:
        return round(self.file_size_bytes / (1024 * 1024), 2)


class SpeechSegment(BaseModel):
    """A time interval classified as speech by VAD."""

    start: float = Field(description="Segment start time in seconds.")
    end: float = Field(description="Segment end time in seconds.")

    @property
    def duration(self) -> float:
        return self.end - self.start


# ─────────────────────────────────────────────────────────────────────────────
# ASR / Transcription
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptSegment(BaseModel):
    """A single transcribed segment from Whisper — the atomic unit of the transcript."""

    start: float = Field(description="Segment start time in seconds.")
    end: float = Field(description="Segment end time in seconds.")
    text: str = Field(description="Transcribed text for this segment.")
    confidence: Optional[float] = Field(
        default=None,
        description="Average log-probability confidence (0–1). May be None if unavailable.",
    )

    @property
    def timestamp_str(self) -> str:
        """Format as [HH:MM:SS]."""
        def fmt(t: float) -> str:
            total = int(t)
            h, r = divmod(total, 3600)
            m, s = divmod(r, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"[{fmt(self.start)}]"


class TranscriptionResult(BaseModel):
    """Complete ASR output from Whisper."""

    language: str = Field(description="Detected or specified language code (e.g. 'en', 'hi').")
    language_probability: Optional[float] = Field(
        default=None,
        description="Confidence that the detected language is correct (0–1).",
    )
    segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="All transcribed segments with timestamps.",
    )
    full_text: str = Field(
        default="",
        description="Complete transcript as a single string (join of all segment texts).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diarization
# ─────────────────────────────────────────────────────────────────────────────

class DiarizationSegment(BaseModel):
    """A time interval attributed to a single speaker by pyannote."""

    speaker: str = Field(description="Speaker label, e.g. 'SPEAKER_00'.")
    start: float = Field(description="Segment start time in seconds.")
    end: float = Field(description="Segment end time in seconds.")


class DiarizationResult(BaseModel):
    """Output from the speaker diarization stage."""

    segments: list[DiarizationSegment] = Field(default_factory=list)
    num_speakers: int = Field(
        default=0,
        description="Number of unique speakers detected.",
    )
    speaker_labels: list[str] = Field(
        default_factory=list,
        description="Sorted list of unique speaker IDs found.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aligned transcript (ASR + diarization combined)
# ─────────────────────────────────────────────────────────────────────────────

class AlignedSegment(BaseModel):
    """
    An ASR segment with an optional speaker label attached.

    This is the core unit of the final transcript shown to the user.
    """

    speaker: Optional[str] = Field(
        default=None,
        description="Speaker label (e.g. 'Speaker 1'). None if diarization unavailable.",
    )
    start: float
    end: float
    text: str
    confidence: Optional[float] = None

    @property
    def timestamp_str(self) -> str:
        total = int(self.start)
        h, r = divmod(total, 3600)
        m, s = divmod(r, 60)
        return f"[{h:02d}:{m:02d}:{s:02d}]"

    def to_display_line(self) -> str:
        """Format as '[HH:MM:SS] Speaker N: text'."""
        prefix = f"{self.speaker}: " if self.speaker else ""
        return f"{self.timestamp_str} {prefix}{self.text.strip()}"


# ─────────────────────────────────────────────────────────────────────────────
# Meeting Analysis
# ─────────────────────────────────────────────────────────────────────────────

class ActionItem(BaseModel):
    """A concrete task extracted from the meeting transcript."""

    task: str = Field(default="", description="Description of the task to be done.")
    owner: Optional[str] = Field(
        default=None,
        description="Person responsible. Null if not explicitly stated — never hallucinated.",
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Deadline string (e.g. 'Friday', 'next Monday'). Null if not stated.",
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="HH:MM:SS timestamp where this action item was mentioned.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LLM confidence that this is a genuine action item (0–1).",
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Verbatim excerpt from the transcript supporting this extraction.",
    )

    @field_validator("task", mode="before")
    @classmethod
    def _coerce_task(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()


class Decision(BaseModel):
    """An explicit decision made during the meeting."""

    decision: str = Field(default="", description="The decision that was made.")
    timestamp: Optional[str] = Field(
        default=None,
        description="HH:MM:SS timestamp of the decision.",
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Verbatim excerpt from the transcript.",
    )

    @field_validator("decision", mode="before")
    @classmethod
    def _coerce_decision(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()


class Topic(BaseModel):
    """A major discussion topic identified in the meeting."""

    topic: str = Field(default="", description="Topic label / title.")
    start: Optional[str] = Field(
        default=None,
        description="Approximate start timestamp (HH:MM:SS). May be None.",
    )
    end: Optional[str] = Field(
        default=None,
        description="Approximate end timestamp (HH:MM:SS). May be None.",
    )

    @field_validator("topic", mode="before")
    @classmethod
    def _coerce_topic(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()


class MeetingSummary(BaseModel):
    """
    The structured output of the LLM meeting analysis.

    Rule: every field that couldn't be determined from the transcript MUST be
    None or an empty list — never a hallucinated value.
    """

    title: Optional[str] = Field(
        default=None,
        description="Inferred meeting title. Null if not determinable.",
    )
    summary: str = Field(
        default="",
        description="Concise paragraph summarising the meeting.",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Bulleted list of major discussion points.",
    )
    decisions: list[Decision] = Field(
        default_factory=list,
        description="Explicit decisions made.",
    )
    action_items: list[ActionItem] = Field(
        default_factory=list,
        description="Concrete tasks extracted from the meeting.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved questions raised during the meeting.",
    )
    topics: list[Topic] = Field(
        default_factory=list,
        description="Major topics discussed.",
    )
    participants: list[str] = Field(
        default_factory=list,
        description="Speaker labels detected in the meeting.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline result — the top-level object persisted to disk
# ─────────────────────────────────────────────────────────────────────────────

class MeetingResult(BaseModel):
    """
    Complete result of processing a single meeting.

    Saved as data/outputs/<meeting_id>/result.json.
    """

    meeting_id: str = Field(description="Unique identifier (timestamp + filename stem).")
    audio_metadata: AudioMetadata
    transcription: TranscriptionResult
    aligned_transcript: list[AlignedSegment] = Field(default_factory=list)
    diarization: Optional[DiarizationResult] = None
    summary: Optional[MeetingSummary] = None
    llm_error: Optional[str] = None
    processing_time_seconds: Optional[float] = None

    # ── Convenience ───────────────────────────────────────────────────────────
    def get_full_transcript_text(self) -> str:
        """Return the entire aligned transcript as a human-readable string."""
        return "\n".join(seg.to_display_line() for seg in self.aligned_transcript)

    def search_transcript(self, query: str) -> list[AlignedSegment]:
        """
        Case-insensitive search over aligned transcript segments.

        Returns segments whose text contains the query string.
        """
        q = query.lower().strip()
        return [seg for seg in self.aligned_transcript if q in seg.text.lower()]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

class WERResult(BaseModel):
    """Word Error Rate evaluation result for a single audio file."""

    reference: str = Field(description="Reference (ground-truth) transcript.")
    hypothesis: str = Field(description="System-generated transcript.")
    wer: float = Field(description="Word Error Rate (0–1+). 0 = perfect.")
    substitutions: int
    deletions: int
    insertions: int
    reference_word_count: int


class LatencyResult(BaseModel):
    """Processing time measurements for a single pipeline run."""

    audio_duration_seconds: float
    total_processing_seconds: float
    preprocessing_seconds: Optional[float] = None
    vad_seconds: Optional[float] = None
    asr_seconds: Optional[float] = None
    diarization_seconds: Optional[float] = None
    llm_seconds: Optional[float] = None

    @property
    def real_time_factor(self) -> float:
        """
        RTF = processing_time / audio_duration.

        RTF < 1.0 means the system processes faster than real-time.
        """
        if self.audio_duration_seconds == 0:
            return 0.0
        return round(self.total_processing_seconds / self.audio_duration_seconds, 3)
