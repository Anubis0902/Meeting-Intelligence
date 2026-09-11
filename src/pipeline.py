"""
src/pipeline.py
─────────────────────────────────────────────────────────────────────────────
Speaker-transcript alignment and the main processing pipeline.

ALIGNMENT ALGORITHM:
─────────────────────
ASR and diarization produce two independent timelines:

  ASR timeline:         |──────────────────|  |──────────────|
                        seg1 (0.0–4.2s)        seg2 (4.3–7.8s)

  Diarization timeline: |──────────────────────────|  |──────────────|
                        SPEAKER_00 (0.0–5.1s)          SPEAKER_01 (5.2–8.0s)

For each ASR segment, we find which diarization segment it overlaps with MOST.

Overlap is computed as the intersection length:
  overlap = min(asr_end, dia_end) - max(asr_start, dia_start)

The speaker with the maximum overlap wins.

EDGE CASES:
───────────
• No overlap found: segment gets speaker=None (shown as "Unknown")
• Multiple diarization speakers overlap equally: first one wins
• ASR segment spans multiple speakers: assigned to the majority speaker

This is a "winner takes all" approach.  It's simple, fast, and works well
for typical meeting audio where speakers rarely overlap.

The pipeline orchestrates all phases and measures timing for each.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.config import settings
from src.models import (
    AlignedSegment,
    AudioMetadata,
    DiarizationResult,
    DiarizationSegment,
    MeetingResult,
    SpeechSegment,
    TranscriptionResult,
)
from src.utils import generate_meeting_id, save_json, seconds_to_hms, timed

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Transcript + Speaker Alignment
# ─────────────────────────────────────────────────────────────────────────────

def _compute_overlap(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    """
    Compute the overlap duration between two time intervals.

    Returns 0 if there is no overlap.

    Example:
        _compute_overlap(1.0, 5.0, 3.0, 7.0) → 2.0
        _compute_overlap(1.0, 3.0, 5.0, 7.0) → 0.0
    """
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    return max(0.0, overlap_end - overlap_start)


def align_speakers(
    transcription: TranscriptionResult,
    diarization: Optional[DiarizationResult],
) -> list[AlignedSegment]:
    """
    Map each ASR segment to the most likely speaker from diarization.

    Parameters
    ----------
    transcription : the cleaned ASR output (segments with timestamps)
    diarization   : pyannote diarization output, or None if unavailable

    Returns
    -------
    List of AlignedSegment — every ASR segment with an optional speaker label.

    When diarization is None, all segments get speaker=None.
    """
    from src.diarization.speaker_diarization import label_to_display_name

    aligned: list[AlignedSegment] = []

    if diarization is None or not diarization.segments:
        logger.info("No diarization available. Transcript will have no speaker labels.")
        for seg in transcription.segments:
            aligned.append(
                AlignedSegment(
                    speaker=None,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    confidence=seg.confidence,
                )
            )
        return aligned

    dia_segs = diarization.segments
    logger.info(
        f"Aligning {len(transcription.segments)} ASR segments with "
        f"{len(dia_segs)} diarization segments..."
    )

    unmatched = 0
    for asr_seg in transcription.segments:
        best_speaker: str | None = None
        best_overlap = 0.0

        for dia_seg in dia_segs:
            overlap = _compute_overlap(
                asr_seg.start, asr_seg.end,
                dia_seg.start, dia_seg.end,
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dia_seg.speaker

        if best_speaker is None:
            unmatched += 1

        display_name = (
            label_to_display_name(best_speaker)
            if best_speaker is not None
            else None
        )

        aligned.append(
            AlignedSegment(
                speaker=display_name,
                start=asr_seg.start,
                end=asr_seg.end,
                text=asr_seg.text,
                confidence=asr_seg.confidence,
            )
        )

    if unmatched:
        logger.warning(
            f"{unmatched} ASR segment(s) had no matching diarization segment. "
            "These will appear without a speaker label."
        )

    logger.info(f"Alignment complete: {len(aligned)} segments aligned.")
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_meeting(
    file_path: str | Path,
    language: Optional[str] = None,
    enable_vad: bool = True,
    enable_diarization: bool = True,
    enable_llm: bool = True,
    num_speakers: Optional[int] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> MeetingResult:
    """
    End-to-end meeting processing pipeline.

    This function orchestrates all pipeline stages in order:
      1. Audio metadata extraction & validation
      2. Audio preprocessing (mono, 16kHz, normalise)
      3. Voice Activity Detection (optional)
      4. Speech-to-Text (Whisper)
      5. Transcript cleaning
      6. Speaker diarization (optional)
      7. Transcript + speaker alignment
      8. LLM meeting analysis (optional)
      9. Save results to disk

    Parameters
    ----------
    file_path          : path to the uploaded audio file
    language           : ISO 639-1 language code or None for auto-detection
    enable_vad         : run VAD before ASR
    enable_diarization : run speaker diarization
    enable_llm         : run LLM meeting analysis
    num_speakers       : hint for diarization (None = auto-detect)
    progress_callback  : callable(message, percent) for UI progress updates

    Returns
    -------
    MeetingResult — the complete structured result

    Graceful degradation:
      • If VAD fails   → continue without it
      • If diarization fails → continue without speaker labels
      • If LLM fails   → continue with transcript only
    """
    def progress(msg: str, pct: int) -> None:
        logger.info(f"[{pct:3d}%] {msg}")
        if progress_callback:
            progress_callback(msg, pct)

    pipeline_start = time.perf_counter()
    file_path = Path(file_path)
    meeting_id = generate_meeting_id(file_path)

    stage_times: dict[str, float] = {}

    # ── Stage 1: Metadata ─────────────────────────────────────────────────────
    progress("Loading and validating audio file...", 5)
    from src.audio.metadata import extract_metadata
    with timed("Metadata extraction") as t:
        metadata = extract_metadata(file_path)
    stage_times["metadata"] = t["elapsed"]

    # ── Stage 2: Preprocessing ────────────────────────────────────────────────
    progress("Preprocessing audio (mono, 16kHz, normalise)...", 15)
    from src.audio.preprocessing import preprocess_audio
    with timed("Preprocessing") as t:
        samples, sr, metadata = preprocess_audio(file_path)
    stage_times["preprocessing"] = t["elapsed"]

    # ── Stage 3: VAD ─────────────────────────────────────────────────────────
    speech_segments: list[SpeechSegment] = []
    if enable_vad:
        progress("Detecting speech segments (VAD)...", 25)
        from src.audio.vad import detect_speech_segments, vad_fallback
        try:
            with timed("VAD") as t:
                speech_segments = detect_speech_segments(samples, sr)
            stage_times["vad"] = t["elapsed"]
        except Exception as exc:
            logger.warning(f"VAD failed: {exc}. Continuing without VAD.")
            speech_segments = vad_fallback(metadata.duration_seconds)
    else:
        from src.audio.vad import vad_fallback
        speech_segments = vad_fallback(metadata.duration_seconds)
        logger.info("VAD disabled. Treating entire audio as speech.")

    # ── Stage 4: ASR ─────────────────────────────────────────────────────────
    progress("Transcribing audio (Whisper)...", 40)
    from src.transcription.asr import transcribe_audio
    with timed("ASR") as t:
        raw_transcription = transcribe_audio(samples, language=language)
    stage_times["asr"] = t["elapsed"]

    # ── Stage 5: Transcript processing ────────────────────────────────────────
    progress("Cleaning transcript...", 55)
    from src.transcription.timestamps import process_transcript
    transcription = process_transcript(raw_transcription)

    # ── Stage 6: Diarization ──────────────────────────────────────────────────
    diarization: Optional[DiarizationResult] = None
    if enable_diarization and settings.diarization_available:
        progress("Identifying speakers (diarization)...", 65)
        from src.diarization.speaker_diarization import diarize_audio
        try:
            with timed("Diarization") as t:
                diarization = diarize_audio(
                    metadata.file_path,
                    num_speakers=num_speakers,
                )
            stage_times["diarization"] = t["elapsed"]
        except Exception as exc:
            logger.warning(
                f"Speaker diarization failed: {exc}. "
                "Continuing without speaker labels."
            )
    elif enable_diarization and not settings.diarization_available:
        logger.warning(
            "Speaker diarization requested but HUGGINGFACE_TOKEN is not set. "
            "Skipping diarization."
        )

    # ── Stage 7: Alignment ────────────────────────────────────────────────────
    progress("Aligning transcript with speaker segments...", 75)
    with timed("Alignment") as t:
        aligned_transcript = align_speakers(transcription, diarization)
    stage_times["alignment"] = t["elapsed"]

    # ── Stage 8: LLM Analysis ─────────────────────────────────────────────────
    summary = None
    llm_error = None
    if enable_llm and settings.llm_available:
        progress("Generating meeting summary and extracting insights...", 85)
        from src.meeting.summarizer import analyze_meeting
        try:
            with timed("LLM analysis") as t:
                summary = analyze_meeting(aligned_transcript)
            stage_times["llm"] = t["elapsed"]
        except Exception as exc:
            llm_error = str(exc)
            logger.error(
                f"LLM analysis failed: {exc}. "
                "Meeting result will contain transcript only."
            )
    elif enable_llm and not settings.llm_available:
        llm_error = "API key not configured"
        logger.warning(
            "LLM analysis requested but OPENAI_API_KEY is not set. Skipping."
        )

    # ── Stage 9: Assemble and save result ─────────────────────────────────────
    progress("Saving results...", 95)
    total_time = time.perf_counter() - pipeline_start

    result = MeetingResult(
        meeting_id=meeting_id,
        audio_metadata=metadata,
        transcription=transcription,
        aligned_transcript=aligned_transcript,
        diarization=diarization,
        summary=summary,
        llm_error=llm_error,
        processing_time_seconds=round(total_time, 2),
    )

    # Save to disk
    output_dir = settings.ensure_output_dir() / meeting_id
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, output_dir / "result.json")
    save_json({"stages": stage_times, "total": total_time}, output_dir / "timing.json")

    progress("Processing complete!", 100)
    logger.info(
        f"Meeting processing complete. "
        f"Total time: {total_time:.1f}s | "
        f"RTF: {total_time / metadata.duration_seconds:.3f}"
    )
    return result
