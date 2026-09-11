"""Transcript formatting, cleaning, and timestamp normalization."""

from __future__ import annotations

import logging
import re

from src.models import TranscriptSegment, TranscriptionResult
from src.utils import seconds_to_hms

logger = logging.getLogger(__name__)

# Minimum number of words for a segment to be kept (avoids single-char artifacts)
MIN_SEGMENT_WORDS = 1


def clean_segment_text(text: str) -> str:
    """
    Clean a single transcript segment's text.

    Operations (in order):
    1. Strip leading/trailing whitespace.
    2. Collapse multiple spaces to one.
    3. Remove double punctuation (e.g., ".. " → ". ").
    4. Ensure the first character is capitalised (Whisper sometimes lowercases
       the first word after timestamps).

    The semantic content is never modified.
    """
    text = text.strip()
    text = re.sub(r" +", " ", text)                  # collapse spaces
    text = re.sub(r"\.{2,}", ".", text)               # collapse ellipsis
    text = re.sub(r"(\. ){2,}", ". ", text)           # repeated ". "
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]             # capitalise first char
    return text


def filter_empty_segments(
    segments: list[TranscriptSegment],
    min_words: int = MIN_SEGMENT_WORDS,
) -> list[TranscriptSegment]:
    """
    Remove segments that are too short to be meaningful.

    A segment with zero or one character after cleaning is almost certainly
    a noise artifact or hallucination artefact (Whisper occasionally outputs
    isolated punctuation on silence).
    """
    filtered: list[TranscriptSegment] = []
    removed = 0
    for seg in segments:
        word_count = len(seg.text.split())
        if word_count >= min_words:
            filtered.append(seg)
        else:
            logger.debug(f"Filtered short segment: {seg.text!r}")
            removed += 1

    if removed:
        logger.info(f"Filtered {removed} short/empty segment(s).")
    return filtered


def process_transcript(raw: TranscriptionResult) -> TranscriptionResult:
    """
    Apply cleaning to every segment in a TranscriptionResult.

    Returns a NEW TranscriptionResult (the original is preserved for export).
    """
    logger.info(f"Processing {len(raw.segments)} raw segments...")

    cleaned_segments: list[TranscriptSegment] = []
    for seg in raw.segments:
        cleaned_text = clean_segment_text(seg.text)
        cleaned_segments.append(
            seg.model_copy(update={"text": cleaned_text})
        )

    cleaned_segments = filter_empty_segments(cleaned_segments)

    full_text = " ".join(s.text for s in cleaned_segments)

    result = TranscriptionResult(
        language=raw.language,
        language_probability=raw.language_probability,
        segments=cleaned_segments,
        full_text=full_text,
    )

    logger.info(
        f"Transcript processing complete: "
        f"{len(cleaned_segments)} segments | "
        f"{len(full_text.split())} words"
    )
    return result


def format_transcript_for_display(segments: list[TranscriptSegment]) -> str:
    """
    Format segments as a readable text block for display or export.

    Example output:
        [00:00:00] Good morning everyone.
        [00:00:04] Let's start with the agenda.
    """
    lines: list[str] = []
    for seg in segments:
        ts = seconds_to_hms(seg.start)
        lines.append(f"[{ts}] {seg.text}")
    return "\n".join(lines)


def format_transcript_for_llm(segments: list[TranscriptSegment]) -> str:
    """
    Format transcript segments for LLM input.

    Includes timestamps so the LLM can extract accurate timestamps for action
    items and decisions.

    Example output:
        [00:00:00] Good morning everyone.
        [00:00:04] Let's start with the agenda.
        [00:00:12] I'll handle the API integration by Friday.
    """
    return format_transcript_for_display(segments)
