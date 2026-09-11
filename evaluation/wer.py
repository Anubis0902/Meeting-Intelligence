"""
evaluation/wer.py
─────────────────────────────────────────────────────────────────────────────
Word Error Rate (WER) calculation.

WHAT IS WER?
─────────────
WER measures how accurate the transcription is by comparing it word-by-word
against a reference (ground-truth) transcript.

WER = (Substitutions + Deletions + Insertions) / Total Reference Words

• Substitution: a word was replaced with a wrong word
  Reference:   "good morning"
  Hypothesis:  "good mourning"  ← "mourning" substituted for "morning"

• Deletion: a word in the reference was missed entirely
  Reference:   "we should discuss the deadline"
  Hypothesis:  "we should the deadline"  ← "discuss" was deleted

• Insertion: an extra word was added that wasn't in the reference
  Reference:   "yes I agree"
  Hypothesis:  "yes uh I agree"  ← "uh" was inserted

INTERPRETATION:
───────────────
• WER = 0.0 → perfect transcription
• WER = 0.1 → 10% error rate (typical for clean English speech)
• WER = 0.3 → 30% error rate (acceptable for noisy/accented speech)
• WER > 0.5 → poor accuracy, usually due to noise, wrong language, or wrong model size

WHAT IS RTF?
─────────────
Real-Time Factor = processing_time / audio_duration

RTF < 1.0 means the system is faster than real-time:
  Audio = 10 min, Processing = 2 min → RTF = 0.2 (5× faster than real-time)
RTF > 1.0 means slower than real-time (not suitable for live transcription).
"""

from __future__ import annotations

import logging
import re

import jiwer

from src.models import WERResult

logger = logging.getLogger(__name__)


def normalize_text_for_wer(text: str) -> str:
    """
    Normalise text before WER calculation to avoid penalising trivial differences.

    Operations:
    • Lowercase everything
    • Remove punctuation (WER measures word content, not punctuation)
    • Collapse multiple spaces
    • Strip leading/trailing whitespace

    Why not correct spelling?
    WER is a strict metric — we want to measure what the ASR system
    actually produces, not what it would produce if we further processed it.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text.strip()


def calculate_wer(reference: str, hypothesis: str) -> WERResult:
    """
    Calculate Word Error Rate between a reference and hypothesis transcript.

    Parameters
    ----------
    reference  : ground-truth transcript (what was actually said)
    hypothesis : system-generated transcript (what Whisper produced)

    Returns
    -------
    WERResult with WER and detailed error counts.

    Example
    -------
    >>> result = calculate_wer("hello world", "hello word")
    >>> result.wer
    0.5
    >>> result.substitutions
    1
    """
    ref_normalized = normalize_text_for_wer(reference)
    hyp_normalized = normalize_text_for_wer(hypothesis)

    if not ref_normalized:
        raise ValueError("Reference transcript is empty after normalization.")

    # jiwer v4 API: process_words returns a WordOutput with error counts
    output = jiwer.process_words(ref_normalized, hyp_normalized)
    word_count = len(ref_normalized.split())

    result = WERResult(
        reference=reference,
        hypothesis=hypothesis,
        wer=round(output.wer, 4),
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        reference_word_count=word_count,
    )

    logger.info(
        f"WER: {result.wer:.1%} | "
        f"S={result.substitutions} D={result.deletions} I={result.insertions} | "
        f"Ref words: {word_count}"
    )
    return result


def wer_report(result: WERResult) -> str:
    """Format a WERResult as a human-readable report string."""
    return (
        f"Word Error Rate: {result.wer:.1%}\n"
        f"  Substitutions: {result.substitutions}\n"
        f"  Deletions:     {result.deletions}\n"
        f"  Insertions:    {result.insertions}\n"
        f"  Reference words: {result.reference_word_count}"
    )
