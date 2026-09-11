"""Word Error Rate (WER) computation and evaluation utilities."""

from __future__ import annotations

import logging
import re

import jiwer

from src.models import WERResult

logger = logging.getLogger(__name__)


def normalize_text_for_wer(text: str) -> str:
    """Normalize text for WER calculation: lowercase, remove punctuation, collapse whitespace."""
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
