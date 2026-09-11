"""
src/audio/vad.py
─────────────────────────────────────────────────────────────────────────────
Voice Activity Detection (VAD) using Silero VAD.

WHAT IS VAD?
─────────────
VAD is a binary classifier: for every short window of audio it decides
"is a human speaking here?" (speech) or "is this silence/noise?" (non-speech).

WHY USE VAD BEFORE TRANSCRIPTION?
───────────────────────────────────
1. Efficiency — Whisper must process every audio frame.  A 60-minute meeting
   might contain 15 minutes of silence (gaps, thinking pauses, background
   noise).  VAD lets us skip those frames entirely, reducing ASR computation.

2. Accuracy — Long silence passages can confuse Whisper into hallucinating
   repeated words or filler text.  VAD removes them before the model sees them.

IMPORTANT CAVEATS:
──────────────────
• False positives (noise labelled as speech): rare environmental sounds, music,
  or HVAC systems may get tagged as speech.  We keep minimum speech durations
  to filter these out.

• False negatives (speech labelled as silence): whispered speech or very quiet
  segments may be dropped.  This is a real limitation — if you notice words
  being missed, try lowering the VAD threshold.

• VAD does NOT improve transcription quality on segments it passes through.
  It only removes non-speech.

IMPLEMENTATION CHOICE — Silero VAD:
────────────────────────────────────
Silero VAD is:
  • Lightweight (~2 MB model, runs fast on CPU)
  • Pre-trained, no tuning required
  • Returns per-second probabilities → easy to threshold
  • Apache 2.0 licensed
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.config import settings
from src.models import SpeechSegment

logger = logging.getLogger(__name__)

# ── Silero VAD model (loaded once, cached globally) ───────────────────────────
_vad_model = None
_vad_utils = None


def _load_vad_model() -> tuple:
    """
    Load the Silero VAD model.

    Tries the silero-vad PyPI package first (cleaner API), then falls back
    to torch.hub (downloads from GitHub on first run).

    The model is cached after the first call — subsequent calls return instantly.
    This is important in Streamlit where module code can re-execute on each
    interaction.
    """
    global _vad_model, _vad_utils

    if _vad_model is not None:
        return _vad_model, _vad_utils

    logger.info("Loading Silero VAD model...")

    # ── Try silero-vad PyPI package (preferred) ───────────────────────────────
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps  # type: ignore
        _vad_model = load_silero_vad()
        # Wrap into the same (model, utils) tuple structure used downstream
        # utils is a 5-tuple: (get_speech_timestamps, _, _, _, _)
        _vad_utils = (get_speech_timestamps, None, None, None, None)
        logger.info("Silero VAD loaded via silero-vad package.")
        return _vad_model, _vad_utils
    except ImportError:
        pass  # Fall through to torch.hub

    # ── Fallback: torch.hub (downloads model from GitHub) ────────────────────
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        _vad_model = model
        _vad_utils = utils
        logger.info("Silero VAD model loaded via torch.hub.")
        return model, utils
    except Exception as exc:
        logger.error(f"Failed to load Silero VAD: {exc}")
        raise RuntimeError(f"Could not load Silero VAD: {exc}") from exc


def detect_speech_segments(
    samples: np.ndarray,
    sample_rate: int,
    threshold: float = 0.5,
    min_speech_duration: float | None = None,
    min_silence_duration: float | None = None,
) -> list[SpeechSegment]:
    """
    Run Silero VAD on audio and return a list of speech segments.

    Parameters
    ----------
    samples : float32 mono audio array at 16 kHz
    sample_rate : must be 16000 (Silero VAD requirement)
    threshold : speech probability threshold (0–1).
        Lower = more sensitive (more false positives).
        Higher = more conservative (may miss quiet speech).
        Default 0.5 is a good starting point.
    min_speech_duration : minimum length of a speech segment to keep (seconds).
        Defaults to settings.vad_min_speech_duration (0.3s).
    min_silence_duration : minimum silence gap to split segments (seconds).
        Defaults to settings.vad_min_silence_duration (0.5s).

    Returns
    -------
    List of SpeechSegment(start, end) in chronological order.

    Notes
    -----
    Silero VAD expects 16 kHz audio.  If you pass a different sample rate
    the results will be incorrect.
    """
    if sample_rate != 16_000:
        raise ValueError(
            f"Silero VAD requires 16000 Hz audio, got {sample_rate} Hz. "
            "Run preprocessing.py first."
        )

    if min_speech_duration is None:
        min_speech_duration = settings.vad_min_speech_duration
    if min_silence_duration is None:
        min_silence_duration = settings.vad_min_silence_duration

    model, utils = _load_vad_model()
    (get_speech_timestamps, _, _, _, _) = utils

    logger.info(
        f"Running VAD | threshold={threshold} | "
        f"min_speech={min_speech_duration}s | min_silence={min_silence_duration}s"
    )

    # Silero VAD needs a torch tensor
    audio_tensor = torch.from_numpy(samples)

    # get_speech_timestamps returns list of {"start": int, "end": int}
    # where start/end are sample indices (not seconds)
    raw_segments = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=int(min_speech_duration * 1000),
        min_silence_duration_ms=int(min_silence_duration * 1000),
        return_seconds=False,  # we convert manually for clarity
    )

    segments: list[SpeechSegment] = []
    for seg in raw_segments:
        start_sec = seg["start"] / sample_rate
        end_sec = seg["end"] / sample_rate
        segments.append(SpeechSegment(start=round(start_sec, 3), end=round(end_sec, 3)))

    total_speech = sum(s.duration for s in segments)
    total_audio = len(samples) / sample_rate
    speech_ratio = total_speech / total_audio if total_audio > 0 else 0

    logger.info(
        f"VAD complete: {len(segments)} segments | "
        f"speech={total_speech:.1f}s / {total_audio:.1f}s "
        f"({speech_ratio:.1%})"
    )

    if not segments:
        logger.warning(
            "VAD found NO speech segments. "
            "Possible causes: very quiet audio, wrong threshold, or pure silence. "
            "Try lowering the threshold or check the audio file."
        )

    return segments


def extract_speech_audio(
    samples: np.ndarray,
    sample_rate: int,
    segments: list[SpeechSegment],
) -> np.ndarray:
    """
    Concatenate only the speech portions of the audio array.

    Returns a new array containing just the frames from the detected speech
    segments.  The timeline is NOT preserved — timestamps from the original
    audio are required to map back.

    Use case: feed this to a model that processes a flat audio stream.
    For models that support timestamps (like Whisper), it is usually better
    to pass the full audio + speech timestamps directly.
    """
    parts: list[np.ndarray] = []
    for seg in segments:
        start_idx = int(seg.start * sample_rate)
        end_idx = int(seg.end * sample_rate)
        parts.append(samples[start_idx:end_idx])

    if not parts:
        logger.warning("No speech segments to extract. Returning empty array.")
        return np.array([], dtype=np.float32)

    concatenated = np.concatenate(parts)
    logger.debug(
        f"Extracted {len(concatenated) / sample_rate:.1f}s of speech "
        f"from {len(segments)} segments."
    )
    return concatenated


def vad_fallback(total_duration: float) -> list[SpeechSegment]:
    """
    Return a single segment covering the entire audio.

    Used when VAD fails or is disabled — ensures the pipeline continues.
    """
    logger.warning("VAD fallback: treating entire audio as speech.")
    return [SpeechSegment(start=0.0, end=total_duration)]
