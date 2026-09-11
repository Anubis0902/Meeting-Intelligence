"""Speaker diarization pipeline using pyannote.audio."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import settings
from src.models import DiarizationResult, DiarizationSegment

logger = logging.getLogger(__name__)

# ── Model singleton ────────────────────────────────────────────────────────────
_pipeline = None


def _load_diarization_pipeline():
    """
    Load the pyannote speaker-diarization-3.1 pipeline.

    Requires HUGGINGFACE_TOKEN to be set.
    Raises RuntimeError if diarization is unavailable.
    """
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    if not settings.diarization_available:
        raise RuntimeError(
            "Speaker diarization unavailable: HUGGINGFACE_TOKEN is not set. "
            "Add it to your .env file and accept the model license at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    try:
        from pyannote.audio import Pipeline
        import torch

        logger.info(
            "Loading pyannote speaker-diarization-3.1 pipeline "
            "(first run downloads ~1 GB model)..."
        )
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.huggingface_token,
        )

        # Move to GPU if available
        try:
            if torch.cuda.is_available():
                _pipeline = _pipeline.to(torch.device("cuda"))
                logger.info("Diarization pipeline moved to GPU.")
        except Exception:
            pass  # GPU move failed, continue on CPU

        logger.info("Pyannote diarization pipeline loaded successfully.")
        return _pipeline

    except ImportError:
        raise RuntimeError(
            "pyannote.audio is not installed. "
            "Run: pip install pyannote.audio"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load diarization pipeline: {exc}") from exc


def diarize_audio(
    audio_path: str | Path,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> DiarizationResult:
    """
    Run speaker diarization on an audio file.

    Parameters
    ----------
    audio_path  : path to the preprocessed WAV file (16 kHz mono).
    num_speakers: exact number of speakers if known (improves accuracy).
    min_speakers: minimum expected speakers (optional hint).
    max_speakers: maximum expected speakers (optional hint).

    Returns
    -------
    DiarizationResult with speaker-labeled time segments.

    The pipeline is given the file path (not a numpy array) because pyannote
    handles its own audio loading internally.
    """
    pipeline = _load_diarization_pipeline()

    audio_path = Path(audio_path)
    logger.info(f"Starting speaker diarization: {audio_path.name}")

    # Build keyword arguments for the pipeline
    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    elif min_speakers is not None or max_speakers is not None:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

    diarization = pipeline(str(audio_path), **kwargs)

    # ── Parse pyannote output ─────────────────────────────────────────────────
    segments: list[DiarizationSegment] = []
    speaker_set: set[str] = set()

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            DiarizationSegment(
                speaker=speaker,
                start=round(turn.start, 3),
                end=round(turn.end, 3),
            )
        )
        speaker_set.add(speaker)

    # Sort by start time (should already be, but guarantee it)
    segments.sort(key=lambda s: s.start)
    speaker_labels = sorted(speaker_set)

    result = DiarizationResult(
        segments=segments,
        num_speakers=len(speaker_labels),
        speaker_labels=speaker_labels,
    )

    logger.info(
        f"Diarization complete: {len(segments)} segments | "
        f"{result.num_speakers} speakers: {speaker_labels}"
    )
    return result


def label_to_display_name(speaker_label: str) -> str:
    """Convert raw technical label (e.g. SPEAKER_00) to human-readable 'Speaker 1'."""
    try:
        idx = int(speaker_label.split("_")[-1])
        return f"Speaker {idx + 1}"
    except (ValueError, IndexError):
        return speaker_label  # Return as-is if format is unexpected
