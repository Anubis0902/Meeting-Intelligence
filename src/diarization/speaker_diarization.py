"""
src/diarization/speaker_diarization.py
─────────────────────────────────────────────────────────────────────────────
Speaker diarization using pyannote.audio.

WHAT IS SPEAKER DIARIZATION?
─────────────────────────────
Diarization answers the question: "WHO spoke WHEN?"

It does NOT transcribe speech — that's ASR's job.
It does NOT identify speakers by name — it assigns anonymous labels
(SPEAKER_00, SPEAKER_01, …) based on voice similarity.

DIARIZATION vs SPEAKER IDENTIFICATION:
────────────────────────────────────────
• Diarization: "This segment was spoken by person A, this by person B."
  Labels are arbitrary (SPEAKER_00, SPEAKER_01).

• Speaker Identification: "This voice matches the profile for Alice."
  Requires a pre-built voice database.

We use diarization here.  Names can be assigned manually by the user.

HOW PYANNOTE WORKS (high-level):
─────────────────────────────────
1. Voice Activity Detection: find speech vs non-speech frames.
2. Speaker Embedding: encode each speech frame as a fixed-size vector
   (x-vector or similar) capturing speaker characteristics.
3. Clustering: group frames with similar embeddings → each cluster = one speaker.
4. Postprocessing: merge adjacent same-speaker segments, remove very short ones.

ALIGNMENT WITH ASR:
────────────────────
Pyannote and Whisper produce independent timelines.  The alignment step
(src/diarization/speaker_diarization.py → align_speakers) maps each Whisper
segment to the speaker who was talking during most of that segment.

IMPORTANT LIMITATIONS:
──────────────────────
• Overlapping speech: when two people talk simultaneously, only one label
  is assigned per time window.
• Short utterances: very short turns (<1s) may be merged into the wrong speaker.
• Similar voices: speakers with similar vocal characteristics may be merged.
• Number of speakers: the model can estimate or be given the expected count.

SETUP REQUIREMENTS:
────────────────────
pyannote.audio requires:
1. A HuggingFace account.
2. Accepting the model license at:
   https://huggingface.co/pyannote/speaker-diarization-3.1
3. A HuggingFace API token in HUGGINGFACE_TOKEN.

The application degrades gracefully if diarization is unavailable.
"""

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
    """
    Convert a pyannote speaker label to a user-friendly display name.

    SPEAKER_00 → Speaker 1
    SPEAKER_01 → Speaker 2

    Why not use pyannote labels directly?
    The raw labels are technical identifiers, not meaningful names.
    Using "Speaker 1" is clearer in reports, while being honest that we
    don't know the person's actual name.
    """
    try:
        idx = int(speaker_label.split("_")[-1])
        return f"Speaker {idx + 1}"
    except (ValueError, IndexError):
        return speaker_label  # Return as-is if format is unexpected
