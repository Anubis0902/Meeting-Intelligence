"""Audio metadata extraction and file validation."""

from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf
import librosa

from src.models import AudioMetadata
from src.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


def extract_metadata(file_path: str | Path) -> AudioMetadata:
    """
    Extract metadata from an audio file and validate it against configured limits.

    Inspects the audio file header to retrieve duration, sample rate, channels,
    and format without reading full uncompressed audio samples into memory.
    """
    5. Return a populated AudioMetadata model.

    Parameters
    ----------
    file_path : path to the audio file (any supported format)

    Returns
    -------
    AudioMetadata

    Raises
    ------
    FileNotFoundError   if the file doesn't exist
    ValueError          if the format is unsupported or file too large
    """
    path = Path(file_path).resolve()

    # ── 1. Existence check ────────────────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # ── 2. Extension check ────────────────────────────────────────────────────
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # ── 3. File size check ────────────────────────────────────────────────────
    file_size_bytes = path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > settings.max_audio_size_mb:
        raise ValueError(
            f"File too large: {file_size_mb:.1f} MB "
            f"(limit is {settings.max_audio_size_mb} MB)."
        )

    logger.info(f"Loading metadata for: {path.name} ({file_size_mb:.1f} MB)")

    # ── 4. Read audio properties ──────────────────────────────────────────────
    # Use librosa to handle all formats uniformly (it calls ffmpeg for MP3/M4A)
    # duration=True loads only enough to determine duration without decoding all samples
    try:
        duration = librosa.get_duration(path=str(path))
        # soundfile can give us sample rate and channels for WAV/FLAC without decoding
        # For formats soundfile can't open directly, fall back to librosa info
        try:
            sf_info = sf.info(str(path))
            sample_rate = sf_info.samplerate
            channels = sf_info.channels
        except Exception:
            # For MP3/M4A, soundfile may fail — use librosa
            y, sr = librosa.load(str(path), sr=None, mono=False, duration=5.0)
            sample_rate = sr
            channels = 1 if y.ndim == 1 else y.shape[0]
    except Exception as exc:
        raise ValueError(f"Could not read audio file '{path.name}': {exc}") from exc

    # ── 5. Duration warning ───────────────────────────────────────────────────
    duration_minutes = duration / 60
    if duration_minutes > settings.max_duration_minutes:
        logger.warning(
            f"Meeting is {duration_minutes:.1f} minutes long. "
            f"Processing will take longer than usual."
        )
    elif duration < 1.0:
        logger.warning(
            f"Audio is very short ({duration:.1f}s). "
            "Check that the file contains actual speech."
        )

    metadata = AudioMetadata(
        file_path=str(path),
        original_file=str(path),
        duration_seconds=round(duration, 3),
        sample_rate=sample_rate,
        channels=channels,
        file_format=ext.lstrip("."),
        file_size_bytes=file_size_bytes,
    )

    logger.info(
        f"Metadata: duration={metadata.duration_formatted}, "
        f"sample_rate={sample_rate} Hz, channels={channels}, "
        f"format={metadata.file_format.upper()}"
    )
    return metadata


def is_supported_format(file_path: str | Path) -> bool:
    """Return True if the file has a supported audio extension."""
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS
