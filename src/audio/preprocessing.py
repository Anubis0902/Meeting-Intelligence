"""Audio preprocessing: format conversion, resampling to 16kHz mono, and normalization."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

from src.config import settings
from src.models import AudioMetadata

logger = logging.getLogger(__name__)


def load_audio(file_path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return a mono, 16 kHz float32 NumPy array.

    Parameters
    ----------
    file_path : path to audio file (WAV, MP3, M4A, FLAC, etc.)

    Returns
    -------
    (samples, sample_rate)
      samples : np.ndarray of shape (n_samples,), dtype float32, range [-1, 1]
      sample_rate : always settings.target_sample_rate (16000)

    Notes
    -----
    • librosa.load converts to mono by averaging channels and resamples to the
      target rate in a single call.
    • The returned dtype is float32 by default from librosa.
    """
    path = Path(file_path)
    logger.info(f"Loading audio: {path.name}")

    # librosa.load handles MP3/M4A via audioread / soundfile
    # sr=target_sample_rate → resample to 16 kHz on the fly
    # mono=True             → mix stereo/multichannel to mono
    samples, sr = librosa.load(
        str(path),
        sr=settings.target_sample_rate,
        mono=True,
        dtype=np.float32,
    )

    logger.info(
        f"Loaded {len(samples)} samples at {sr} Hz "
        f"({len(samples) / sr:.1f}s)"
    )
    return samples, sr


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """
    Apply peak normalization so the maximum absolute sample amplitude is 1.0.

    Skips normalization if the audio is near-zero to prevent division by zero.
    """
    peak = np.abs(samples).max()
    if peak < 1e-6:
        logger.warning("Audio appears to be silent (peak amplitude near zero).")
        return samples
    normalised = samples / peak
    logger.debug(f"Audio normalised. Peak was {peak:.4f}, now 1.0.")
    return normalised.astype(np.float32)


def preprocess_audio(file_path: str | Path) -> tuple[np.ndarray, int, AudioMetadata]:
    """
    Full preprocessing pipeline for a single audio file.

    Steps
    ─────
    1. Extract metadata (validation happens in metadata.py before calling this)
    2. Load audio → mono, 16 kHz float32
    3. Normalise amplitude
    4. Optionally save the preprocessed WAV for debugging

    Parameters
    ----------
    file_path : path to the original audio file

    Returns
    -------
    (samples, sample_rate, updated_metadata)
    """
    from src.audio.metadata import extract_metadata  # avoid circular import

    path = Path(file_path)

    # Metadata is re-extracted here so the returned object reflects the
    # processed file (same file, but now we know the pipeline used it)
    metadata = extract_metadata(path)

    # Step 1: Load
    samples, sr = load_audio(path)

    # Step 2: Normalise
    samples = normalize_audio(samples)

    # Step 3 (optional): Save processed file for inspection / debugging
    processed_path = _save_processed(samples, sr, path)
    metadata = metadata.model_copy(update={"file_path": str(processed_path)})

    logger.info(
        f"Preprocessing complete. "
        f"Output: {processed_path.name} | "
        f"Shape: {samples.shape} | "
        f"SR: {sr} Hz"
    )
    return samples, sr, metadata


def _save_processed(
    samples: np.ndarray,
    sr: int,
    original_path: Path,
) -> Path:
    """
    Save the preprocessed audio as a 16-bit PCM WAV in data/processed/.

    Returns the path to the saved file.

    This file is used by:
    • VAD (needs a file path for some implementations)
    • The Streamlit player (to play back the normalised audio)
    """
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = original_path.stem
    out_path = output_dir / f"{stem}_processed.wav"

    # soundfile expects float32 with subtype PCM_16 when writing
    # We convert to 16-bit for smaller file size; ASR uses the float32 array directly
    sf.write(str(out_path), samples, sr, subtype="PCM_16")
    logger.debug(f"Saved preprocessed WAV: {out_path}")
    return out_path


def get_audio_duration(samples: np.ndarray, sample_rate: int) -> float:
    """Return audio duration in seconds from a samples array."""
    return len(samples) / sample_rate


def trim_silence(
    samples: np.ndarray,
    sr: int,
    top_db: float = 40.0,
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    *top_db* : the threshold (in dB relative to peak) below which audio is
    considered silent.  40 dB works well for clean recordings; use a higher
    value for noisier environments.

    This is a simple utility — for robust silence removal during the meeting,
    use VAD (src/audio/vad.py) instead.
    """
    trimmed, _ = librosa.effects.trim(samples, top_db=top_db)
    removed = len(samples) - len(trimmed)
    logger.debug(
        f"Trimmed {removed / sr:.2f}s of silence "
        f"({100 * removed / max(len(samples), 1):.1f}% of audio)"
    )
    return trimmed
