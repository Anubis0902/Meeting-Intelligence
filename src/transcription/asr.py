"""
src/transcription/asr.py
─────────────────────────────────────────────────────────────────────────────
Speech-to-Text using faster-whisper.

WHY FASTER-WHISPER?
────────────────────
faster-whisper is a reimplementation of OpenAI's Whisper model using
CTranslate2, a C++ inference engine optimised for transformers:

  • 2–4× faster than the original openai-whisper on CPU.
  • Lower memory usage (supports int8 quantisation).
  • Same accuracy as the original Whisper.
  • Word-level timestamps available.

HOW WHISPER WORKS (high-level):
─────────────────────────────────
1. Audio → 80-channel mel spectrogram (a 2D frequency-time representation).
2. Encoder (transformer) encodes the spectrogram into rich audio embeddings.
3. Decoder (transformer) autoregressively generates text tokens.
4. Special tokens tell the model the task (transcribe vs. translate) and
   the language.  If no language is given, Whisper runs a short detection
   pass first.
5. Timestamps are inserted as special tokens at the token level, then
   aggregated to segment level.

WHAT AFFECTS TRANSCRIPTION ACCURACY?
───────────────────────────────────────
• Model size: larger models are more accurate but slower.
• Audio quality: background noise, echo, and reverberation hurt accuracy.
• Accent / dialect: Whisper handles many accents well but can struggle with
  strong regional accents.
• Speaking rate: very fast speakers are harder to transcribe.
• Overlapping speech: Whisper is not designed for overlapping speech.
• Language: English accuracy is highest; Indian languages (Hindi, Marathi)
  are supported but accuracy varies.
• Code-switching: switching languages mid-sentence is challenging.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from src.config import settings
from src.models import TranscriptSegment, TranscriptionResult
from src.utils import seconds_to_hms

logger = logging.getLogger(__name__)

# ── Model singleton ────────────────────────────────────────────────────────────
_whisper_model: WhisperModel | None = None


def _detect_device() -> tuple[str, str]:
    """
    Return (device, compute_type) based on available hardware.

    Prefers CUDA GPU when available; falls back to CPU with int8.
    """
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected. Using GPU for ASR.")
            return "cuda", "float16"
    except ImportError:
        pass
    logger.info("No GPU detected. Using CPU for ASR.")
    return "cpu", settings.whisper_compute_type


def load_whisper_model(
    model_size: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> WhisperModel:
    """
    Load the Whisper model and cache it globally.

    The model is ~140 MB for 'base', ~461 MB for 'small'.
    Loading takes 2–5 seconds on the first call; subsequent calls return
    the cached model immediately.

    Parameters
    ----------
    model_size   : Whisper model size.  Defaults to settings.whisper_model_size.
    device       : 'cpu' or 'cuda'.  Auto-detected if None.
    compute_type : CTranslate2 compute type.  Auto-selected based on device if None.
    """
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    size = model_size or settings.whisper_model_size
    if device is None or compute_type is None:
        auto_device, auto_ct = _detect_device()
        device = device or auto_device
        compute_type = compute_type or auto_ct

    logger.info(
        f"Loading Whisper model: {size} | device={device} | compute_type={compute_type}"
    )
    _whisper_model = WhisperModel(
        size,
        device=device,
        compute_type=compute_type,
    )
    logger.info(f"Whisper model '{size}' loaded successfully.")
    return _whisper_model


def transcribe_audio(
    audio: np.ndarray | str | Path,
    language: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = False,
) -> TranscriptionResult:
    """
    Transcribe audio using faster-whisper and return structured results.

    Parameters
    ----------
    audio    : float32 numpy array (16 kHz mono) OR a file path string/Path.
               Passing a numpy array is preferred when you've already preprocessed
               the audio.
    language : ISO 639-1 code (e.g. 'en', 'hi') or None for auto-detection.
               Auto-detection uses Whisper's built-in language ID on the first
               30 seconds of audio.
    beam_size: Beam search width.  5 gives good accuracy; lower is faster but
               less accurate.
    vad_filter: Use faster-whisper's built-in VAD filter (Silero-based).
               Usually False here since we run our own VAD step beforehand.

    Returns
    -------
    TranscriptionResult with structured segments, timestamps, and language info.

    Notes
    ─────
    • Transcription runs synchronously and blocks until complete.
    • Whisper processes audio in 30-second windows internally.
    • Confidence is reported as avg_logprob (log probability), converted to
      a 0–1 scale here for readability.
    """
    model = load_whisper_model()

    if isinstance(audio, (str, Path)):
        logger.info(f"Transcribing file: {Path(audio).name}")
        audio_input = str(audio)
    else:
        logger.info(f"Transcribing audio array: {len(audio) / 16000:.1f}s")
        audio_input = audio

    # ── Run transcription ──────────────────────────────────────────────────────
    segments_generator, info = model.transcribe(
        audio_input,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=False,  # segment-level timestamps are sufficient
    )

    detected_language = info.language
    lang_probability = round(info.language_probability, 3)

    logger.info(
        f"Detected language: {detected_language} "
        f"(confidence: {lang_probability:.1%})"
    )

    # ── Collect segments ───────────────────────────────────────────────────────
    # segments_generator is a lazy iterator — consume it fully
    transcript_segments: list[TranscriptSegment] = []
    full_text_parts: list[str] = []

    for seg in segments_generator:
        # avg_logprob is a negative float (log probability).
        # Convert to a 0–1 confidence: e^logprob maps (-inf, 0] → (0, 1].
        confidence = round(math.exp(seg.avg_logprob), 3) if seg.avg_logprob else None

        text = seg.text.strip()
        if not text:
            continue  # Skip empty segments

        transcript_segments.append(
            TranscriptSegment(
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=text,
                confidence=confidence,
            )
        )
        full_text_parts.append(text)

    full_text = " ".join(full_text_parts)

    result = TranscriptionResult(
        language=detected_language,
        language_probability=lang_probability,
        segments=transcript_segments,
        full_text=full_text,
    )

    logger.info(
        f"Transcription complete: {len(transcript_segments)} segments | "
        f"{len(full_text.split())} words"
    )
    return result
