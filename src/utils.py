"""Shared utility functions: logging, timing, serialization, and string formatting."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger with a readable format.

    Call once at application startup (main.py or app.py).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def timed(label: str) -> Generator[dict[str, float], None, None]:
    """
    Context manager that measures wall-clock time for a block of code.

    Usage:
        with timed("Transcription") as t:
            result = transcribe(audio)
        print(t["elapsed"])   # seconds

    The dict is populated after the block exits.
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["elapsed"] = elapsed
        logger.info(f"{label} completed in {elapsed:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Meeting ID generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_meeting_id(file_path: str | Path) -> str:
    """Generate a unique, deterministic meeting ID formatted as YYYYMMDD_HHMMSS_<stem>_<hash>."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    content = f"{file_path}{timestamp}"
    short_hash = hashlib.md5(content.encode()).hexdigest()[:6]
    stem = Path(file_path).stem[:20]  # Keep filename readable
    return f"{timestamp}_{stem}_{short_hash}"


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp formatting
# ─────────────────────────────────────────────────────────────────────────────

def seconds_to_hms(seconds: float) -> str:
    """
    Convert a float number of seconds to HH:MM:SS string.

    Examples:
        seconds_to_hms(0.0)     → "00:00:00"
        seconds_to_hms(90.5)    → "00:01:30"
        seconds_to_hms(3661.0)  → "01:01:01"
    """
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def hms_to_seconds(hms: str) -> float:
    """
    Convert HH:MM:SS string back to float seconds.

    Examples:
        hms_to_seconds("00:01:30") → 90.0
        hms_to_seconds("01:01:01") → 3661.0
    """
    parts = hms.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS format, got: {hms!r}")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return float(h * 3600 + m * 60 + s)


# ─────────────────────────────────────────────────────────────────────────────
# JSON I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_json(data: Any, path: str | Path) -> None:
    """
    Serialise *data* to a JSON file, creating parent directories as needed.

    Accepts dicts, lists, and Pydantic models (via .model_dump()).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # If it's a Pydantic model, use its own serialiser for proper type handling
    if hasattr(data, "model_dump"):
        payload = data.model_dump(mode="json")
    else:
        payload = data

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.debug(f"Saved JSON → {path}")


def load_json(path: str | Path) -> Any:
    """Load and parse a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    """Count words in a string (split on whitespace)."""
    return len(text.split())


def chunk_text_by_words(text: str, max_words: int) -> list[str]:
    """
    Split *text* into chunks of at most *max_words* words.

    Used before sending a long transcript to the LLM so we never exceed the
    model's context window.

    Chunks are split at word boundaries (never mid-word).
    """
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
    return chunks
