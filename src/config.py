"""
src/config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the Meeting Intelligence System.

All settings are loaded from environment variables (via .env) so that:
  • No secrets are ever hard-coded.
  • The application fails loudly when a required key is missing.
  • Users can tweak behaviour without touching source code.

Usage:
    from src.config import settings
    print(settings.whisper_model_size)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Pydantic-Settings automatically reads matching env-var names
    (case-insensitive), so ``OPENAI_API_KEY`` in .env maps to
    ``settings.openai_api_key`` in Python.
    """

    model_config = SettingsConfigDict(
        env_file=".env",           # Load from .env in the project root
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",            # Ignore unknown env vars gracefully
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (optional fallback).",
    )
    # NVIDIA NIM API — uses OpenAI-compatible SDK at a custom base URL
    # Get your key at: https://build.nvidia.com
    nvidia_api_key: str | None = Field(
        default=None,
        description="NVIDIA API key for Llama / NIM models. Primary LLM provider.",
    )
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA NIM API base URL.",
    )
    # Model to use — NVIDIA NIM hosted Llama
    # Options: meta/llama-3.2-11b-vision-instruct (fast, active on NIM)
    #          meta/llama-3.2-90b-vision-instruct (accurate, active on NIM)
    openai_model: str = Field(
        default="meta/llama-3.2-11b-vision-instruct",
        description="LLM model name. NVIDIA NIM or OpenAI model ID.",
    )

    # ── HuggingFace (diarization) ─────────────────────────────────────────────
    huggingface_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("huggingface_token", "huggingfacehub_api_token"),
        description="HuggingFace token required for pyannote speaker diarization.",
    )

    # ── Whisper ASR ───────────────────────────────────────────────────────────
    whisper_model_size: Literal[
        "tiny", "base", "small", "medium", "large-v2", "large-v3"
    ] = Field(
        default="base",
        description="Whisper model size. 'base' works on most laptops; use 'large-v3' for accuracy.",
    )
    whisper_compute_type: Literal["int8", "float16", "float32"] = Field(
        default="int8",
        description="CTranslate2 compute type. 'int8' is fastest on CPU.",
    )

    # ── Audio Validation ──────────────────────────────────────────────────────
    max_audio_size_mb: int = Field(
        default=500,
        ge=1,
        description="Maximum allowed audio file size in megabytes.",
    )
    max_duration_minutes: int = Field(
        default=180,
        ge=1,
        description="Warn if meeting exceeds this duration (minutes).",
    )

    # ── VAD ───────────────────────────────────────────────────────────────────
    vad_min_speech_duration: float = Field(
        default=0.3,
        ge=0.0,
        description="Minimum speech segment duration in seconds.",
    )
    vad_min_silence_duration: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum silence gap required to split speech segments.",
    )

    # ── Output paths ──────────────────────────────────────────────────────────
    output_dir: Path = Field(
        default=Path("data/outputs"),
        description="Directory where processed meeting results are saved.",
    )

    # ── Target audio format for ASR ───────────────────────────────────────────
    target_sample_rate: int = Field(
        default=16_000,
        description="Resample audio to this sample rate before ASR (Whisper expects 16 kHz).",
    )
    target_channels: int = Field(
        default=1,
        description="Convert audio to mono (1 channel) before processing.",
    )

    # ── LLM chunking ─────────────────────────────────────────────────────────
    max_chunk_words: int = Field(
        default=3000,
        description="Max words per transcript chunk before splitting for LLM summarisation.",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────────────────

    @field_validator("output_dir", mode="before")
    @classmethod
    def _coerce_output_dir(cls, v: str | Path) -> Path:
        return Path(v)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience helpers
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def llm_available(self) -> bool:
        """True if either NVIDIA or OpenAI API key is configured."""
        return bool(self.nvidia_api_key or self.openai_api_key)

    @property
    def active_api_key(self) -> str | None:
        """Return the active LLM API key (NVIDIA preferred over OpenAI)."""
        return self.nvidia_api_key or self.openai_api_key

    @property
    def active_base_url(self) -> str | None:
        """Return the base URL: NVIDIA endpoint when using NVIDIA key, else None (OpenAI default)."""
        if self.nvidia_api_key:
            return self.nvidia_base_url
        return None  # OpenAI SDK defaults to api.openai.com

    @property
    def diarization_available(self) -> bool:
        """True if a HuggingFace token is configured."""
        return bool(self.huggingface_token)

    def ensure_output_dir(self) -> Path:
        """Create the output directory if it doesn't exist and return it."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def warn_if_keys_missing(self) -> None:
        """Log warnings for missing optional API keys."""
        if self.nvidia_api_key:
            logger.info(
                f"Using NVIDIA NIM API | model={self.openai_model}"
            )
        elif self.openai_api_key:
            logger.info(
                f"Using OpenAI API | model={self.openai_model}"
            )
        else:
            logger.warning(
                "Neither NVIDIA_API_KEY nor OPENAI_API_KEY is set. "
                "Meeting summarisation and extraction will be unavailable. "
                "Set NVIDIA_API_KEY in your .env file (free at build.nvidia.com)."
            )
        if not self.diarization_available:
            logger.warning(
                "HUGGINGFACE_TOKEN is not set. "
                "Speaker diarization will be unavailable. "
                "Set it in your .env file."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton — import this everywhere
# ─────────────────────────────────────────────────────────────────────────────
settings = Settings()
