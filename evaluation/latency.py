"""
evaluation/latency.py
─────────────────────────────────────────────────────────────────────────────
Timing utilities for measuring pipeline performance.
"""

from __future__ import annotations

import logging
from src.models import LatencyResult

logger = logging.getLogger(__name__)


def compute_latency(
    audio_duration: float,
    total_time: float,
    stage_times: dict[str, float] | None = None,
) -> LatencyResult:
    """
    Build a LatencyResult from timing data.

    Parameters
    ----------
    audio_duration : length of the audio file in seconds
    total_time     : wall-clock time from pipeline start to finish (seconds)
    stage_times    : optional dict with per-stage timing (keyed by stage name)

    Returns
    -------
    LatencyResult with RTF computed automatically.
    """
    stage_times = stage_times or {}

    result = LatencyResult(
        audio_duration_seconds=round(audio_duration, 2),
        total_processing_seconds=round(total_time, 2),
        preprocessing_seconds=stage_times.get("preprocessing"),
        vad_seconds=stage_times.get("vad"),
        asr_seconds=stage_times.get("asr"),
        diarization_seconds=stage_times.get("diarization"),
        llm_seconds=stage_times.get("llm"),
    )

    logger.info(
        f"Latency report: RTF={result.real_time_factor:.3f} | "
        f"audio={audio_duration:.1f}s | total={total_time:.1f}s"
    )
    return result


def latency_report(result: LatencyResult) -> str:
    """Format a LatencyResult as a human-readable report."""
    lines = [
        f"Audio Duration:      {result.audio_duration_seconds:.1f}s",
        f"Total Processing:    {result.total_processing_seconds:.1f}s",
        f"Real-Time Factor:    {result.real_time_factor:.3f}",
        f"  (RTF < 1.0 = faster than real-time)",
    ]
    if result.preprocessing_seconds is not None:
        lines.append(f"  Preprocessing:     {result.preprocessing_seconds:.2f}s")
    if result.vad_seconds is not None:
        lines.append(f"  VAD:               {result.vad_seconds:.2f}s")
    if result.asr_seconds is not None:
        lines.append(f"  ASR (Whisper):     {result.asr_seconds:.2f}s")
    if result.diarization_seconds is not None:
        lines.append(f"  Diarization:       {result.diarization_seconds:.2f}s")
    if result.llm_seconds is not None:
        lines.append(f"  LLM Analysis:      {result.llm_seconds:.2f}s")
    return "\n".join(lines)
