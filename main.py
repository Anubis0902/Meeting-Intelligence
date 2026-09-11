"""
main.py
─────────────────────────────────────────────────────────────────────────────
CLI entry point for the Meeting Intelligence System.

Usage:
    python main.py <audio_file> [OPTIONS]

Examples:
    python main.py meeting.mp3
    python main.py meeting.wav --language en --no-diarization
    python main.py meeting.m4a --model small --no-llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils import configure_logging
from src.config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meeting Intelligence System — process a meeting recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio_file",
        help="Path to the meeting audio file (WAV, MP3, M4A, etc.)",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Language code (e.g. en, hi, mr). Default: auto-detect.",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help=f"Whisper model size. Default: {settings.whisper_model_size}",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Voice Activity Detection.",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM meeting analysis (transcript only).",
    )
    parser.add_argument(
        "--speakers", "-s",
        type=int,
        default=None,
        help="Number of speakers (hint for diarization). Default: auto.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory. Default: data/outputs/<meeting_id>/",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import logging
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    # Override model size if specified via CLI
    if args.model:
        settings.whisper_model_size = args.model

    # Warn about missing API keys early
    settings.warn_if_keys_missing()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"ERROR: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Meeting Intelligence System")
    print(f"{'='*60}")
    print(f"  File:     {audio_path.name}")
    print(f"  Language: {args.language or 'auto-detect'}")
    print(f"  Model:    {settings.whisper_model_size}")
    print(f"  VAD:      {'disabled' if args.no_vad else 'enabled'}")
    print(f"  Diarize:  {'disabled' if args.no_diarization else 'enabled'}")
    print(f"  LLM:      {'disabled' if args.no_llm else 'enabled'}")
    print(f"{'='*60}\n")

    from src.pipeline import process_meeting

    def progress_handler(msg: str, pct: int) -> None:
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r[{bar}] {pct:3d}% — {msg}", end="", flush=True)
        if pct == 100:
            print()  # newline after completion

    result = process_meeting(
        file_path=audio_path,
        language=args.language,
        enable_vad=not args.no_vad,
        enable_diarization=not args.no_diarization,
        enable_llm=not args.no_llm,
        num_speakers=args.speakers,
        progress_callback=progress_handler,
    )

    # Export
    from src.meeting.export import save_report
    output_dir = Path(args.output) if args.output else (
        settings.output_dir / result.meeting_id
    )
    saved = save_report(result, output_dir)

    print(f"\n{'─'*60}")
    print("Processing complete!")
    print(f"  Duration:   {result.audio_metadata.duration_formatted}")
    print(f"  Language:   {result.transcription.language.upper()}")
    print(f"  Segments:   {len(result.aligned_transcript)}")
    print(f"  Words:      {len(result.transcription.full_text.split())}")
    if result.processing_time_seconds:
        rtf = result.processing_time_seconds / result.audio_metadata.duration_seconds
        print(f"  Process time: {result.processing_time_seconds:.1f}s (RTF: {rtf:.3f})")
    if result.summary:
        print(f"  Action items:  {len(result.summary.action_items)}")
        print(f"  Decisions:     {len(result.summary.decisions)}")
    print(f"\nOutputs saved to: {output_dir}/")
    for fmt, path in saved.items():
        print(f"  {fmt:8s}: {path.name}")
    print()


if __name__ == "__main__":
    main()
