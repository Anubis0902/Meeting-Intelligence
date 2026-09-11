"""
evaluation/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Run evaluation on a batch of audio files with reference transcripts.

Usage:
    python evaluation/evaluate.py --audio meeting.wav --reference transcript.txt
    python evaluation/evaluate.py --batch data/eval_set.json

The batch JSON format:
    [
      {"audio": "data/raw/meeting1.wav", "reference": "data/raw/meeting1.txt"},
      {"audio": "data/raw/meeting2.mp3", "reference": "data/raw/meeting2.txt"}
    ]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import configure_logging
from evaluation.wer import calculate_wer, wer_report
from evaluation.latency import compute_latency, latency_report


def evaluate_single(audio_path: str, reference_path: str) -> dict:
    """
    Run the pipeline on one audio file and evaluate WER and latency.

    Does NOT run diarization or LLM to keep evaluation fast.
    Use --full to include those stages.
    """
    from src.pipeline import process_meeting

    audio = Path(audio_path)
    ref_text = Path(reference_path).read_text(encoding="utf-8")

    print(f"\n{'─'*50}")
    print(f"Evaluating: {audio.name}")
    print(f"{'─'*50}")

    start = time.perf_counter()
    result = process_meeting(
        file_path=audio,
        enable_diarization=False,
        enable_llm=False,
    )
    elapsed = time.perf_counter() - start

    wer_result = calculate_wer(
        reference=ref_text,
        hypothesis=result.transcription.full_text,
    )

    latency = compute_latency(
        audio_duration=result.audio_metadata.duration_seconds,
        total_time=elapsed,
    )

    report = {
        "file": audio.name,
        "duration_seconds": result.audio_metadata.duration_seconds,
        "wer": wer_result.wer,
        "substitutions": wer_result.substitutions,
        "deletions": wer_result.deletions,
        "insertions": wer_result.insertions,
        "total_processing_seconds": elapsed,
        "rtf": latency.real_time_factor,
    }

    print(wer_report(wer_result))
    print()
    print(latency_report(latency))
    return report


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Evaluate ASR performance")
    parser.add_argument("--audio", help="Path to audio file")
    parser.add_argument("--reference", help="Path to reference transcript text file")
    parser.add_argument("--batch", help="Path to JSON batch file")
    parser.add_argument("--output", default="evaluation_results.json",
                        help="Output JSON file for results")
    args = parser.parse_args()

    results = []

    if args.batch:
        batch = json.loads(Path(args.batch).read_text())
        for item in batch:
            results.append(evaluate_single(item["audio"], item["reference"]))
    elif args.audio and args.reference:
        results.append(evaluate_single(args.audio, args.reference))
    else:
        print("ERROR: Provide --audio and --reference, or --batch")
        parser.print_help()
        sys.exit(1)

    # Summary
    if len(results) > 1:
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_rtf = sum(r["rtf"] for r in results) / len(results)
        print(f"\n{'='*50}")
        print(f"BATCH SUMMARY ({len(results)} files)")
        print(f"  Average WER: {avg_wer:.1%}")
        print(f"  Average RTF: {avg_rtf:.3f}")
        print(f"{'='*50}")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
