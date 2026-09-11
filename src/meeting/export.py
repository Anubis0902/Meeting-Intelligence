"""
src/meeting/export.py
─────────────────────────────────────────────────────────────────────────────
Export meeting results to JSON, TXT, and Markdown.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models import MeetingResult
from src.utils import seconds_to_hms


def to_json(result: MeetingResult) -> str:
    """Serialise the full MeetingResult to a pretty-printed JSON string."""
    return result.model_dump_json(indent=2)


def to_txt(result: MeetingResult) -> str:
    """Export as plain text — transcript + summary side by side."""
    lines: list[str] = []
    meta = result.audio_metadata

    lines += [
        "=" * 60,
        "MEETING REPORT",
        "=" * 60,
        f"Duration : {meta.duration_formatted}",
        f"Language : {result.transcription.language.upper()}",
        f"File     : {Path(meta.original_file).name}",
        "",
    ]

    if result.summary:
        s = result.summary
        if s.title:
            lines += [f"TOPIC: {s.title}", ""]
        lines += ["SUMMARY", "─" * 40, s.summary, ""]

        if s.key_points:
            lines += ["KEY DISCUSSION POINTS", "─" * 40]
            lines += [f"• {p}" for p in s.key_points]
            lines += [""]

        if s.decisions:
            lines += ["DECISIONS", "─" * 40]
            for d in s.decisions:
                ts = f" [{d.timestamp}]" if d.timestamp else ""
                lines.append(f"• {d.decision}{ts}")
            lines += [""]

        if s.action_items:
            lines += ["ACTION ITEMS", "─" * 40]
            for a in s.action_items:
                owner = a.owner or "TBD"
                deadline = a.deadline or "Not specified"
                ts = f" [{a.timestamp}]" if a.timestamp else ""
                lines.append(f"• [{owner}] {a.task} — Due: {deadline}{ts}")
            lines += [""]

        if s.open_questions:
            lines += ["OPEN QUESTIONS", "─" * 40]
            lines += [f"• {q}" for q in s.open_questions]
            lines += [""]

    lines += ["TRANSCRIPT", "─" * 40]
    for seg in result.aligned_transcript:
        lines.append(seg.to_display_line())

    return "\n".join(lines)


def to_markdown(result: MeetingResult) -> str:
    """Export as a professional Markdown report."""
    lines: list[str] = []
    meta = result.audio_metadata
    s = result.summary

    title = (s.title if s and s.title else "Meeting Report")
    lines += [
        f"# {title}",
        "",
        "## Meeting Information",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Duration | {meta.duration_formatted} |",
        f"| Language | {result.transcription.language.upper()} |",
        f"| File | {Path(meta.original_file).name} |",
        f"| Words | {len(result.transcription.full_text.split())} |",
        "",
    ]

    if s:
        if s.participants:
            lines += ["## Participants", ""]
            lines += [f"- {p}" for p in s.participants]
            lines += [""]

        lines += ["## Summary", "", s.summary, ""]

        if s.key_points:
            lines += ["## Key Discussion Points", ""]
            lines += [f"- {p}" for p in s.key_points]
            lines += [""]

        if s.decisions:
            lines += ["## Decisions", ""]
            for d in s.decisions:
                ts = f" *(at {d.timestamp})*" if d.timestamp else ""
                lines.append(f"- {d.decision}{ts}")
            lines += [""]

        if s.action_items:
            lines += [
                "## Action Items",
                "",
                "| Owner | Task | Deadline | Timestamp |",
                "|-------|------|----------|-----------|",
            ]
            for a in s.action_items:
                owner = a.owner or "TBD"
                deadline = a.deadline or "—"
                ts = a.timestamp or "—"
                lines.append(f"| {owner} | {a.task} | {deadline} | {ts} |")
            lines += [""]

        if s.open_questions:
            lines += ["## Open Questions", ""]
            lines += [f"- {q}" for q in s.open_questions]
            lines += [""]

        if s.topics:
            lines += ["## Topics Discussed", ""]
            for t in s.topics:
                timing = ""
                if t.start and t.end:
                    timing = f" ({t.start} – {t.end})"
                lines.append(f"- **{t.topic}**{timing}")
            lines += [""]

    lines += ["## Transcript", ""]
    for seg in result.aligned_transcript:
        lines.append(seg.to_display_line())
        lines.append("")

    return "\n".join(lines)


def save_report(result: MeetingResult, output_dir: str | Path) -> dict[str, Path]:
    """
    Save the meeting report in all three formats.

    Returns a dict mapping format name → file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    json_path = output_dir / "report.json"
    json_path.write_text(to_json(result), encoding="utf-8")
    paths["json"] = json_path

    txt_path = output_dir / "report.txt"
    txt_path.write_text(to_txt(result), encoding="utf-8")
    paths["txt"] = txt_path

    md_path = output_dir / "report.md"
    md_path.write_text(to_markdown(result), encoding="utf-8")
    paths["markdown"] = md_path

    return paths
